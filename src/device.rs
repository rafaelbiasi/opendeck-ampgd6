use data_url::DataUrl;
use image::{
    DynamicImage, GenericImage, ImageBuffer, Rgb, RgbImage, imageops::FilterType,
    load_from_memory_with_format,
};
use mirajazz::{device::Device, error::MirajazzError, state::DeviceStateUpdate};
use openaction::{OUTBOUND_EVENT_MANAGER, SetImageEvent};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

use crate::{
    DEVICE_IMAGE_LOCKS, DEVICES, IMAGE_FLUSH_GENERATIONS, LAST_IMAGE_HASHES, PROFILE_REDRAW_GUARD,
    TOKENS,
    inputs::opendeck_to_device,
    mappings::{
        COL_COUNT, CandidateDevice, ENCODER_COUNT, KEY_COUNT, Kind, ROW_COUNT,
        get_image_format_for_key,
    },
};

const IMAGE_FLUSH_DEBOUNCE: Duration = Duration::from_millis(35);
const PRESSED_IMAGE_FLUSH_WINDOW: Duration = Duration::from_millis(120);
const PROFILE_REDRAW_WINDOW: Duration = Duration::from_millis(1500);

/// Initializes a device and listens for events
pub async fn device_task(candidate: CandidateDevice, token: CancellationToken) {
    log::info!("Running device task for {:?}", candidate);

    // Wrap in a closure so we can use `?` operator
    let device = async || -> Result<Device, MirajazzError> {
        log::info!("Connecting to device...");
        let device = connect(&candidate).await?;
        log::info!("Device connected successfully");

        // Try to set brightness - some devices may not support this command
        log::info!("Setting brightness...");
        if let Err(e) = device.set_brightness(50).await {
            log::warn!(
                "Failed to set brightness (this may be normal for this device): {}",
                e
            );
            // Continue anyway - brightness setting might not be supported
        } else {
            log::info!("Brightness set successfully");
        }

        // Use the native clear once during init to wipe the factory splash/framebuffer.
        // If it fails, fall back to overwriting every key with a black frame.
        log::info!("Clearing all button images...");
        if let Err(e) = device.clear_all_button_images().await {
            log::warn!(
                "Failed to clear all button images with native command, falling back to per-key black frames: {}",
                e
            );

            for position in 0..KEY_COUNT as u8 {
                if let Err(e) = clear_button_with_black_frame(&device, position).await {
                    log::warn!(
                        "Failed to clear button {} during init fallback (this may be normal for this device): {}",
                        position,
                        e
                    );
                    break;
                }
            }
        } else {
            log::info!("Button images cleared successfully");
        }

        // Try to flush - some devices may not need this
        log::info!("Flushing device...");
        if let Err(e) = device.flush().await {
            log::warn!(
                "Failed to flush device (this may be normal for this device): {}",
                e
            );
            // Continue anyway
        } else {
            log::info!("Device flushed successfully");
        }

        Ok(device)
    }()
    .await;

    let device: Device = match device {
        Ok(device) => device,
        Err(err) => {
            handle_error(&candidate.id, err).await;

            log::error!(
                "Had error during device init, finishing device task: {:?}",
                candidate
            );

            return;
        }
    };

    log::info!("Registering device {}", candidate.id);
    if let Some(outbound) = OUTBOUND_EVENT_MANAGER.lock().await.as_mut() {
        outbound
            .register_device(
                candidate.id.clone(),
                candidate.kind.human_name(),
                ROW_COUNT as u8,
                COL_COUNT as u8,
                ENCODER_COUNT as u8,
                0,
            )
            .await
            .unwrap();
    }

    DEVICES.write().await.insert(candidate.id.clone(), device);

    tokio::select! {
        _ = device_events_task(&candidate) => {},
        _ = token.cancelled() => {}
    };

    log::info!("Shutting down device {:?}", candidate);

    if let Some(device) = DEVICES.read().await.get(&candidate.id) {
        device.shutdown().await.ok();
    }

    log::info!("Device task finished for {:?}", candidate);
}

/// Handles errors, returning true if should continue, returning false if an error is fatal
pub async fn handle_error(id: &String, err: MirajazzError) -> bool {
    log::error!("Device {} error: {}", id, err);

    // Some errors are not critical and can be ignored without sending disconnected event
    if matches!(err, MirajazzError::ImageError(_) | MirajazzError::BadData) {
        return true;
    }

    log::info!("Deregistering device {}", id);
    if let Some(outbound) = OUTBOUND_EVENT_MANAGER.lock().await.as_mut() {
        outbound.deregister_device(id.clone()).await.unwrap();
    }

    log::info!("Cancelling tasks for device {}", id);
    if let Some(token) = TOKENS.read().await.get(id) {
        token.cancel();
    }

    log::info!("Removing device {} from the list", id);
    DEVICES.write().await.remove(id);
    IMAGE_FLUSH_GENERATIONS.lock().await.remove(id);
    DEVICE_IMAGE_LOCKS.lock().await.remove(id);
    clear_cached_frames(id).await;

    log::info!("Finished clean-up for {}", id);

    false
}

pub async fn connect(candidate: &CandidateDevice) -> Result<Device, MirajazzError> {
    let result = Device::connect(
        &candidate.dev,
        candidate.kind.protocol_version(),
        KEY_COUNT,
        ENCODER_COUNT,
    )
    .await;

    match result {
        Ok(device) => Ok(device),
        Err(e) => {
            log::error!("Error while connecting to device: {e}");

            Err(e)
        }
    }
}

/// Handles events from device to OpenDeck
async fn device_events_task(candidate: &CandidateDevice) -> Result<(), MirajazzError> {
    log::info!("Connecting to {} for incoming events", candidate.id);

    let devices_lock = DEVICES.read().await;
    let reader = match devices_lock.get(&candidate.id) {
        Some(device) => device.get_reader(crate::inputs::process_input),
        None => return Ok(()),
    };
    drop(devices_lock);

    log::info!("Connected to {} for incoming events", candidate.id);

    log::info!("Reader is ready for {}", candidate.id);

    // Keep normal upstream behavior, but ignore repeated ButtonDown reports for a
    // key that is still physically held. This targets the intermittent extra
    // activation that can happen right after a profile switch redraw.
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    #[derive(Hash, PartialEq, Eq, Clone, Copy)]
    enum EventKey {
        ButtonDown(u8),
        ButtonUp(u8),
        EncoderDown(u8),
        EncoderUp(u8),
        EncoderTwist(u8, i16),
    }

    let mut last_events: HashSet<(EventKey, Instant)> = HashSet::new();
    let mut pressed_buttons: HashSet<u8> = HashSet::new();
    let dedup_window = Duration::from_millis(500); // 500ms window for deduplication

    loop {
        log::info!("Reading updates...");

        let updates = match reader.read(None).await {
            Ok(updates) => updates,
            Err(e) => {
                if !handle_error(&candidate.id, e).await {
                    break;
                }

                continue;
            }
        };

        // Clean up old events from deduplication cache
        let now = Instant::now();
        last_events.retain(|(_, time)| now.duration_since(*time) < dedup_window);

        for update in updates {
            log::info!("New update: {:#?}", update);

            match &update {
                DeviceStateUpdate::ButtonDown(key) => {
                    let now = Instant::now();
                    let mut redraw_guard = PROFILE_REDRAW_GUARD.lock().await;
                    let should_suppress =
                        redraw_guard
                            .suppress_key_until
                            .is_some_and(|(suppressed_key, until)| {
                                suppressed_key == *key && now < until
                            });

                    if should_suppress {
                        log::info!(
                            "Suppressing repeated key_down for key {} during profile redraw",
                            key
                        );
                        continue;
                    }

                    redraw_guard.last_key_down = Some((*key, now));
                    drop(redraw_guard);

                    if !pressed_buttons.insert(*key) {
                        log::debug!(
                            "Skipping repeated button_down while key {} is still held",
                            key
                        );
                        continue;
                    }
                }
                DeviceStateUpdate::ButtonUp(key) => {
                    pressed_buttons.remove(key);

                    let redraw_guard = PROFILE_REDRAW_GUARD.lock().await;
                    if redraw_guard
                        .suppress_key_until
                        .is_some_and(|(suppressed_key, until)| {
                            suppressed_key == *key && Instant::now() < until
                        })
                    {
                        log::info!(
                            "Suppressing paired key_up for key {} during profile redraw",
                            key
                        );
                        continue;
                    }
                }
                _ => {}
            }

            // Create a key for deduplication
            let event_key = match &update {
                DeviceStateUpdate::ButtonDown(key) => EventKey::ButtonDown(*key),
                DeviceStateUpdate::ButtonUp(key) => EventKey::ButtonUp(*key),
                DeviceStateUpdate::EncoderDown(enc) => EventKey::EncoderDown(*enc),
                DeviceStateUpdate::EncoderUp(enc) => EventKey::EncoderUp(*enc),
                DeviceStateUpdate::EncoderTwist(enc, val) => {
                    EventKey::EncoderTwist(*enc, *val as i16)
                }
            };

            // Check for duplicates (same event type and key/encoder within the dedup window)
            let is_duplicate = last_events.iter().any(|(key, _)| *key == event_key);

            if is_duplicate {
                log::debug!("Skipping duplicate event: {:#?}", update);
                continue;
            }

            // Add to deduplication cache
            last_events.insert((event_key, now));

            let id = candidate.id.clone();

            if let Some(outbound) = OUTBOUND_EVENT_MANAGER.lock().await.as_mut() {
                match update {
                    DeviceStateUpdate::ButtonDown(key) => {
                        log::info!("Sending key_down event: device_id={}, key={}", id, key);
                        if let Err(err) = outbound.key_down(id.clone(), key).await {
                            log::warn!(
                                "Failed to send key_down event: device_id={}, key={}, err={}",
                                id,
                                key,
                                err
                            );
                        }
                    }
                    DeviceStateUpdate::ButtonUp(key) => {
                        log::info!("Sending key_up event: device_id={}, key={}", id, key);
                        if let Err(err) = outbound.key_up(id.clone(), key).await {
                            log::warn!(
                                "Failed to send key_up event: device_id={}, key={}, err={}",
                                id,
                                key,
                                err
                            );
                        }
                    }
                    DeviceStateUpdate::EncoderDown(encoder) => {
                        outbound.encoder_down(id, encoder).await.unwrap();
                    }
                    DeviceStateUpdate::EncoderUp(encoder) => {
                        outbound.encoder_up(id, encoder).await.unwrap();
                    }
                    DeviceStateUpdate::EncoderTwist(encoder, val) => {
                        outbound
                            .encoder_change(id, encoder, val as i16)
                            .await
                            .unwrap();
                    }
                }
            }
        }
    }

    Ok(())
}

fn normalize_button_image(image: DynamicImage, width: u32, height: u32) -> DynamicImage {
    // Keep the whole icon visible and center it on a black canvas.
    // This avoids clipping icons that are slightly off-center in the source artwork.
    let resized = image.resize(width, height, FilterType::Triangle).to_rgb8();
    let mut canvas: RgbImage = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));
    let x = (width.saturating_sub(resized.width())) / 2;
    let y = (height.saturating_sub(resized.height())) / 2;

    let _ = canvas.copy_from(&resized, x, y);

    DynamicImage::ImageRgb8(canvas)
}

fn blank_button_image(width: u32, height: u32) -> DynamicImage {
    let blank: RgbImage = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));
    DynamicImage::ImageRgb8(blank)
}

async fn clear_button_with_black_frame(device: &Device, position: u8) -> Result<(), MirajazzError> {
    let kind = Kind::from_vid_pid(device.vid, device.pid).ok_or_else(|| {
        log::error!(
            "Unable to resolve device kind while clearing button: vid={:#06x}, pid={:#06x}, position={}",
            device.vid,
            device.pid,
            position
        );
        MirajazzError::BadData
    })?;
    let format = get_image_format_for_key(&kind, position);
    let image = blank_button_image(format.size.0 as u32, format.size.1 as u32);

    device
        .set_button_image(opendeck_to_device(position), format, image)
        .await
}

fn schedule_debounced_flush(device_id: String) {
    tokio::spawn(async move {
        let generation = {
            let mut generations = IMAGE_FLUSH_GENERATIONS.lock().await;
            let generation = generations.entry(device_id.clone()).or_insert(0);
            *generation += 1;
            *generation
        };

        sleep(IMAGE_FLUSH_DEBOUNCE).await;

        let should_flush = {
            let generations = IMAGE_FLUSH_GENERATIONS.lock().await;
            generations
                .get(&device_id)
                .is_some_and(|current_generation| *current_generation == generation)
        };

        if !should_flush {
            return;
        }

        let image_lock = get_device_image_lock(&device_id).await;
        let _image_guard = image_lock.lock().await;

        let flush_result = {
            let devices = DEVICES.read().await;
            if let Some(device) = devices.get(&device_id) {
                device.flush().await
            } else {
                return;
            }
        };

        if let Err(err) = flush_result {
            handle_error(&device_id, err).await;
        }
    });
}

async fn get_device_image_lock(device_id: &str) -> Arc<Mutex<()>> {
    let mut locks = DEVICE_IMAGE_LOCKS.lock().await;
    locks
        .entry(device_id.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn hash_image_payload(image: &Option<String>) -> Option<u64> {
    image.as_ref().map(|payload| {
        let mut hasher = DefaultHasher::new();
        payload.hash(&mut hasher);
        hasher.finish()
    })
}

async fn is_duplicate_button_frame(device_id: &str, position: u8, hash: Option<u64>) -> bool {
    let hashes = LAST_IMAGE_HASHES.lock().await;
    hashes
        .get(&(device_id.to_string(), position))
        .is_some_and(|last_hash| *last_hash == hash)
}

async fn remember_button_frame(device_id: &str, position: u8, hash: Option<u64>) {
    LAST_IMAGE_HASHES
        .lock()
        .await
        .insert((device_id.to_string(), position), hash);
}

async fn clear_cached_frames(device_id: &str) {
    LAST_IMAGE_HASHES
        .lock()
        .await
        .retain(|(cached_device_id, _), _| cached_device_id != device_id);
}

/// Handles different combinations of "set image" event, including clearing the specific buttons and whole device
pub async fn handle_set_image(device: &Device, evt: SetImageEvent) -> Result<(), MirajazzError> {
    let device_id = evt.device.clone();
    let mut should_batch_flush = false;
    let mut should_flush_immediately = false;
    let image_lock = get_device_image_lock(&device_id).await;
    let _image_guard = image_lock.lock().await;

    if evt.position.is_some() {
        let mut redraw_guard = PROFILE_REDRAW_GUARD.lock().await;
        let now = Instant::now();

        match redraw_guard.burst_started_at {
            Some(started_at) if now.duration_since(started_at) <= Duration::from_millis(400) => {
                redraw_guard.burst_count += 1;
            }
            _ => {
                redraw_guard.burst_started_at = Some(now);
                redraw_guard.burst_count = 1;
            }
        }

        if let Some((last_key, last_pressed_at)) = redraw_guard.last_key_down {
            let since_key_down = now.duration_since(last_pressed_at);
            should_flush_immediately =
                evt.position == Some(last_key) && since_key_down <= PRESSED_IMAGE_FLUSH_WINDOW;
            should_batch_flush =
                since_key_down <= PROFILE_REDRAW_WINDOW && !should_flush_immediately;
        }

        // A rapid burst of slot image updates soon after a key press strongly
        // suggests a page/profile redraw caused by that key. Keep extending the
        // suppression window while the redraw is still active.
        if redraw_guard.burst_count >= 5 {
            if let Some((last_key, last_pressed_at)) = redraw_guard.last_key_down {
                if now.duration_since(last_pressed_at) <= PROFILE_REDRAW_WINDOW {
                    redraw_guard.suppress_key_until =
                        Some((last_key, now + Duration::from_millis(800)));
                }
            }
        }
    }

    match (evt.position, evt.image) {
        (Some(position), Some(image)) => {
            let image_hash = hash_image_payload(&Some(image.clone()));
            if is_duplicate_button_frame(&device_id, position, image_hash).await {
                log::debug!("Skipping duplicate image for button {}", position);
                return Ok(());
            }

            log::info!("Setting image for button {}", position);

            // OpenDeck sends image as a data url, so parse it using a library
            let url = match DataUrl::process(image.as_str()) {
                Ok(url) => url,
                Err(err) => {
                    log::error!(
                        "Received malformed data URL for device {}, button {}: {}",
                        device_id,
                        position,
                        err
                    );
                    return Err(MirajazzError::BadData);
                }
            };
            let (body, _fragment) = match url.decode_to_vec() {
                Ok(decoded) => decoded,
                Err(err) => {
                    log::error!(
                        "Failed to decode data URL for device {}, button {}: {}",
                        device_id,
                        position,
                        err
                    );
                    return Err(MirajazzError::BadData);
                }
            };

            // Allow only image/jpeg mime for now
            if url.mime_type().subtype != "jpeg" {
                log::error!("Incorrect mime type: {}", url.mime_type());

                return Ok(()); // Not a fatal error, enough to just log it
            }

            let image = load_from_memory_with_format(body.as_slice(), image::ImageFormat::Jpeg)?;

            let kind = Kind::from_vid_pid(device.vid, device.pid).ok_or_else(|| {
                log::error!(
                    "Unable to resolve device kind while setting image: vid={:#06x}, pid={:#06x}, device_id={}, button={}",
                    device.vid,
                    device.pid,
                    device_id,
                    position
                );
                MirajazzError::BadData
            })?;
            let format = get_image_format_for_key(&kind, position);
            let image = normalize_button_image(image, format.size.0 as u32, format.size.1 as u32);

            device
                .set_button_image(opendeck_to_device(position), format, image)
                .await?;
            remember_button_frame(&device_id, position, image_hash).await;
            if should_flush_immediately {
                device.flush().await?;
            } else if should_batch_flush {
                schedule_debounced_flush(device_id);
            } else {
                device.flush().await?;
            }
        }
        (Some(position), None) => {
            if is_duplicate_button_frame(&device_id, position, None).await {
                log::debug!("Skipping duplicate clear for button {}", position);
                return Ok(());
            }

            clear_button_with_black_frame(device, position).await?;
            remember_button_frame(&device_id, position, None).await;
            if should_flush_immediately {
                device.flush().await?;
            } else if should_batch_flush {
                schedule_debounced_flush(device_id);
            } else {
                device.flush().await?;
            }
        }
        (None, None) => {
            for position in 0..KEY_COUNT as u8 {
                clear_button_with_black_frame(device, position).await?;
            }
            clear_cached_frames(&device_id).await;
            schedule_debounced_flush(device_id);
        }
        _ => {}
    }

    Ok(())
}

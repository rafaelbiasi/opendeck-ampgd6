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
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    DEVICE_IMAGE_STATES, DEVICES, DeviceImageState, TOKENS, TRACKER,
    inputs::{apply_input_event, decode_input_report, ignore_process_input, opendeck_to_device},
    mappings::{
        COL_COUNT, CandidateDevice, ENCODER_COUNT, KEY_COUNT, Kind, ROW_COUNT,
        get_image_format_for_key,
    },
};

const IMAGE_FLUSH_DEBOUNCE: Duration = Duration::from_millis(15);

/// Initializes a device and listens for events
pub async fn device_task(candidate: CandidateDevice, token: CancellationToken) {
    log::info!("Running device task for {:?}", candidate);
    if candidate.kind.known_singleton_limit() {
        log::info!(
            "Device {} uses a stable synthetic identity; multiple identical D6 units are not supported simultaneously",
            candidate.id
        );
    }
    if !candidate.kind.supports_keepalive() {
        log::debug!("Keepalive is disabled for {}", candidate.id);
    }

    // Wrap in a closure so we can use `?` operator
    let device = async {
        log::info!("Connecting to device...");
        let device = connect(&candidate).await?;
        log::info!("Device connected successfully");

        if candidate.kind.supports_brightness() {
            log::info!("Setting brightness...");
            if let Err(e) = device.set_brightness(50).await {
                log::warn!(
                    "Failed to set brightness (this may be normal for this device): {}",
                    e
                );
            } else {
                log::info!("Brightness set successfully");
            }
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
    }
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
    let mut registered = false;
    if let Some(outbound) = OUTBOUND_EVENT_MANAGER.lock().await.as_mut() {
        match outbound
            .register_device(
                candidate.id.clone(),
                candidate.kind.human_name().to_string(),
                ROW_COUNT as u8,
                COL_COUNT as u8,
                ENCODER_COUNT as u8,
                0,
            )
            .await
        {
            Ok(()) => registered = true,
            Err(err) => log::warn!("Failed to register device {}: {}", candidate.id, err),
        }
    }

    if !registered {
        log::error!(
            "Device {} could not be registered with OpenDeck, aborting device task",
            candidate.id
        );
        device.shutdown().await.ok();
        return;
    }

    DEVICES
        .write()
        .await
        .insert(candidate.id.clone(), Arc::new(device));

    tokio::select! {
        _ = device_events_task(&candidate) => {},
        _ = token.cancelled() => {}
    };

    log::info!("Shutting down device {:?}", candidate);

    if let Some(device) = DEVICES.read().await.get(&candidate.id).cloned() {
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
        if let Err(err) = outbound.deregister_device(id.clone()).await {
            log::warn!("Failed to deregister device {}: {}", id, err);
        }
    }

    cleanup_device_state(id).await;

    log::info!("Finished clean-up for {}", id);

    false
}

pub async fn cleanup_device_state(id: &str) {
    log::info!("Cancelling tasks for device {}", id);
    if let Some(token) = TOKENS.write().await.remove(id) {
        token.cancel();
    }

    log::info!("Removing device {} from the list", id);
    DEVICES.write().await.remove(id);
    if let Some(image_state) = DEVICE_IMAGE_STATES.lock().await.remove(id) {
        image_state.shutdown_token.cancel();
    }
}

pub async fn connect(candidate: &CandidateDevice) -> Result<Device, MirajazzError> {
    let result = Device::connect(
        &candidate.dev,
        candidate.kind.write_protocol_version(),
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

    let device = DEVICES.read().await.get(&candidate.id).cloned();
    let reader = match device {
        Some(device) => device.get_reader(ignore_process_input),
        None => return Ok(()),
    };

    log::info!("Connected to {} for incoming events", candidate.id);
    log::info!(
        "Reader is ready for {} (write pv={}, read pv={})",
        candidate.id,
        candidate.kind.write_protocol_version(),
        candidate.kind.read_protocol_version()
    );

    let mut pressed_buttons = 0u16;

    loop {
        log::debug!("Reading updates...");

        let report = match reader.raw_read_data(512).await {
            Ok(report) => report,
            Err(e) => {
                if !handle_error(&candidate.id, e).await {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }
        };

        let updates = match decode_input_report(&report) {
            Ok(Some(event)) => apply_input_event(&mut pressed_buttons, event),
            Ok(None) => continue,
            Err(e) => {
                if !handle_error(&candidate.id, e).await {
                    break;
                }
                continue;
            }
        };

        for update in updates {
            let id = candidate.id.clone();

            if let Some(outbound) = OUTBOUND_EVENT_MANAGER.lock().await.as_mut() {
                match update {
                    DeviceStateUpdate::ButtonDown(key) => {
                        log::debug!("Sending key_down event: device_id={}, key={}", id, key);
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
                        log::debug!("Sending key_up event: device_id={}, key={}", id, key);
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
                        if let Err(err) = outbound.encoder_down(id, encoder).await {
                            log::warn!("Failed to send encoder_down event: {}", err);
                        }
                    }
                    DeviceStateUpdate::EncoderUp(encoder) => {
                        if let Err(err) = outbound.encoder_up(id, encoder).await {
                            log::warn!("Failed to send encoder_up event: {}", err);
                        }
                    }
                    DeviceStateUpdate::EncoderTwist(encoder, val) => {
                        if let Err(err) = outbound.encoder_change(id, encoder, val as i16).await {
                            log::warn!("Failed to send encoder_change event: {}", err);
                        }
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

async fn schedule_debounced_flush(device_id: String) {
    let image_state = get_device_image_state(&device_id).await;
    if image_state.flush_tx.try_send(()).is_err() {
        log::debug!("Debounced flush already pending for device {}", device_id);
    }
}

async fn debounced_flush_worker(
    device_id: String,
    image_state: Arc<DeviceImageState>,
    mut flush_rx: mpsc::Receiver<()>,
) {
    loop {
        let recv = tokio::select! {
            recv = flush_rx.recv() => recv,
            _ = image_state.shutdown_token.cancelled() => return,
        };

        if recv.is_none() {
            return;
        }

        tokio::time::sleep(IMAGE_FLUSH_DEBOUNCE).await;

        if image_state.shutdown_token.is_cancelled() {
            return;
        }

        while flush_rx.try_recv().is_ok() {}

        let Some(device) = DEVICES.read().await.get(&device_id).cloned() else {
            return;
        };
        let _io_guard = image_state.io_mutex.lock().await;
        let flush_result = device.flush().await;

        if let Err(err) = flush_result {
            handle_error(&device_id, err).await;
        }
    }
}

async fn get_device_image_state(device_id: &str) -> Arc<DeviceImageState> {
    let (state, flush_rx) = {
        let mut states = DEVICE_IMAGE_STATES.lock().await;
        if let Some(state) = states.get(device_id) {
            return state.clone();
        }

        let (flush_tx, flush_rx) = mpsc::channel(1);
        let state = Arc::new(DeviceImageState {
            state_mutex: tokio::sync::Mutex::new(Default::default()),
            io_mutex: tokio::sync::Mutex::new(()),
            flush_tx,
            shutdown_token: CancellationToken::new(),
        });

        states.insert(device_id.to_string(), state.clone());

        (state, flush_rx)
    };

    let tracker = TRACKER.lock().await.clone();
    tracker.spawn(debounced_flush_worker(
        device_id.to_string(),
        state.clone(),
        flush_rx,
    ));

    state
}

fn hash_image_payload(image: Option<&str>) -> Option<u64> {
    image.map(|payload| {
        let mut hasher = DefaultHasher::new();
        payload.hash(&mut hasher);
        hasher.finish()
    })
}

fn validate_button_position(position: u8) -> Result<usize, MirajazzError> {
    let index = position as usize;
    if index < KEY_COUNT {
        Ok(index)
    } else {
        log::warn!(
            "Ignoring out-of-range button position {} (max {})",
            position,
            KEY_COUNT - 1
        );
        Err(MirajazzError::BadData)
    }
}

fn decode_button_image(
    device: &Device,
    device_id: &str,
    position: u8,
    image: &str,
) -> Result<(mirajazz::types::ImageFormat, DynamicImage), MirajazzError> {
    let url = match DataUrl::process(image) {
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

    if url.mime_type().subtype != "jpeg" {
        log::error!("Incorrect mime type: {}", url.mime_type());
        return Err(MirajazzError::BadData);
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

    Ok((format, image))
}

async fn clear_pending_image_hash(
    image_state: &Arc<DeviceImageState>,
    index: usize,
    image_hash: Option<u64>,
) {
    let mut image_state = image_state.state_mutex.lock().await;
    if image_state.pending_image_hashes[index] == image_hash {
        image_state.pending_image_hashes[index] = None;
    }
}

/// Handles different combinations of "set image" event, including clearing the specific buttons and whole device
pub async fn handle_set_image(device: &Device, evt: SetImageEvent) -> Result<(), MirajazzError> {
    let device_id = evt.device.clone();
    let image_state = get_device_image_state(&device_id).await;

    match (evt.position, evt.image) {
        (Some(position), Some(image)) => {
            let index = validate_button_position(position)?;
            let image_hash = hash_image_payload(Some(image.as_str()));
            {
                let mut image_state = image_state.state_mutex.lock().await;
                if image_state.last_image_hashes[index] == image_hash
                    || image_state.pending_image_hashes[index] == image_hash
                {
                    log::debug!("Skipping duplicate image for button {}", position);
                    return Ok(());
                }
                image_state.pending_image_hashes[index] = image_hash;
            }

            let (format, image) =
                match decode_button_image(device, &device_id, position, image.as_str()) {
                    Ok(decoded) => decoded,
                    Err(MirajazzError::BadData) => {
                        clear_pending_image_hash(&image_state, index, image_hash).await;
                        return Ok(());
                    }
                    Err(err) => {
                        clear_pending_image_hash(&image_state, index, image_hash).await;
                        return Err(err);
                    }
                };
            let _io_guard = image_state.io_mutex.lock().await;
            if let Err(err) = device
                .set_button_image(opendeck_to_device(position), format, image)
                .await
            {
                drop(_io_guard);
                clear_pending_image_hash(&image_state, index, image_hash).await;
                return Err(err);
            }
            drop(_io_guard);
            let mut image_state = image_state.state_mutex.lock().await;
            image_state.last_image_hashes[index] = image_hash;
            image_state.pending_image_hashes[index] = None;
            schedule_debounced_flush(device_id).await;
        }
        (Some(position), None) => {
            let index = validate_button_position(position)?;
            {
                let mut image_state = image_state.state_mutex.lock().await;
                if image_state.last_image_hashes[index].is_none()
                    && image_state.pending_image_hashes[index].is_none()
                {
                    log::debug!("Skipping duplicate clear for button {}", position);
                    return Ok(());
                }
                image_state.pending_image_hashes[index] = None;
            }

            let _io_guard = image_state.io_mutex.lock().await;
            clear_button_with_black_frame(device, position).await?;
            drop(_io_guard);
            let mut image_state = image_state.state_mutex.lock().await;
            image_state.last_image_hashes[index] = None;
            image_state.pending_image_hashes[index] = None;
            schedule_debounced_flush(device_id).await;
        }
        (None, None) => {
            let _io_guard = image_state.io_mutex.lock().await;
            if let Err(err) = device.clear_all_button_images().await {
                log::warn!(
                    "Failed to clear all button images natively for device {}, falling back to per-key black frames: {}",
                    device_id,
                    err
                );

                for position in 0..KEY_COUNT as u8 {
                    clear_button_with_black_frame(device, position).await?;
                }
            }
            drop(_io_guard);
            let mut image_state = image_state.state_mutex.lock().await;
            image_state.last_image_hashes.fill(None);
            image_state.pending_image_hashes.fill(None);
            schedule_debounced_flush(device_id).await;
        }
        _ => {}
    }

    Ok(())
}

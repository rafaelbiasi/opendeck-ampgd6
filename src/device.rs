use data_url::DataUrl;
use image::{
    DynamicImage, GenericImage, ImageBuffer, Rgb, RgbImage, imageops::FilterType,
    load_from_memory_with_format,
};
use mirajazz::{device::Device, error::MirajazzError, state::DeviceStateUpdate};
use openaction::{OUTBOUND_EVENT_MANAGER, SetImageEvent};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{
    DEVICE_IMAGE_STATES, DEVICES, DeviceImageState, PendingButtonOp, TOKENS, TRACKER,
    inputs::{apply_input_event, decode_input_report, ignore_process_input, opendeck_to_device},
    mappings::{
        COL_COUNT, CandidateDevice, ENCODER_COUNT, KEY_COUNT, Kind, ROW_COUNT,
        get_image_format_for_key,
    },
};

const IMAGE_FLUSH_DEBOUNCE: Duration = Duration::from_millis(5);
const IMAGE_CACHE_LIMIT: usize = 64;

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
            let kind = resolve_device_kind(&device, &candidate.id)?;

            for position in 0..KEY_COUNT as u8 {
                let format = get_image_format_for_key(&kind, position);
                let black_frame =
                    Arc::new(blank_button_image(format.size.0 as u32, format.size.1 as u32));
                if let Err(e) =
                    clear_button_with_black_frame(&device, position, format, black_frame).await
                {
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

async fn clear_button_with_black_frame(
    device: &Device,
    position: u8,
    format: mirajazz::types::ImageFormat,
    image: Arc<DynamicImage>,
) -> Result<(), MirajazzError> {
    device
        .set_button_image(opendeck_to_device(position), format, (*image).clone())
        .await
}

async fn schedule_debounced_flush(device_id: &str, image_state: &Arc<DeviceImageState>) {
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

fn resolve_device_kind(device: &Device, device_id: &str) -> Result<Kind, MirajazzError> {
    Kind::from_vid_pid(device.vid, device.pid).ok_or_else(|| {
        log::error!(
            "Unable to resolve device kind: vid={:#06x}, pid={:#06x}, device_id={}",
            device.vid,
            device.pid,
            device_id
        );
        MirajazzError::BadData
    })
}

async fn get_device_image_state(
    device: &Device,
    device_id: &str,
) -> Result<Arc<DeviceImageState>, MirajazzError> {
    let (state, flush_rx) = {
        let mut states = DEVICE_IMAGE_STATES.lock().await;
        if let Some(state) = states.get(device_id) {
            return Ok(state.clone());
        }

        let kind = resolve_device_kind(device, device_id)?;
        let button_formats = (0..KEY_COUNT)
            .map(|position| get_image_format_for_key(&kind, position as u8))
            .collect::<Vec<_>>();
        let black_frames = button_formats
            .iter()
            .map(|format| Arc::new(blank_button_image(format.size.0 as u32, format.size.1 as u32)))
            .collect::<Vec<_>>();
        let (flush_tx, flush_rx) = mpsc::channel(1);
        let state = Arc::new(DeviceImageState {
            state_mutex: tokio::sync::Mutex::new(Default::default()),
            io_mutex: tokio::sync::Mutex::new(()),
            button_formats,
            black_frames,
            normalized_image_cache: tokio::sync::Mutex::new(HashMap::new()),
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

    Ok(state)
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
    device_id: &str,
    position: u8,
    format: mirajazz::types::ImageFormat,
    image: &str,
) -> Result<DynamicImage, MirajazzError> {
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
    Ok(normalize_button_image(
        image,
        format.size.0 as u32,
        format.size.1 as u32,
    ))
}

async fn clear_pending_op(
    image_state: &Arc<DeviceImageState>,
    index: usize,
    pending_op: Option<PendingButtonOp>,
) {
    let mut image_state = image_state.state_mutex.lock().await;
    if image_state.pending_ops[index] == pending_op {
        image_state.pending_ops[index] = None;
    }
}

fn should_skip_image_update(
    last_image_hash: Option<u64>,
    pending_op: Option<PendingButtonOp>,
    image_hash: u64,
) -> bool {
    last_image_hash == Some(image_hash) || pending_op == Some(PendingButtonOp::Image(image_hash))
}

fn should_skip_clear(last_image_hash: Option<u64>, pending_op: Option<PendingButtonOp>) -> bool {
    last_image_hash.is_none() && pending_op.is_none_or(|op| op == PendingButtonOp::Clear)
}

async fn get_cached_image(
    image_state: &Arc<DeviceImageState>,
    image_hash: u64,
) -> Option<Arc<DynamicImage>> {
    let cache = image_state.normalized_image_cache.lock().await;
    cache.get(&image_hash).cloned()
}

async fn insert_cached_image(
    image_state: &Arc<DeviceImageState>,
    image_hash: u64,
    image: Arc<DynamicImage>,
) {
    let mut cache = image_state.normalized_image_cache.lock().await;
    if cache.len() >= IMAGE_CACHE_LIMIT && !cache.contains_key(&image_hash) {
        cache.clear();
    }
    cache.insert(image_hash, image);
}

/// Handles different combinations of "set image" event, including clearing the specific buttons and whole device
pub async fn handle_set_image(device: &Device, evt: SetImageEvent) -> Result<(), MirajazzError> {
    let device_id = evt.device.clone();
    let image_state = get_device_image_state(device, &device_id).await?;

    match (evt.position, evt.image) {
        (Some(position), Some(image)) => {
            let index = validate_button_position(position)?;
            let image_hash = hash_image_payload(Some(image.as_str())).ok_or(MirajazzError::BadData)?;
            {
                let mut state_guard = image_state.state_mutex.lock().await;
                if should_skip_image_update(
                    state_guard.last_image_hashes[index],
                    state_guard.pending_ops[index],
                    image_hash,
                ) {
                    log::debug!("Skipping duplicate image for button {}", position);
                    return Ok(());
                }
                state_guard.pending_ops[index] = Some(PendingButtonOp::Image(image_hash));
            }

            let format = image_state.button_formats[index].clone();
            let image = match get_cached_image(&image_state, image_hash).await {
                Some(image) => image,
                None => {
                    let decoded = match decode_button_image(&device_id, position, format.clone(), image.as_str()) {
                        Ok(decoded) => decoded,
                        Err(MirajazzError::BadData) => {
                            clear_pending_op(
                                &image_state,
                                index,
                                Some(PendingButtonOp::Image(image_hash)),
                            )
                            .await;
                            return Ok(());
                        }
                        Err(err) => {
                            clear_pending_op(
                                &image_state,
                                index,
                                Some(PendingButtonOp::Image(image_hash)),
                            )
                            .await;
                            return Err(err);
                        }
                    };
                    let decoded = Arc::new(decoded);
                    insert_cached_image(&image_state, image_hash, decoded.clone()).await;
                    decoded
                }
            };

            {
                let image_state_guard = image_state.state_mutex.lock().await;
                if image_state_guard.pending_ops[index] != Some(PendingButtonOp::Image(image_hash)) {
                    return Ok(());
                }
            }

            let _io_guard = image_state.io_mutex.lock().await;
            {
                let image_state_guard = image_state.state_mutex.lock().await;
                if image_state_guard.pending_ops[index] != Some(PendingButtonOp::Image(image_hash)) {
                    return Ok(());
                }
            }
            if let Err(err) = device
                .set_button_image(opendeck_to_device(position), format, (*image).clone())
                .await
            {
                drop(_io_guard);
                clear_pending_op(&image_state, index, Some(PendingButtonOp::Image(image_hash))).await;
                return Err(err);
            }
            drop(_io_guard);
            let mut state_guard = image_state.state_mutex.lock().await;
            if state_guard.pending_ops[index] == Some(PendingButtonOp::Image(image_hash)) {
                state_guard.last_image_hashes[index] = Some(image_hash);
                state_guard.pending_ops[index] = None;
                drop(state_guard);
                schedule_debounced_flush(&device_id, &image_state).await;
            }
        }
        (Some(position), None) => {
            let index = validate_button_position(position)?;
            {
                let mut state_guard = image_state.state_mutex.lock().await;
                if should_skip_clear(state_guard.last_image_hashes[index], state_guard.pending_ops[index]) {
                    log::debug!("Skipping duplicate clear for button {}", position);
                    return Ok(());
                }
                state_guard.pending_ops[index] = Some(PendingButtonOp::Clear);
            }

            let format = image_state.button_formats[index].clone();
            let black_frame = image_state.black_frames[index].clone();
            {
                let image_state_guard = image_state.state_mutex.lock().await;
                if image_state_guard.pending_ops[index] != Some(PendingButtonOp::Clear) {
                    return Ok(());
                }
            }
            let _io_guard = image_state.io_mutex.lock().await;
            {
                let image_state_guard = image_state.state_mutex.lock().await;
                if image_state_guard.pending_ops[index] != Some(PendingButtonOp::Clear) {
                    return Ok(());
                }
            }
            clear_button_with_black_frame(device, position, format, black_frame).await?;
            drop(_io_guard);
            let mut state_guard = image_state.state_mutex.lock().await;
            if state_guard.pending_ops[index] == Some(PendingButtonOp::Clear) {
                state_guard.last_image_hashes[index] = None;
                state_guard.pending_ops[index] = None;
                drop(state_guard);
                schedule_debounced_flush(&device_id, &image_state).await;
            }
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
                    let index = position as usize;
                    clear_button_with_black_frame(
                        device,
                        position,
                        image_state.button_formats[index].clone(),
                        image_state.black_frames[index].clone(),
                    )
                    .await?;
                }
            }
            drop(_io_guard);
            let mut state_guard = image_state.state_mutex.lock().await;
            state_guard.last_image_hashes.fill(None);
            state_guard.pending_ops.fill(None);
            drop(state_guard);
            schedule_debounced_flush(&device_id, &image_state).await;
        }
        _ => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PendingButtonOp, should_skip_clear, should_skip_image_update};

    #[test]
    fn duplicate_image_update_is_skipped_when_already_applied() {
        assert!(should_skip_image_update(
            Some(42),
            Some(PendingButtonOp::Clear),
            42,
        ));
    }

    #[test]
    fn duplicate_image_update_is_skipped_when_same_image_is_pending() {
        assert!(should_skip_image_update(
            None,
            Some(PendingButtonOp::Image(42)),
            42,
        ));
    }

    #[test]
    fn newer_pending_image_replaces_older_one() {
        assert!(!should_skip_image_update(
            Some(11),
            Some(PendingButtonOp::Image(22)),
            33,
        ));
    }

    #[test]
    fn clear_is_skipped_when_button_is_already_empty() {
        assert!(should_skip_clear(None, None));
    }

    #[test]
    fn clear_is_skipped_when_clear_is_already_pending() {
        assert!(should_skip_clear(None, Some(PendingButtonOp::Clear)));
    }

    #[test]
    fn clear_is_not_skipped_when_an_image_is_pending() {
        assert!(!should_skip_clear(None, Some(PendingButtonOp::Image(7))));
    }
}

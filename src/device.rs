use data_url::DataUrl;
use image::{
    DynamicImage, GenericImage, ImageBuffer, Rgb, RgbImage, imageops::FilterType,
    load_from_memory_with_format,
};
use mirajazz::{
    device::Device, error::MirajazzError, images::convert_image_with_format,
    state::DeviceStateUpdate, types::ImageFormat,
};
use openaction::{OUTBOUND_EVENT_MANAGER, SetImageEvent};
use std::{
    array,
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::{Arc, LazyLock},
    time::Duration,
};
use tokio::{
    sync::{Mutex, mpsc},
    task::spawn_blocking,
};
use tokio_util::sync::CancellationToken;

use crate::{
    DEVICES, TOKENS, TRACKER,
    inputs::{apply_input_event, decode_input_report, ignore_process_input, opendeck_to_device},
    mappings::{
        COL_COUNT, CandidateDevice, ENCODER_COUNT, KEY_COUNT, Kind, ROW_COUNT,
        get_image_format_for_key,
    },
};

const IMAGE_CACHE_LIMIT: usize = 64;
const CLEAR_ALL_BATCH_WINDOW: Duration = Duration::from_millis(8);
const IMAGE_BATCH_WINDOW: Duration = Duration::from_millis(1);
const BUTTON_CORNER_RADIUS: u32 = 16;

static DEVICE_RENDERERS: LazyLock<Mutex<HashMap<String, Arc<DeviceRenderHandle>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Clone, Debug, PartialEq, Eq)]
enum RenderCommand {
    SetImage {
        position: u8,
        image_hash: u64,
        payload: String,
    },
    ClearButton {
        position: u8,
    },
    ClearAll,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BatchedRenderOp {
    Clear,
    Image { image_hash: u64, payload: String },
}

struct DeviceRenderHandle {
    command_tx: mpsc::UnboundedSender<RenderCommand>,
    shutdown_token: CancellationToken,
}

type RenderBatch = [Option<BatchedRenderOp>; KEY_COUNT];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ConvertedImageCacheKey {
    image_hash: u64,
    width: usize,
    height: usize,
    mode: u8,
    rotation: u8,
    mirror: u8,
}

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

        log::info!("Clearing all button images...");
        if let Err(e) =
            clear_all_button_images_for_kind(&device, &candidate.id, &candidate.kind).await
        {
            log::warn!(
                "Failed to clear all button images during init for device {}: {}",
                candidate.id,
                e
            );
        } else {
            log::info!("Button images cleared successfully");
        }

        log::info!("Flushing device...");
        if let Err(e) = device.flush().await {
            log::warn!(
                "Failed to flush device (this may be normal for this device): {}",
                e
            );
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

    if matches!(err, MirajazzError::ImageError(_) | MirajazzError::BadData) {
        return true;
    }

    log::info!("Deregistering device {}", id);
    if let Some(outbound) = OUTBOUND_EVENT_MANAGER.lock().await.as_mut()
        && let Err(err) = outbound.deregister_device(id.clone()).await
    {
        log::warn!("Failed to deregister device {}: {}", id, err);
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

    if let Some(renderer) = DEVICE_RENDERERS.lock().await.remove(id) {
        renderer.shutdown_token.cancel();
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

    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let tracker = TRACKER.lock().await.clone();
    tracker.spawn(input_dispatch_worker(candidate.id.clone(), event_rx));

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
            if event_tx.send(update).is_err() {
                return Ok(());
            }
        }
    }

    Ok(())
}

async fn dispatch_single_update(
    outbound: &mut openaction::OutboundEventManager,
    device_id: &str,
    update: DeviceStateUpdate,
) {
    match update {
        DeviceStateUpdate::ButtonDown(key) => {
            log::debug!(
                "Sending key_down event: device_id={}, key={}",
                device_id,
                key
            );
            if let Err(err) = outbound.key_down(device_id.to_string(), key).await {
                log::warn!(
                    "Failed to send key_down event: device_id={}, key={}, err={}",
                    device_id,
                    key,
                    err
                );
            }
        }
        DeviceStateUpdate::ButtonUp(key) => {
            log::debug!("Sending key_up event: device_id={}, key={}", device_id, key);
            if let Err(err) = outbound.key_up(device_id.to_string(), key).await {
                log::warn!(
                    "Failed to send key_up event: device_id={}, key={}, err={}",
                    device_id,
                    key,
                    err
                );
            }
        }
        DeviceStateUpdate::EncoderDown(encoder) => {
            if let Err(err) = outbound.encoder_down(device_id.to_string(), encoder).await {
                log::warn!("Failed to send encoder_down event: {}", err);
            }
        }
        DeviceStateUpdate::EncoderUp(encoder) => {
            if let Err(err) = outbound.encoder_up(device_id.to_string(), encoder).await {
                log::warn!("Failed to send encoder_up event: {}", err);
            }
        }
        DeviceStateUpdate::EncoderTwist(encoder, val) => {
            if let Err(err) = outbound
                .encoder_change(device_id.to_string(), encoder, val as i16)
                .await
            {
                log::warn!("Failed to send encoder_change event: {}", err);
            }
        }
    }
}

async fn input_dispatch_worker(
    device_id: String,
    mut event_rx: mpsc::UnboundedReceiver<DeviceStateUpdate>,
) {
    while let Some(first) = event_rx.recv().await {
        let mut updates = vec![first];
        while let Ok(more) = event_rx.try_recv() {
            updates.push(more);
        }

        for update in updates {
            if let Some(outbound) = OUTBOUND_EVENT_MANAGER.lock().await.as_mut() {
                dispatch_single_update(outbound, &device_id, update).await;
            }
        }
    }
}

fn normalize_button_image(image: DynamicImage, width: u32, height: u32) -> DynamicImage {
    let resized = image.resize(width, height, FilterType::Triangle).to_rgb8();
    let mut canvas: RgbImage = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));
    let x = (width.saturating_sub(resized.width())) / 2;
    let y = (height.saturating_sub(resized.height())) / 2;

    let _ = canvas.copy_from(&resized, x, y);
    apply_rounded_corners(&mut canvas, BUTTON_CORNER_RADIUS);

    DynamicImage::ImageRgb8(canvas)
}

fn apply_rounded_corners(image: &mut RgbImage, radius: u32) {
    let width = image.width();
    let height = image.height();
    let radius = radius.min(width / 2).min(height / 2);

    if radius == 0 {
        return;
    }

    let black = Rgb([0, 0, 0]);

    for y in 0..height {
        for x in 0..width {
            if pixel_is_outside_rounded_rect(x, y, width, height, radius) {
                *image.get_pixel_mut(x, y) = black;
            }
        }
    }
}

fn pixel_is_outside_rounded_rect(x: u32, y: u32, width: u32, height: u32, radius: u32) -> bool {
    let right_start = width - radius;
    let bottom_start = height - radius;
    let radius_edge = u64::from(radius.saturating_sub(1));
    let radius_squared = u64::from(radius) * u64::from(radius);

    let dx = if x < radius {
        Some(radius_edge - u64::from(x))
    } else if x >= right_start {
        Some(u64::from(x - right_start))
    } else {
        None
    };

    let dy = if y < radius {
        Some(radius_edge - u64::from(y))
    } else if y >= bottom_start {
        Some(u64::from(y - bottom_start))
    } else {
        None
    };

    match (dx, dy) {
        (Some(dx), Some(dy)) => dx * dx + dy * dy >= radius_squared,
        _ => false,
    }
}

fn blank_button_image(width: u32, height: u32) -> DynamicImage {
    let blank: RgbImage = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));
    DynamicImage::ImageRgb8(blank)
}

async fn queue_button_image_data(
    device: &Device,
    position: u8,
    image_data: &[u8],
) -> Result<(), MirajazzError> {
    device
        .write_image(opendeck_to_device(position), image_data)
        .await
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

fn hash_image_payload(image: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    image.hash(&mut hasher);
    hasher.finish()
}

fn image_mode_code(format: ImageFormat) -> u8 {
    match format.mode {
        mirajazz::types::ImageMode::None => 0,
        mirajazz::types::ImageMode::BMP => 1,
        mirajazz::types::ImageMode::JPEG => 2,
    }
}

fn image_rotation_code(format: ImageFormat) -> u8 {
    match format.rotation {
        mirajazz::types::ImageRotation::Rot0 => 0,
        mirajazz::types::ImageRotation::Rot90 => 1,
        mirajazz::types::ImageRotation::Rot180 => 2,
        mirajazz::types::ImageRotation::Rot270 => 3,
    }
}

fn image_mirror_code(format: ImageFormat) -> u8 {
    match format.mirror {
        mirajazz::types::ImageMirroring::None => 0,
        mirajazz::types::ImageMirroring::X => 1,
        mirajazz::types::ImageMirroring::Y => 2,
        mirajazz::types::ImageMirroring::Both => 3,
    }
}

fn converted_image_cache_key(image_hash: u64, format: ImageFormat) -> ConvertedImageCacheKey {
    ConvertedImageCacheKey {
        image_hash,
        width: format.size.0,
        height: format.size.1,
        mode: image_mode_code(format),
        rotation: image_rotation_code(format),
        mirror: image_mirror_code(format),
    }
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
    format: ImageFormat,
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

/// Decodes a data-URL payload and converts it to the device's native image
/// format in a single `spawn_blocking` call, keeping all CPU work off the
/// Tokio runtime.
async fn decode_and_convert_button_image(
    device_id: &str,
    position: u8,
    format: ImageFormat,
    payload: String,
) -> Result<Vec<u8>, MirajazzError> {
    let worker_device_id = device_id.to_string();
    let join_device_id = worker_device_id.clone();

    spawn_blocking(move || {
        let decoded = decode_button_image(&worker_device_id, position, format, &payload)?;
        convert_image_with_format_sync(format, decoded)
    })
    .await
    .map_err(|err| {
        log::error!(
            "Image decode+convert task panicked for device {}, button {}: {}",
            join_device_id,
            position,
            err
        );
        MirajazzError::BadData
    })?
}

/// Synchronous wrapper for mirajazz image conversion, usable inside
/// `spawn_blocking`. The upstream `convert_image_with_format` uses
/// `block_in_place` internally, but we call the same image operations
/// directly here to avoid the extra async layer.
fn convert_image_with_format_sync(
    image_format: ImageFormat,
    image: DynamicImage,
) -> Result<Vec<u8>, MirajazzError> {
    // We can safely call the async version in a blocking context via
    // a lightweight single-threaded runtime, but since mirajazz's impl
    // is actually sync behind block_in_place, we use futures_lite to
    // drive it.
    futures_lite::future::block_on(convert_image_with_format(image_format, image))
        .map_err(MirajazzError::ImageError)
}

fn should_apply_image(last_image_hash: Option<u64>, image_hash: u64) -> bool {
    last_image_hash != Some(image_hash)
}

fn should_apply_clear(last_image_hash: Option<u64>) -> bool {
    last_image_hash.is_some()
}

fn empty_render_batch() -> RenderBatch {
    array::from_fn(|_| None)
}

fn follow_up_batch_window(command: &RenderCommand) -> Option<Duration> {
    match command {
        RenderCommand::ClearAll => Some(CLEAR_ALL_BATCH_WINDOW),
        RenderCommand::SetImage { .. } => Some(IMAGE_BATCH_WINDOW),
        RenderCommand::ClearButton { .. } => None,
    }
}

fn batch_contains_image_updates(batch: &RenderBatch) -> bool {
    batch
        .iter()
        .any(|op| matches!(op, Some(BatchedRenderOp::Image { .. })))
}

fn apply_render_command_to_batch(
    batch: &mut RenderBatch,
    clear_all: &mut bool,
    command: RenderCommand,
) -> Result<(), MirajazzError> {
    match command {
        RenderCommand::SetImage {
            position,
            image_hash,
            payload,
        } => {
            let index = validate_button_position(position)?;
            batch[index] = Some(BatchedRenderOp::Image {
                image_hash,
                payload,
            });
        }
        RenderCommand::ClearButton { position } => {
            let index = validate_button_position(position)?;
            batch[index] = Some(BatchedRenderOp::Clear);
        }
        RenderCommand::ClearAll => {
            *clear_all = true;
            batch.fill(None);
        }
    }

    Ok(())
}

async fn build_render_assets(
    kind: &Kind,
) -> Result<(Vec<ImageFormat>, Vec<Arc<Vec<u8>>>), MirajazzError> {
    let button_formats = (0..KEY_COUNT)
        .map(|position| get_image_format_for_key(kind, position as u8))
        .collect::<Vec<_>>();
    let mut black_frames = Vec::with_capacity(button_formats.len());

    for format in &button_formats {
        let black_frame = blank_button_image(format.size.0 as u32, format.size.1 as u32);
        black_frames.push(Arc::new(
            convert_image_with_format(*format, black_frame).await?,
        ));
    }

    Ok((button_formats, black_frames))
}

async fn clear_all_button_images_with_assets(
    device: &Device,
    device_id: &str,
    black_frames: &[Arc<Vec<u8>>],
) -> Result<(), MirajazzError> {
    if let Err(err) = device.clear_all_button_images().await {
        log::warn!(
            "Failed to clear all button images natively for device {}, falling back to per-key black frames: {}",
            device_id,
            err
        );

        for position in 0..KEY_COUNT as u8 {
            let index = position as usize;
            queue_button_image_data(device, position, black_frames[index].as_slice()).await?;
        }
    }

    Ok(())
}

async fn clear_all_button_images_for_kind(
    device: &Device,
    device_id: &str,
    kind: &Kind,
) -> Result<(), MirajazzError> {
    let (_, black_frames) = build_render_assets(kind).await?;
    clear_all_button_images_with_assets(device, device_id, &black_frames).await
}

async fn remove_renderer_handle(device_id: &str, handle: &Arc<DeviceRenderHandle>) {
    let mut renderers = DEVICE_RENDERERS.lock().await;
    if renderers
        .get(device_id)
        .is_some_and(|existing| Arc::ptr_eq(existing, handle))
    {
        renderers.remove(device_id);
    }
}

async fn get_device_renderer(
    device: &Device,
    device_id: &str,
) -> Result<Arc<DeviceRenderHandle>, MirajazzError> {
    {
        let renderers = DEVICE_RENDERERS.lock().await;
        if let Some(renderer) = renderers.get(device_id) {
            return Ok(renderer.clone());
        }
    }

    let kind = resolve_device_kind(device, device_id)?;
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let handle = Arc::new(DeviceRenderHandle {
        command_tx,
        shutdown_token: CancellationToken::new(),
    });

    {
        let mut renderers = DEVICE_RENDERERS.lock().await;
        if let Some(renderer) = renderers.get(device_id) {
            return Ok(renderer.clone());
        }
        renderers.insert(device_id.to_string(), handle.clone());
    }

    let tracker = TRACKER.lock().await.clone();
    tracker.spawn(render_worker(
        device_id.to_string(),
        kind,
        handle.clone(),
        command_rx,
    ));

    Ok(handle)
}

async fn enqueue_render_command(
    device: &Device,
    device_id: &str,
    command: RenderCommand,
) -> Result<(), MirajazzError> {
    for _ in 0..2 {
        let renderer = get_device_renderer(device, device_id).await?;
        if renderer.command_tx.send(command.clone()).is_ok() {
            return Ok(());
        }
        remove_renderer_handle(device_id, &renderer).await;
    }

    log::warn!("Render queue is unavailable for device {}", device_id);
    Err(MirajazzError::BadData)
}

async fn process_render_batch(
    device: &Device,
    device_id: &str,
    button_formats: &[ImageFormat],
    black_frames: &[Arc<Vec<u8>>],
    converted_image_cache: &mut HashMap<ConvertedImageCacheKey, Arc<Vec<u8>>>,
    last_image_hashes: &mut [Option<u64>; KEY_COUNT],
    clear_all: bool,
    batch: RenderBatch,
) -> Result<(), MirajazzError> {
    let mut writes_to_device = 0usize;
    let queued_ops = batch.iter().filter(|op| op.is_some()).count();

    if clear_all {
        if batch_contains_image_updates(&batch) {
            for index in 0..KEY_COUNT {
                if batch[index].is_some() || !should_apply_clear(last_image_hashes[index]) {
                    continue;
                }

                queue_button_image_data(device, index as u8, black_frames[index].as_slice())
                    .await?;
                last_image_hashes[index] = None;
                writes_to_device += 1;
                tokio::task::yield_now().await;
            }
        } else {
            clear_all_button_images_with_assets(device, device_id, black_frames).await?;
            last_image_hashes.fill(None);
            writes_to_device += 1;
        }
    }

    // Phase 1: Collect all image decode/convert tasks and resolve cache hits.
    // Cache misses are launched in parallel so all CPU work overlaps.
    struct PendingImage {
        index: usize,
        image_hash: u64,
        image_data: Arc<Vec<u8>>,
    }

    let mut pending_clears: Vec<usize> = Vec::new();
    let mut resolved_images: Vec<PendingImage> = Vec::new();
    let mut decode_tasks: tokio::task::JoinSet<Result<PendingImage, (usize, MirajazzError)>> =
        tokio::task::JoinSet::new();

    for (index, op) in batch.into_iter().enumerate() {
        match op {
            Some(BatchedRenderOp::Clear) => {
                if should_apply_clear(last_image_hashes[index]) {
                    pending_clears.push(index);
                }
            }
            Some(BatchedRenderOp::Image {
                image_hash,
                payload,
            }) => {
                if !should_apply_image(last_image_hashes[index], image_hash) {
                    continue;
                }

                let format = button_formats[index];
                let cache_key = converted_image_cache_key(image_hash, format);

                if let Some(cached) = converted_image_cache.get(&cache_key) {
                    // Cache hit — resolve immediately
                    resolved_images.push(PendingImage {
                        index,
                        image_hash,
                        image_data: cached.clone(),
                    });
                } else {
                    // Cache miss — launch decode+convert in parallel
                    let dev_id = device_id.to_string();
                    decode_tasks.spawn(async move {
                        let data = decode_and_convert_button_image(
                            &dev_id,
                            index as u8,
                            format,
                            payload,
                        )
                        .await
                        .map_err(|e| (index, e))?;
                        Ok(PendingImage {
                            index,
                            image_hash,
                            image_data: Arc::new(data),
                        })
                    });
                }
            }
            None => {}
        }
    }

    // Wait for all parallel decode tasks to complete
    while let Some(result) = decode_tasks.join_next().await {
        match result {
            Ok(Ok(pending)) => {
                // Insert into cache
                let format = button_formats[pending.index];
                let cache_key = converted_image_cache_key(pending.image_hash, format);
                if converted_image_cache.len() >= IMAGE_CACHE_LIMIT
                    && !converted_image_cache.contains_key(&cache_key)
                {
                    converted_image_cache.clear();
                }
                converted_image_cache.insert(cache_key, pending.image_data.clone());
                resolved_images.push(pending);
            }
            Ok(Err((_, MirajazzError::BadData | MirajazzError::ImageError(_)))) => {
                // Non-fatal image error, skip this button
            }
            Ok(Err((_, err))) => return Err(err),
            Err(join_err) => {
                log::error!("Image decode task panic: {}", join_err);
            }
        }
    }

    // Sort resolved images by index for deterministic device write order
    resolved_images.sort_by_key(|p| p.index);

    // Phase 2: Write to device sequentially (required by USB/HID protocol)
    for index in pending_clears {
        queue_button_image_data(device, index as u8, black_frames[index].as_slice()).await?;
        last_image_hashes[index] = None;
        writes_to_device += 1;
        tokio::task::yield_now().await;
    }

    for pending in resolved_images {
        queue_button_image_data(device, pending.index as u8, pending.image_data.as_slice())
            .await?;
        last_image_hashes[pending.index] = Some(pending.image_hash);
        writes_to_device += 1;
        tokio::task::yield_now().await;
    }

    log::debug!(
        "Processed render batch: device_id={}, clear_all={}, queued_ops={}, writes={}",
        device_id,
        clear_all,
        queued_ops,
        writes_to_device
    );

    if writes_to_device > 0 {
        device.flush().await?;
    }

    Ok(())
}

async fn render_worker(
    device_id: String,
    kind: Kind,
    handle: Arc<DeviceRenderHandle>,
    mut command_rx: mpsc::UnboundedReceiver<RenderCommand>,
) {
    let (button_formats, black_frames) = match build_render_assets(&kind).await {
        Ok(assets) => assets,
        Err(err) => {
            handle_error(&device_id, err).await;
            remove_renderer_handle(&device_id, &handle).await;
            return;
        }
    };
    let mut converted_image_cache = HashMap::new();
    let mut last_image_hashes = [None; KEY_COUNT];

    loop {
        let recv = tokio::select! {
            recv = command_rx.recv() => recv,
            _ = handle.shutdown_token.cancelled() => None,
        };

        let Some(command) = recv else {
            break;
        };

        let mut batch = empty_render_batch();
        let mut clear_all = false;

        if let Some(window) = follow_up_batch_window(&command) {
            // Page switches may arrive either as `ClearAll + SetImage...` or as a burst of
            // `SetImage` updates only. A short idle window lets us collapse both patterns into
            // a single flush.
            tokio::select! {
                _ = tokio::time::sleep(window) => {},
                _ = handle.shutdown_token.cancelled() => break,
            }
        }

        if let Err(err) = apply_render_command_to_batch(&mut batch, &mut clear_all, command) {
            if !handle_error(&device_id, err).await {
                break;
            }
            continue;
        }

        while let Ok(command) = command_rx.try_recv() {
            if let Err(err) = apply_render_command_to_batch(&mut batch, &mut clear_all, command)
                && !handle_error(&device_id, err).await
            {
                remove_renderer_handle(&device_id, &handle).await;
                return;
            }
        }

        let Some(device) = DEVICES.read().await.get(&device_id).cloned() else {
            break;
        };

        log::debug!(
            "Rendering batch: device_id={}, clear_all={}, ops={}, image_updates={}",
            device_id,
            clear_all,
            batch.iter().filter(|op| op.is_some()).count(),
            batch_contains_image_updates(&batch)
        );

        if let Err(err) = process_render_batch(
            device.as_ref(),
            &device_id,
            &button_formats,
            &black_frames,
            &mut converted_image_cache,
            &mut last_image_hashes,
            clear_all,
            batch,
        )
        .await
            && !handle_error(&device_id, err).await
        {
            break;
        }

        // Yield to let the input reader task process any pending HID reports
        // that arrived while the device was busy with image writes.
        tokio::task::yield_now().await;
    }

    remove_renderer_handle(&device_id, &handle).await;
}

/// Handles different combinations of "set image" event, including clearing the specific buttons and whole device
pub async fn handle_set_image(device: &Device, evt: SetImageEvent) -> Result<(), MirajazzError> {
    let device_id = evt.device.clone();

    match (evt.position, evt.image) {
        (Some(position), Some(image)) => {
            validate_button_position(position)?;
            enqueue_render_command(
                device,
                &device_id,
                RenderCommand::SetImage {
                    position,
                    image_hash: hash_image_payload(image.as_str()),
                    payload: image,
                },
            )
            .await?;
        }
        (Some(position), None) => {
            validate_button_position(position)?;
            enqueue_render_command(device, &device_id, RenderCommand::ClearButton { position })
                .await?;
        }
        (None, None) => {
            enqueue_render_command(device, &device_id, RenderCommand::ClearAll).await?;
        }
        _ => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
    use mirajazz::types::{ImageMirroring, ImageMode, ImageRotation};

    use crate::mappings::{KEY_COUNT, Kind};

    use super::{
        BatchedRenderOp, BUTTON_CORNER_RADIUS, CLEAR_ALL_BATCH_WINDOW, IMAGE_BATCH_WINDOW,
        RenderCommand, apply_render_command_to_batch, batch_contains_image_updates,
        build_render_assets, converted_image_cache_key, empty_render_batch,
        follow_up_batch_window, normalize_button_image, should_apply_clear, should_apply_image,
    };

    #[test]
    fn duplicate_applied_image_is_skipped() {
        assert!(!should_apply_image(Some(42), 42));
    }

    #[test]
    fn newer_image_is_applied() {
        assert!(should_apply_image(Some(11), 33));
    }

    #[test]
    fn clear_is_skipped_when_button_is_already_empty() {
        assert!(!should_apply_clear(None));
    }

    #[test]
    fn clear_is_applied_when_button_has_content() {
        assert!(should_apply_clear(Some(7)));
    }

    #[test]
    fn latest_command_wins_for_same_button() {
        let mut batch = empty_render_batch();
        let mut clear_all = false;

        apply_render_command_to_batch(
            &mut batch,
            &mut clear_all,
            RenderCommand::SetImage {
                position: 2,
                image_hash: 10,
                payload: "a".to_string(),
            },
        )
        .unwrap();
        apply_render_command_to_batch(
            &mut batch,
            &mut clear_all,
            RenderCommand::ClearButton { position: 2 },
        )
        .unwrap();

        assert!(!clear_all);
        assert_eq!(batch[2], Some(BatchedRenderOp::Clear));
    }

    #[test]
    fn clear_all_discards_previous_button_batch() {
        let mut batch = empty_render_batch();
        let mut clear_all = false;

        apply_render_command_to_batch(
            &mut batch,
            &mut clear_all,
            RenderCommand::SetImage {
                position: 1,
                image_hash: 10,
                payload: "a".to_string(),
            },
        )
        .unwrap();
        apply_render_command_to_batch(&mut batch, &mut clear_all, RenderCommand::ClearAll).unwrap();

        assert!(clear_all);
        assert!(batch.iter().all(Option::is_none));
    }

    #[test]
    fn updates_after_clear_all_are_preserved() {
        let mut batch = empty_render_batch();
        let mut clear_all = false;

        apply_render_command_to_batch(&mut batch, &mut clear_all, RenderCommand::ClearAll).unwrap();
        apply_render_command_to_batch(
            &mut batch,
            &mut clear_all,
            RenderCommand::SetImage {
                position: 4,
                image_hash: 99,
                payload: "b".to_string(),
            },
        )
        .unwrap();

        assert!(clear_all);
        assert_eq!(
            batch[4],
            Some(BatchedRenderOp::Image {
                image_hash: 99,
                payload: "b".to_string(),
            })
        );
    }

    #[test]
    fn clear_all_waits_for_follow_up_batch() {
        assert_eq!(
            follow_up_batch_window(&RenderCommand::ClearAll),
            Some(CLEAR_ALL_BATCH_WINDOW)
        );
    }

    #[test]
    fn regular_image_update_waits_for_follow_up_batch() {
        assert_eq!(
            follow_up_batch_window(&RenderCommand::SetImage {
                position: 0,
                image_hash: 1,
                payload: "a".to_string(),
            }),
            Some(IMAGE_BATCH_WINDOW)
        );
    }

    #[test]
    fn clear_button_does_not_wait_for_follow_up_batch() {
        assert_eq!(
            follow_up_batch_window(&RenderCommand::ClearButton { position: 0 }),
            Option::<Duration>::None
        );
    }

    #[test]
    fn converted_image_cache_key_includes_format() {
        let jpeg = mirajazz::types::ImageFormat {
            mode: ImageMode::JPEG,
            size: (100, 100),
            rotation: ImageRotation::Rot180,
            mirror: ImageMirroring::None,
        };
        let bmp = mirajazz::types::ImageFormat {
            mode: ImageMode::BMP,
            size: (100, 100),
            rotation: ImageRotation::Rot180,
            mirror: ImageMirroring::None,
        };

        assert_ne!(
            converted_image_cache_key(7, jpeg),
            converted_image_cache_key(7, bmp)
        );
    }

    #[test]
    fn normalized_image_keeps_requested_dimensions() {
        let source = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 100, Rgb([255, 255, 255])));

        let normalized = normalize_button_image(source, 100, 100);

        assert_eq!(normalized.dimensions(), (100, 100));
    }

    #[test]
    fn normalized_image_masks_all_four_corners_to_black() {
        let source = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 100, Rgb([255, 255, 255])));

        let normalized = normalize_button_image(source, 100, 100).to_rgb8();

        assert_eq!(normalized.get_pixel(0, 0), &Rgb([0, 0, 0]));
        assert_eq!(normalized.get_pixel(99, 0), &Rgb([0, 0, 0]));
        assert_eq!(normalized.get_pixel(0, 99), &Rgb([0, 0, 0]));
        assert_eq!(normalized.get_pixel(99, 99), &Rgb([0, 0, 0]));
    }

    #[test]
    fn normalized_image_preserves_center_content() {
        let source = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(100, 100, Rgb([255, 255, 255])));

        let normalized = normalize_button_image(source, 100, 100).to_rgb8();

        assert_eq!(normalized.get_pixel(50, 50), &Rgb([255, 255, 255]));
        assert_eq!(
            normalized.get_pixel(BUTTON_CORNER_RADIUS, BUTTON_CORNER_RADIUS),
            &Rgb([255, 255, 255])
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_render_assets_preconverts_black_frames() {
        let (formats, black_frames) = build_render_assets(&Kind::AMPGD6).await.unwrap();

        assert_eq!(formats.len(), KEY_COUNT);
        assert_eq!(black_frames.len(), KEY_COUNT);
        assert!(black_frames.iter().all(|payload| !payload.is_empty()));
    }

    #[test]
    fn clear_all_batch_with_images_avoids_native_clear_path() {
        let mut batch = empty_render_batch();
        let mut clear_all = false;

        apply_render_command_to_batch(&mut batch, &mut clear_all, RenderCommand::ClearAll).unwrap();
        apply_render_command_to_batch(
            &mut batch,
            &mut clear_all,
            RenderCommand::SetImage {
                position: 3,
                image_hash: 55,
                payload: "x".to_string(),
            },
        )
        .unwrap();

        assert!(clear_all);
        assert!(batch_contains_image_updates(&batch));
    }

    #[test]
    fn clear_only_batch_keeps_native_clear_path_available() {
        let mut batch = empty_render_batch();
        let mut clear_all = false;

        apply_render_command_to_batch(&mut batch, &mut clear_all, RenderCommand::ClearAll).unwrap();
        apply_render_command_to_batch(
            &mut batch,
            &mut clear_all,
            RenderCommand::ClearButton { position: 3 },
        )
        .unwrap();

        assert!(clear_all);
        assert!(!batch_contains_image_updates(&batch));
    }
}

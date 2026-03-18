use device::{handle_error, handle_set_image};
use mirajazz::device::Device;
use openaction::*;
use std::{collections::HashMap, sync::Arc, sync::LazyLock};
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use watcher::watcher_task;

#[cfg(not(target_os = "windows"))]
use tokio::signal::unix::{SignalKind, signal};

mod device;
mod inputs;
mod mappings;
mod watcher;

pub static DEVICES: LazyLock<RwLock<HashMap<String, Arc<Device>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
pub static TOKENS: LazyLock<RwLock<HashMap<String, CancellationToken>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
pub static TRACKER: LazyLock<Mutex<TaskTracker>> = LazyLock::new(|| Mutex::new(TaskTracker::new()));
pub static DEVICE_IMAGE_STATES: LazyLock<Mutex<HashMap<String, Arc<DeviceImageState>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub struct DeviceImageState {
    pub state_mutex: Mutex<DeviceImageStateInner>,
    pub io_mutex: Mutex<()>,
    pub flush_tx: mpsc::Sender<()>,
    pub shutdown_token: CancellationToken,
}

pub struct DeviceImageStateInner {
    pub last_image_hashes: [Option<u64>; mappings::KEY_COUNT],
    pub pending_image_hashes: [Option<u64>; mappings::KEY_COUNT],
}

impl Default for DeviceImageStateInner {
    fn default() -> Self {
        Self {
            last_image_hashes: [None; mappings::KEY_COUNT],
            pending_image_hashes: [None; mappings::KEY_COUNT],
        }
    }
}

struct GlobalEventHandler {}
impl openaction::GlobalEventHandler for GlobalEventHandler {
    async fn plugin_ready(
        &self,
        _outbound: &mut openaction::OutboundEventManager,
    ) -> EventHandlerResult {
        let tracker = TRACKER.lock().await.clone();

        let token = CancellationToken::new();
        tracker.spawn(watcher_task(token.clone()));

        TOKENS
            .write()
            .await
            .insert("_watcher_task".to_string(), token);

        log::info!("Plugin initialized");

        Ok(())
    }

    async fn set_image(
        &self,
        event: SetImageEvent,
        _outbound: &mut OutboundEventManager,
    ) -> EventHandlerResult {
        log::debug!("Asked to set image: {:#?}", event);

        // Skip knobs images
        if event.controller == Some("Encoder".to_string()) {
            log::debug!("Looks like a knob, no need to set image");
            return Ok(());
        }

        let id = event.device.clone();

        let device = DEVICES.read().await.get(&event.device).cloned();

        if let Some(device) = device {
            if let Err(err) = handle_set_image(device.as_ref(), event).await {
                handle_error(&id, err).await;
            }
        } else {
            log::error!("Received event for unknown device: {}", event.device);
        }

        Ok(())
    }

    async fn set_brightness(
        &self,
        event: SetBrightnessEvent,
        _outbound: &mut OutboundEventManager,
    ) -> EventHandlerResult {
        log::debug!("Asked to set brightness: {:#?}", event);

        let id = event.device.clone();

        let device = DEVICES.read().await.get(&event.device).cloned();

        if let Some(device) = device {
            if let Err(err) = device.set_brightness(event.brightness).await {
                handle_error(&id, err).await;
            }
        } else {
            log::error!("Received event for unknown device: {}", event.device);
        }

        Ok(())
    }
}

struct ActionEventHandler {}
impl openaction::ActionEventHandler for ActionEventHandler {}

async fn shutdown() {
    let tokens = TOKENS.write().await;

    for (_, token) in tokens.iter() {
        token.cancel();
    }
}

async fn connect() -> EventHandlerResult {
    init_plugin(GlobalEventHandler {}, ActionEventHandler {})
        .await
        .map_err(|error| {
            log::error!("Failed to initialize plugin: {}", error);
            error
        })
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
async fn sigterm() -> EventHandlerResult {
    let mut sig = signal(SignalKind::terminate())?;

    sig.recv().await;

    Ok(())
}

#[cfg(target_os = "windows")]
async fn sigterm() -> EventHandlerResult {
    tokio::signal::ctrl_c().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> EventHandlerResult {
    simplelog::TermLogger::init(
        simplelog::LevelFilter::Info,
        simplelog::Config::default(),
        simplelog::TerminalMode::Stdout,
        simplelog::ColorChoice::Never,
    )
    .unwrap_or_else(|err| eprintln!("Failed to initialize logger: {err}"));

    let result = tokio::select! {
        result = connect() => result,
        result = sigterm() => result,
    };

    log::info!("Shutting down");

    shutdown().await;

    let tracker = TRACKER.lock().await.clone();

    log::info!("Waiting for tasks to finish");

    tracker.close();
    tracker.wait().await;

    log::info!("Tasks are finished, exiting now");

    result
}

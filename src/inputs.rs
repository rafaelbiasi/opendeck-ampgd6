use mirajazz::{error::MirajazzError, types::DeviceInput};
use std::sync::atomic::{AtomicU16, Ordering};

use crate::mappings::KEY_COUNT;

static GLOBAL_BUTTON_STATE: AtomicU16 = AtomicU16::new(0);

pub fn process_input(input: u8, state: u8) -> Result<DeviceInput, MirajazzError> {
    log::debug!("Processing input: {}, {}", input, state);

    match input as usize {
        (0..=KEY_COUNT) => read_button_press(input, state),
        _ => Err(MirajazzError::BadData),
    }
}

fn read_button_states(state_bitmask: u16) -> Vec<bool> {
    let mut bools = vec![false; KEY_COUNT];

    for (i, b) in bools.iter_mut().enumerate().take(KEY_COUNT) {
        *b = (state_bitmask & (1 << i)) != 0;
    }

    bools
}

/// Converts opendeck key index to device key index
/// For 3x5 layout (15 buttons), OpenDeck indexes: 0-14
/// OpenDeck layout (row-major):
/// Row 1: 0, 1, 2, 3, 4
/// Row 2: 5, 6, 7, 8, 9
/// Row 3: 10, 11, 12, 13, 14
///
/// AMPGD6 mapping - trying ss550 mapping pattern first, adjust if needed
/// This maps OpenDeck positions to device button indexes
pub fn opendeck_to_device(key: u8) -> u8 {
    if key < KEY_COUNT as u8 {
        // Try ss550-like mapping: [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
        // This means: OpenDeck 0 -> Device 10, OpenDeck 1 -> Device 11, etc.
        [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4][key as usize]
    } else {
        key
    }
}

/// Converts device key index to opendeck key index
/// Device sends 1-based indexes (1-15), we convert to 0-based OpenDeck indexes (0-14)
///
/// User testing shows:
/// - Image 0 -> Action 10 (when pressed) - WRONG, should be Action 0
/// - Image 10 -> Action 0 (when pressed) - WRONG, should be Action 10
///
/// opendeck_to_device: [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
/// This means: Device 10 shows image 0, Device 0 shows image 10
///
/// For events: When pressing Device 10 (shows image 0), we want action 0 to trigger
/// When pressing Device 0 (shows image 10), we want action 10 to trigger
///
/// So: device_to_opendeck(10) should return 0, device_to_opendeck(0) should return 10
///
/// The correct inverse mapping: find where each device index appears in opendeck_to_device
/// Device 0 appears at position 10 -> should return 10 (but we want 0 for image 0...)
/// Wait, that's confusing. Let me think:
///
/// If Device 10 shows image 0, and we want pressing Device 10 to trigger action 0,
/// then device_to_opendeck(10) must return 0.
///
/// The inverse of [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4] is:
/// Device 0 -> OpenDeck 10, Device 1 -> OpenDeck 11, ..., Device 10 -> OpenDeck 0, Device 11 -> OpenDeck 1, ...
/// Which is: [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
///
/// But that's what we had and it gave wrong results. The issue might be that the device
/// sends a different index than what we use for images. Let's create a mapping based on the actual behavior.
pub fn device_to_opendeck(key: usize) -> usize {
    // Try the ss550 approach: simple linear mapping
    // The device sends 1-based indexes (1-15), we convert to 0-based (0-14)
    // For ss550, this works because the device sends linear indexes for events
    // even though images use a mapped layout
    let result = key - 1;
    log::debug!(
        "device_to_opendeck: device_index_1based={}, opendeck_index={}",
        key,
        result
    );
    result
}

fn read_button_press(input: u8, state: u8) -> Result<DeviceInput, MirajazzError> {
    if input == 0 {
        GLOBAL_BUTTON_STATE.store(0, Ordering::SeqCst);
        return Ok(DeviceInput::ButtonStateChange(read_button_states(0)));
    }

    let pressed_index: usize = device_to_opendeck(input as usize);
    let is_pressed = state != 0;
    
    log::debug!(
        "Button event: device_index={}, opendeck_index={}, raw_state={}, is_pressed={}",
        input,
        pressed_index,
        state,
        is_pressed
    );

    let mut current_state = GLOBAL_BUTTON_STATE.load(Ordering::SeqCst);

    if pressed_index < KEY_COUNT {
        if is_pressed {
            current_state |= 1 << pressed_index;
        } else {
            current_state &= !(1 << pressed_index);
        }
        GLOBAL_BUTTON_STATE.store(current_state, Ordering::SeqCst);
    } else {
        log::warn!(
            "Button index {} out of range (max: {})",
            pressed_index,
            KEY_COUNT - 1
        );
    }

    Ok(DeviceInput::ButtonStateChange(read_button_states(
        current_state,
    )))
}

#[cfg(test)]
mod tests {
    use super::{device_to_opendeck, opendeck_to_device};

    #[test]
    fn image_mapping_matches_expected_layout() {
        let expected = [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4];

        for (opendeck, device) in expected.into_iter().enumerate() {
            assert_eq!(opendeck_to_device(opendeck as u8), device);
        }
    }

    #[test]
    fn event_mapping_is_linear_one_based() {
        for device_index in 1..=15 {
            assert_eq!(device_to_opendeck(device_index), device_index - 1);
        }
    }
}

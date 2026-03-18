use mirajazz::{error::MirajazzError, state::DeviceStateUpdate, types::DeviceInput};

use crate::mappings::KEY_COUNT;

const ACK_PREFIX: [u8; 3] = [65, 67, 75];
const MAX_DEVICE_INPUT: u8 = KEY_COUNT as u8;
const OPENDECK_TO_DEVICE: [u8; KEY_COUNT] = [10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D6InputEvent {
    Reset,
    Button { key: u8, pressed: bool },
}

pub fn ignore_process_input(_input: u8, _state: u8) -> Result<DeviceInput, MirajazzError> {
    Ok(DeviceInput::NoData)
}

pub fn decode_input_report(data: &[u8]) -> Result<Option<D6InputEvent>, MirajazzError> {
    if data.len() < 11 {
        return Ok(None);
    }

    if !data.starts_with(&ACK_PREFIX) {
        return Ok(None);
    }

    let input = data[9];
    let state = data[10];

    log::debug!(
        "Decoding raw input report: input={}, state={}, len={}",
        input,
        state,
        data.len()
    );

    match input {
        0 => Ok(Some(D6InputEvent::Reset)),
        1..=MAX_DEVICE_INPUT => Ok(Some(D6InputEvent::Button {
            key: device_to_opendeck(input as usize)? as u8,
            pressed: state != 0,
        })),
        _ => Err(MirajazzError::BadData),
    }
}

pub fn apply_input_event(state_bitmask: &mut u16, event: D6InputEvent) -> Vec<DeviceStateUpdate> {
    match event {
        D6InputEvent::Reset => {
            let previous = *state_bitmask;
            *state_bitmask = 0;

            (0..KEY_COUNT)
                .filter(|index| (previous & (1 << index)) != 0)
                .map(|index| DeviceStateUpdate::ButtonUp(index as u8))
                .collect()
        }
        D6InputEvent::Button { key, pressed } => {
            let bit = 1u16 << key;
            let was_pressed = (*state_bitmask & bit) != 0;

            if pressed == was_pressed {
                return Vec::new();
            }

            if pressed {
                *state_bitmask |= bit;
                vec![DeviceStateUpdate::ButtonDown(key)]
            } else {
                *state_bitmask &= !bit;
                vec![DeviceStateUpdate::ButtonUp(key)]
            }
        }
    }
}

/// Converts opendeck key index to device key index
/// For 3x5 layout (15 buttons), OpenDeck indexes: 0-14
/// OpenDeck layout (row-major):
/// Row 1: 0, 1, 2, 3, 4
/// Row 2: 5, 6, 7, 8, 9
/// Row 3: 10, 11, 12, 13, 14
pub fn opendeck_to_device(key: u8) -> u8 {
    if key < KEY_COUNT as u8 {
        OPENDECK_TO_DEVICE[key as usize]
    } else {
        key
    }
}

/// Converts device key index to opendeck key index
/// Device sends 1-based indexes (1-15), we convert to 0-based OpenDeck indexes (0-14)
pub fn device_to_opendeck(key: usize) -> Result<usize, MirajazzError> {
    if !(1..=KEY_COUNT).contains(&key) {
        log::warn!(
            "Button index {} out of range in device_to_opendeck (expected 1..={})",
            key,
            KEY_COUNT
        );
        return Err(MirajazzError::BadData);
    }

    Ok(key - 1)
}

#[cfg(test)]
mod tests {
    use mirajazz::state::DeviceStateUpdate;

    use super::{
        D6InputEvent, apply_input_event, decode_input_report, device_to_opendeck,
        opendeck_to_device,
    };

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
            assert_eq!(device_to_opendeck(device_index).unwrap(), device_index - 1);
        }
    }

    #[test]
    fn decode_ack_button_report() {
        let mut data = vec![0u8; 11];
        data[..3].copy_from_slice(b"ACK");
        data[9] = 3;
        data[10] = 1;

        assert_eq!(
            decode_input_report(&data).unwrap(),
            Some(D6InputEvent::Button {
                key: 2,
                pressed: true,
            })
        );
    }

    #[test]
    fn ignore_non_ack_report() {
        let data = vec![0u8; 11];
        assert!(decode_input_report(&data).unwrap().is_none());
    }

    #[test]
    fn reset_releases_pressed_buttons() {
        let mut state = 0b101;
        let updates = apply_input_event(&mut state, D6InputEvent::Reset);

        assert_eq!(state, 0);
        assert_eq!(updates.len(), 2);
        assert!(matches!(updates[0], DeviceStateUpdate::ButtonUp(0)));
        assert!(matches!(updates[1], DeviceStateUpdate::ButtonUp(2)));
    }

    #[test]
    fn duplicate_state_does_not_emit_repeat() {
        let mut state = 0;

        let first = apply_input_event(
            &mut state,
            D6InputEvent::Button {
                key: 4,
                pressed: true,
            },
        );
        let second = apply_input_event(
            &mut state,
            D6InputEvent::Button {
                key: 4,
                pressed: true,
            },
        );

        assert_eq!(first.len(), 1);
        assert!(matches!(first[0], DeviceStateUpdate::ButtonDown(4)));
        assert!(second.is_empty());
    }
}

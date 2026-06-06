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

            if pressed && was_pressed {
                // Button already pressed — the release report was likely lost.
                // Synthesize a release before re-pressing so the action fires.
                log::debug!(
                    "Button {} already pressed, synthesizing release before re-press",
                    key
                );
                vec![
                    DeviceStateUpdate::ButtonUp(key),
                    DeviceStateUpdate::ButtonDown(key),
                ]
            } else if !pressed && !was_pressed {
                // Already released — ignore duplicate release
                Vec::new()
            } else if pressed {
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
///
/// AMPGD6 mapping - trying ss550 mapping pattern first, adjust if needed
/// This maps OpenDeck positions to device button indexes
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
        apply_input_event, decode_input_report, device_to_opendeck, opendeck_to_device,
        D6InputEvent,
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
    fn duplicate_press_synthesizes_release_and_repress() {
        let mut state = 0;
        let press = D6InputEvent::Button {
            key: 4,
            pressed: true,
        };

        let first = apply_input_event(&mut state, press);
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0], DeviceStateUpdate::ButtonDown(4)));
        assert_eq!(
            state,
            1 << 4,
            "state should have bit 4 set after first press"
        );

        let second = apply_input_event(&mut state, press);
        assert_eq!(second.len(), 2);
        assert!(matches!(second[0], DeviceStateUpdate::ButtonUp(4)));
        assert!(matches!(second[1], DeviceStateUpdate::ButtonDown(4)));
        assert_eq!(
            state,
            1 << 4,
            "state should still have bit 4 set after duplicate press (auto-recovery)"
        );
    }

    #[test]
    fn duplicate_release_is_ignored() {
        let mut state = 0;
        let release = D6InputEvent::Button {
            key: 4,
            pressed: false,
        };

        let result = apply_input_event(&mut state, release);
        assert!(result.is_empty());
        assert_eq!(state, 0, "state should remain 0 after duplicate release");
    }

    #[test]
    fn state_evolution_through_press_release_cycle() {
        let mut state = 0;
        let key = 3;
        let bit = 1 << key;

        // First press
        let press1 = apply_input_event(&mut state, D6InputEvent::Button { key, pressed: true });
        assert_eq!(press1.len(), 1);
        assert!(matches!(press1[0], DeviceStateUpdate::ButtonDown(3)));
        assert_eq!(state, bit);

        // Duplicate press (auto-recovery)
        let press2 = apply_input_event(&mut state, D6InputEvent::Button { key, pressed: true });
        assert_eq!(press2.len(), 2);
        assert!(matches!(press2[0], DeviceStateUpdate::ButtonUp(3)));
        assert!(matches!(press2[1], DeviceStateUpdate::ButtonDown(3)));
        assert_eq!(
            state, bit,
            "state should remain unchanged during auto-recovery"
        );

        // Release
        let release1 = apply_input_event(
            &mut state,
            D6InputEvent::Button {
                key,
                pressed: false,
            },
        );
        assert_eq!(release1.len(), 1);
        assert!(matches!(release1[0], DeviceStateUpdate::ButtonUp(3)));
        assert_eq!(state, 0);

        // Duplicate release (ignored)
        let release2 = apply_input_event(
            &mut state,
            D6InputEvent::Button {
                key,
                pressed: false,
            },
        );
        assert!(release2.is_empty());
        assert_eq!(state, 0);

        // Press again after full cycle
        let press3 = apply_input_event(&mut state, D6InputEvent::Button { key, pressed: true });
        assert_eq!(press3.len(), 1);
        assert!(matches!(press3[0], DeviceStateUpdate::ButtonDown(3)));
        assert_eq!(state, bit);
    }
}

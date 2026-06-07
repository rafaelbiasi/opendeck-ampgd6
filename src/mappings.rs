use mirajazz::{
    device::DeviceQuery,
    types::{HidDeviceInfo, ImageFormat, ImageMirroring, ImageMode, ImageRotation},
};

// Device namespace: must be unique between all the plugins, 2 characters long and match `DeviceNamespace` field in `manifest.json`
// Previously used "99" from the source project akp153, now changed to "d6" for this plugin
pub const DEVICE_NAMESPACE: &str = "d6";

pub const ROW_COUNT: usize = 3;
pub const COL_COUNT: usize = 5;
pub const KEY_COUNT: usize = ROW_COUNT * COL_COUNT;
pub const ENCODER_COUNT: usize = 0;

#[derive(Debug, Clone)]
pub enum Kind {
    AMPGD6,
    AMPGD6REV2,
}

pub const FIFINE_VID: u16 = 0x3142;
pub const AMPGD6_PID: u16 = 0x0007;
pub const AMPGD6REV2_PID: u16 = 0x0060;

pub const AMPGD6_QUERY: DeviceQuery = DeviceQuery::new(65440, 1, FIFINE_VID, AMPGD6_PID);
pub const AMPGD6REV2_QUERY: DeviceQuery = DeviceQuery::new(65440, 1, FIFINE_VID, AMPGD6REV2_PID);

pub const QUERIES: [DeviceQuery; 2] = [AMPGD6_QUERY, AMPGD6REV2_QUERY];

/// Returns correct image format for device kind and key
pub fn get_image_format_for_key(kind: &Kind, _key: u8) -> ImageFormat {
    // AMPGD6 doesn't need rotation or mirroring - images are displayed normally
    let size = if kind.protocol_version() == 1 {
        (100, 100)
    } else {
        (100, 100)
    };

    ImageFormat {
        mode: ImageMode::JPEG,
        size,
        rotation: ImageRotation::Rot180, // AMPGD6 needs 180° rotation
        mirror: ImageMirroring::None,  // No mirroring needed for AMPGD6
    }
}

impl Kind {
    /// Matches devices VID+PID pairs to correct kinds
    pub fn from_vid_pid(vid: u16, pid: u16) -> Option<Self> {
        match vid {
            FIFINE_VID => match pid {
                AMPGD6_PID => Some(Kind::AMPGD6),
                 AMPGD6REV2_PID => Some(Kind::AMPGD6REV2),
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns protocol version for device
    pub fn protocol_version(&self) -> usize {
        match self {
            Self::AMPGD6 => 1, // Back to version 1 - the error might be related to button count or initialization
            Self::AMPGD6REV2 => 2,
        }
    }
    /// There is no point relying on manufacturer/device names reported by the USB stack,
    /// so we return custom names for all the kinds of devices
    pub fn human_name(&self) -> &'static str {
        match &self {
            Self::AMPGD6 => "FIFINE Ampligame D6",
            Self::AMPGD6REV2 => "FIFINE Ampligame D6 Rev. 2",
        }
    }

    pub fn image_format(&self, _key: u8) -> ImageFormat {
        // Larger frames get cropped by the device firmware.
        // Keep the conservative size and let the renderer center the source image.
        let size = (100, 100);

        ImageFormat {
            mode: ImageMode::JPEG,
            size,
            rotation: ImageRotation::Rot180,
            mirror: ImageMirroring::None,
        }
    }

    pub fn supports_brightness(&self) -> bool {
        match self {
            Self::AMPGD6 => true,
            Self::AMPGD6REV2 => true,
        }
    }

    pub fn supports_keepalive(&self) -> bool {
        match self {
            Self::AMPGD6 => false,
            Self::AMPGD6REV2 => false,
        }
    }

    pub fn known_singleton_limit(&self) -> bool {
        match self {
            Self::AMPGD6 => true,
            Self::AMPGD6REV2 => true,
        }
    }

    /// Because "v1" devices all share the same serial number, use custom suffix to be able to connect
    /// two devices with the different revisions at the same time
    pub fn id_suffix(&self) -> &'static str {
        match &self {
            Self::AMPGD6 => "AMPGD6",
            Self::AMPGD6REV2 => "AMPGD6REV2",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CandidateDevice {
    pub id: String,
    pub dev: HidDeviceInfo,
    pub kind: Kind,
}

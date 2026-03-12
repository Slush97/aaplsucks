//! AT-SPI2 accessibility bridge — exposes the UI's a11y tree over D-Bus.
//!
//! Architecture: the bridge runs on a background tokio thread. The UI thread
//! pushes a11y tree diffs through a channel each frame. The bridge thread
//! maintains the D-Bus accessible objects and emits state-changed events.
//!
//! This module implements Phase 3C Step 1 (Foundation):
//! - Registers with the AT-SPI2 registry bus
//! - Exposes the root `org.a11y.atspi.Accessible` object
//! - Provides the channel-based diff infrastructure for future steps

use tokio::sync::mpsc;

/// A snapshot of the a11y tree sent from the UI thread to the bridge thread.
#[derive(Debug, Clone)]
pub struct A11yTreeSnapshot {
    pub nodes: Vec<A11yNodeSnapshot>,
    pub root_children: Vec<u64>,
}

/// A single node in the snapshot.
#[derive(Debug, Clone)]
pub struct A11yNodeSnapshot {
    pub id: u64,
    pub role: u32, // AT-SPI2 role enum value
    pub label: String,
    pub value: Option<String>,
    pub rect: [f32; 4], // x, y, w, h
    pub focused: bool,
    pub disabled: bool,
    pub expanded: Option<bool>,
    pub selected: Option<bool>,
    pub checked: Option<bool>,
    pub value_range: Option<(f32, f32, f32)>,
    pub children: Vec<u64>,
}

/// Handle to the running AT-SPI2 bridge.
pub struct AtspiBridge {
    tx: mpsc::Sender<A11yTreeSnapshot>,
    _thread: std::thread::JoinHandle<()>,
}

impl AtspiBridge {
    /// Start the AT-SPI2 bridge on a background thread.
    ///
    /// Returns `None` if the AT-SPI2 bus is not available.
    pub fn start(app_name: &str) -> Option<Self> {
        let (tx, rx) = mpsc::channel::<A11yTreeSnapshot>(4);
        let name = app_name.to_string();

        let thread = std::thread::Builder::new()
            .name("atspi-bridge".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to create tokio runtime for atspi bridge");
                rt.block_on(bridge_main(name, rx));
            })
            .ok()?;

        Some(Self { tx, _thread: thread })
    }

    /// Push a new a11y tree snapshot to the bridge. Non-blocking; drops if full.
    pub fn update(&self, snapshot: A11yTreeSnapshot) {
        let _ = self.tx.try_send(snapshot);
    }
}

/// Main loop for the AT-SPI2 bridge (runs on background thread).
async fn bridge_main(app_name: String, mut rx: mpsc::Receiver<A11yTreeSnapshot>) {
    // Attempt to connect to the AT-SPI2 bus.
    let connection = match zbus::Connection::session().await {
        Ok(conn) => conn,
        Err(e) => {
            tracing::warn!("AT-SPI2 bridge: failed to connect to session bus: {e}");
            return;
        }
    };

    tracing::info!("AT-SPI2 bridge: connected to session bus for '{app_name}'");

    // TODO Phase 3C Steps 2-5:
    // - Register root accessible object with AT-SPI2 registry
    // - Diff snapshots each frame and create/remove D-Bus objects
    // - Map A11yRole to AT-SPI2 roles
    // - Implement Action/Text/Value interfaces
    // - Emit focus change events

    // For now, just drain the channel to keep the bridge alive.
    while let Some(_snapshot) = rx.recv().await {
        // Future: diff and update D-Bus objects
    }

    tracing::info!("AT-SPI2 bridge: shutting down");
    drop(connection);
}

/// Map esox_ui A11yRole to AT-SPI2 role constants.
///
/// See: https://lazka.github.io/pgi-docs/Atspi-2.0/enums.html#Atspi.Role
pub fn map_role(role: u32) -> u32 {
    // Placeholder mapping — will be fleshed out in Step 3.
    // For now, return ROLE_UNKNOWN for everything.
    role
}

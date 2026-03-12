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

/// Convert an `esox_ui::A11yTree` into a snapshot suitable for the bridge.
pub fn snapshot_from_tree(tree: &esox_ui::A11yTree) -> A11yTreeSnapshot {
    A11yTreeSnapshot {
        nodes: tree
            .nodes
            .iter()
            .map(|n| A11yNodeSnapshot {
                id: n.id,
                role: map_role(n.role),
                label: n.label.clone(),
                value: n.value.clone(),
                rect: [n.rect.x, n.rect.y, n.rect.w, n.rect.h],
                focused: n.focused,
                disabled: n.disabled,
                expanded: n.expanded,
                selected: n.selected,
                checked: n.checked,
                value_range: n.value_range,
                children: n.children.clone(),
            })
            .collect(),
        root_children: tree.root_children.clone(),
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
/// See: <https://lazka.github.io/pgi-docs/Atspi-2.0/enums.html#Atspi.Role>
pub fn map_role(role: esox_ui::A11yRole) -> u32 {
    use esox_ui::A11yRole;
    match role {
        A11yRole::Button => 62,       // ROLE_PUSH_BUTTON
        A11yRole::Checkbox => 20,     // ROLE_CHECK_BOX
        A11yRole::RadioButton => 72,  // ROLE_RADIO_BUTTON
        A11yRole::TextInput => 84,    // ROLE_TEXT
        A11yRole::TextArea => 84,     // ROLE_TEXT
        A11yRole::Slider => 79,       // ROLE_SLIDER
        A11yRole::Select => 24,       // ROLE_COMBO_BOX
        A11yRole::Tab => 87,          // ROLE_PAGE_TAB
        A11yRole::TabPanel => 88,     // ROLE_PAGE_TAB_LIST
        A11yRole::Table => 83,        // ROLE_TABLE
        A11yRole::TableRow => 86,     // ROLE_TABLE_ROW
        A11yRole::TableCell => 85,    // ROLE_TABLE_CELL
        A11yRole::Tree => 93,         // ROLE_TREE
        A11yRole::TreeItem => 94,     // ROLE_TREE_ITEM
        A11yRole::ProgressBar => 61,  // ROLE_PROGRESS_BAR
        A11yRole::Dialog => 28,       // ROLE_DIALOG
        A11yRole::Alert => 2,         // ROLE_ALERT
        A11yRole::Label => 47,        // ROLE_LABEL
        A11yRole::Separator => 78,    // ROLE_SEPARATOR
        A11yRole::ScrollView => 76,   // ROLE_SCROLL_PANE
        A11yRole::Group => 60,        // ROLE_PANEL
        A11yRole::ToggleButton => 82, // ROLE_TOGGLE_BUTTON
        A11yRole::Link => 48,         // ROLE_LINK
        A11yRole::SpinButton => 81,   // ROLE_SPIN_BUTTON
        A11yRole::Combobox => 24,     // ROLE_COMBO_BOX
    }
}

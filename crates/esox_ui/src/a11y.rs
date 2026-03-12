//! Accessibility — semantic tree foundation.
//!
//! Phase 1 builds the `A11yTree` each frame. AT-SPI2 D-Bus bridge is future work.
//! Types are re-exported from `state.rs` where the tree lives alongside other UI state.

// All types (A11yRole, A11yNode, A11yTree) live in state.rs for proximity to UiState.
// This module exists as a namespace for future AT-SPI2 bridge code.

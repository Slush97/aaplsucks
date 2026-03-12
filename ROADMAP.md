# esox_ui Widget Roadmap

## Tier 1 — Foundation (done)

- **Text wrapping & truncation** — `label_wrapped()`, `label_truncated()`, `wrap_lines()`, `draw_text_truncated()`
- **Disabled widget state** — `ui.disabled(true, |ui| { ... })` scoped API, all interactive widgets render dimmed
- **Progress bar** — `progress_bar()`, `progress_bar_colored()`
- **Multi-line text area** — `text_area()` with vertical scroll, line-aware cursor, Enter/Tab/Up/Down

## Tier 2 — Makes it feel real (done)

- **Tooltips** — hover delay + floating label, reuse overlay system
- **Radio buttons** — checkbox variant with exclusive group semantics
- **Context menu** — right-click overlay, reuse dropdown machinery
- **Widget disabled styling refinements** — per-widget opacity, dashed borders

## Tier 3 — Production widget toolkit (done)

- **Flex/weighted layout** — `columns()`, `columns_spaced()` with relative weights
- **Tabs widget** — `tabs()` / `tab_bar()` with accent underline, keyboard Left/Right
- **Text area word wrap** — `text_area_wrapped()` with visual-line cursor navigation
- **Rich text labels** — `RichText` builder, `rich_label()`, `rich_label_wrapped()`, `draw_text_styled()` bold support
- **Virtual scrolling** — `virtual_scroll()` with uniform item height, only renders visible items, `scroll_to`
- **Drag-and-drop** — `drag_source()` / `drop_target()` / `accept_drop()` + platform `on_file_dropped()` / `on_file_hover()`
- **Table widget** — `table()` with `Fixed`/`Weight`/`Auto` column widths, virtual-scrolled body, zebra striping, row selection, keyboard nav
- **Tree widget** — `tree_node()` + `tree_indent()`, expand/collapse, selection, keyboard nav

## Tier 4 — Polish & power features (done)

- **Resizable table columns** — drag column borders to resize ✓
- **Table sorting** — click header to sort, sort indicator arrows ✓
- **Multi-select** — Shift+click range select, Ctrl+click toggle in table/tree ✓
- **Keyboard focus ring** — visible focus indicator for all focusable widgets ✓
- **Animations** — expand/collapse transitions for tree, tab content fade ✓
- **Accessibility** — screen reader labels, a11y tree, AT-SPI2 bridge foundation ✓
- **Theming** — runtime theme switching, `ThemeBuilder`, `ThemeTransition` ✓
- **Layout constraints** — min/max width/height, aspect ratio ✓
- **Modal dialogs** — overlay with backdrop, focus trap ✓
- **Toasts/notifications** — timed popups with dismiss ✓

## Tier 5 — Hardening

- **Unit tests** — ~70 tests covering pure logic modules (id, layout, state, theme, paint) ✓
- **Public animation API** — `ui.animate()`, `ui.animate_bool()`, `ui.is_animating()` ✓
- **Smooth scrolling** — inertial momentum with exponential decay ✓
- **Damage tracking** — frame-skip when idle via `DamageTracker` integration ✓
- **AT-SPI2 bridge** — widget a11y metadata, snapshot conversion, role mapping ✓
- **Documentation** — crate-level docs, architecture notes ✓

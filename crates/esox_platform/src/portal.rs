//! XDG Desktop Portal bridge for file dialogs, notifications, and system settings.
//!
//! Requires the `portals` feature (enables `ashpd` + `tokio`).
//!
//! Architecture mirrors the AT-SPI2 bridge: a background thread with a
//! single-threaded tokio runtime communicates via `mpsc` channels and wakes
//! the main event loop through `EventLoopProxy`.

use std::path::PathBuf;
use std::sync::mpsc;

use winit::event_loop::EventLoopProxy;

use crate::AppUserEvent;

/// Errors from portal operations.
#[derive(Debug, thiserror::Error)]
pub enum PortalError {
    /// The portal is not available (no desktop environment or no D-Bus).
    #[error("portal unavailable")]
    Unavailable,
    /// The user cancelled the operation.
    #[error("cancelled")]
    Cancelled,
    /// D-Bus communication error.
    #[error("dbus error: {0}")]
    Dbus(String),
}

/// System color scheme preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    NoPreference,
    Dark,
    Light,
}

/// System settings read from the Settings portal.
#[derive(Debug, Clone)]
pub struct SystemSettings {
    pub color_scheme: ColorScheme,
    pub accent_color: Option<[f32; 3]>,
    pub text_scale: f64,
}

/// A file filter for open/save dialogs.
#[derive(Debug, Clone)]
pub struct FileFilter {
    pub label: String,
    pub patterns: Vec<String>,
}

/// Portal request sent from the main thread to the background runtime.
enum PortalRequest {
    OpenFile {
        title: String,
        filters: Vec<FileFilter>,
        multiple: bool,
        reply: tokio::sync::oneshot::Sender<Result<Vec<PathBuf>, PortalError>>,
    },
    OpenDirectory {
        title: String,
        reply: tokio::sync::oneshot::Sender<Result<Vec<PathBuf>, PortalError>>,
    },
    SaveFile {
        title: String,
        filters: Vec<FileFilter>,
        suggested_name: Option<String>,
        reply: tokio::sync::oneshot::Sender<Result<Option<PathBuf>, PortalError>>,
    },
    Notify {
        summary: String,
        body: String,
        reply: tokio::sync::oneshot::Sender<Result<(), PortalError>>,
    },
    ReadSettings {
        reply: tokio::sync::oneshot::Sender<Result<SystemSettings, PortalError>>,
    },
    Screenshot {
        interactive: bool,
        reply: tokio::sync::oneshot::Sender<Result<PathBuf, PortalError>>,
    },
}

/// Clone-cheap handle for sending portal requests from the main thread.
#[derive(Clone)]
pub struct PortalHandle {
    tx: mpsc::Sender<PortalRequest>,
}

impl PortalHandle {
    /// Open a file dialog. Returns selected file paths.
    pub fn open_file(
        &self,
        title: impl Into<String>,
        filters: Vec<FileFilter>,
        multiple: bool,
    ) -> tokio::sync::oneshot::Receiver<Result<Vec<PathBuf>, PortalError>> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(PortalRequest::OpenFile {
            title: title.into(),
            filters,
            multiple,
            reply,
        });
        rx
    }

    /// Open a directory chooser dialog.
    pub fn open_directory(
        &self,
        title: impl Into<String>,
    ) -> tokio::sync::oneshot::Receiver<Result<Vec<PathBuf>, PortalError>> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(PortalRequest::OpenDirectory {
            title: title.into(),
            reply,
        });
        rx
    }

    /// Open a save-file dialog.
    pub fn save_file(
        &self,
        title: impl Into<String>,
        filters: Vec<FileFilter>,
        suggested_name: Option<String>,
    ) -> tokio::sync::oneshot::Receiver<Result<Option<PathBuf>, PortalError>> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(PortalRequest::SaveFile {
            title: title.into(),
            filters,
            suggested_name,
            reply,
        });
        rx
    }

    /// Send a desktop notification.
    pub fn notify(
        &self,
        summary: impl Into<String>,
        body: impl Into<String>,
    ) -> tokio::sync::oneshot::Receiver<Result<(), PortalError>> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(PortalRequest::Notify {
            summary: summary.into(),
            body: body.into(),
            reply,
        });
        rx
    }

    /// Read current system settings (color scheme, accent, text scale).
    pub fn read_settings(
        &self,
    ) -> tokio::sync::oneshot::Receiver<Result<SystemSettings, PortalError>> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(PortalRequest::ReadSettings { reply });
        rx
    }

    /// Take a screenshot (interactive = show portal UI for region selection).
    pub fn screenshot(
        &self,
        interactive: bool,
    ) -> tokio::sync::oneshot::Receiver<Result<PathBuf, PortalError>> {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(PortalRequest::Screenshot { interactive, reply });
        rx
    }
}

/// Start the portal bridge on a background thread.
///
/// Returns a `PortalHandle` that can be used to issue requests. The bridge
/// sends `AppUserEvent::PortalReady` once the D-Bus connection is established.
pub fn start_portal_bridge(proxy: EventLoopProxy<AppUserEvent>) -> PortalHandle {
    let (tx, rx) = mpsc::channel::<PortalRequest>();

    std::thread::Builder::new()
        .name("portal-bridge".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create portal tokio runtime");

            rt.block_on(async move {
                let _ = proxy.send_event(AppUserEvent::PortalReady);
                portal_loop(rx).await;
            });
        })
        .expect("failed to spawn portal bridge thread");

    PortalHandle { tx }
}

async fn portal_loop(rx: mpsc::Receiver<PortalRequest>) {
    while let Ok(req) = rx.recv() {
        match req {
            PortalRequest::OpenFile { title, filters, multiple, reply } => {
                let result = handle_open_file(&title, &filters, multiple).await;
                let _ = reply.send(result);
            }
            PortalRequest::OpenDirectory { title, reply } => {
                let result = handle_open_directory(&title).await;
                let _ = reply.send(result);
            }
            PortalRequest::SaveFile { title, filters, suggested_name, reply } => {
                let result = handle_save_file(&title, &filters, suggested_name.as_deref()).await;
                let _ = reply.send(result);
            }
            PortalRequest::Notify { summary, body, reply } => {
                let result = handle_notify(&summary, &body).await;
                let _ = reply.send(result);
            }
            PortalRequest::ReadSettings { reply } => {
                let result = handle_read_settings().await;
                let _ = reply.send(result);
            }
            PortalRequest::Screenshot { interactive, reply } => {
                let result = handle_screenshot(interactive).await;
                let _ = reply.send(result);
            }
        }
    }
}

async fn handle_open_file(
    title: &str,
    filters: &[FileFilter],
    multiple: bool,
) -> Result<Vec<PathBuf>, PortalError> {
    use ashpd::desktop::file_chooser::{FileFilter as AshpdFilter, OpenFileRequest};

    let mut request = OpenFileRequest::default()
        .title(Some(title))
        .multiple(multiple);

    for f in filters {
        let mut filter = AshpdFilter::new(&f.label);
        for pat in &f.patterns {
            filter = filter.glob(pat);
        }
        request = request.filter(filter);
    }

    let response = request.send().await.map_err(portal_err)?;
    let response = response.response().map_err(portal_err)?;

    Ok(response
        .uris()
        .iter()
        .filter_map(|uri| uri.to_file_path().ok())
        .collect())
}

async fn handle_open_directory(title: &str) -> Result<Vec<PathBuf>, PortalError> {
    use ashpd::desktop::file_chooser::OpenFileRequest;

    let response = OpenFileRequest::default()
        .title(Some(title))
        .directory(true)
        .send()
        .await
        .map_err(portal_err)?;
    let response = response.response().map_err(portal_err)?;

    Ok(response
        .uris()
        .iter()
        .filter_map(|uri| uri.to_file_path().ok())
        .collect())
}

async fn handle_save_file(
    title: &str,
    filters: &[FileFilter],
    suggested_name: Option<&str>,
) -> Result<Option<PathBuf>, PortalError> {
    use ashpd::desktop::file_chooser::{FileFilter as AshpdFilter, SaveFileRequest};

    let mut request = SaveFileRequest::default().title(Some(title));

    if let Some(name) = suggested_name {
        request = request.current_name(Some(name));
    }

    for f in filters {
        let mut filter = AshpdFilter::new(&f.label);
        for pat in &f.patterns {
            filter = filter.glob(pat);
        }
        request = request.filter(filter);
    }

    let response = request.send().await.map_err(portal_err)?;
    let response = response.response().map_err(portal_err)?;

    Ok(response.uris().first().and_then(|uri| uri.to_file_path().ok()))
}

async fn handle_notify(summary: &str, body: &str) -> Result<(), PortalError> {
    use ashpd::desktop::notification::{NotificationProxy, Priority};

    let proxy = NotificationProxy::new().await.map_err(portal_err)?;
    let id = format!("esox-{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis());

    let notification = ashpd::desktop::notification::Notification::default()
        .title(summary)
        .body(body)
        .priority(Priority::Normal);

    proxy
        .add_notification(&id, notification)
        .await
        .map_err(portal_err)?;

    Ok(())
}

async fn handle_read_settings() -> Result<SystemSettings, PortalError> {
    use ashpd::desktop::settings::Settings;

    let settings = Settings::new().await.map_err(portal_err)?;

    let color_scheme = match settings
        .color_scheme()
        .await
    {
        Ok(ashpd::desktop::settings::ColorScheme::PreferDark) => ColorScheme::Dark,
        Ok(ashpd::desktop::settings::ColorScheme::PreferLight) => ColorScheme::Light,
        _ => ColorScheme::NoPreference,
    };

    let accent_color = settings
        .accent_color()
        .await
        .ok()
        .map(|c| [c.red() as f32, c.green() as f32, c.blue() as f32]);

    // Text scale: read from GNOME interface settings via the portal.
    let text_scale = settings
        .read::<f64>("org.gnome.desktop.interface", "text-scaling-factor")
        .await
        .unwrap_or(1.0);

    Ok(SystemSettings {
        color_scheme,
        accent_color,
        text_scale,
    })
}

async fn handle_screenshot(interactive: bool) -> Result<PathBuf, PortalError> {
    use ashpd::desktop::screenshot::ScreenshotRequest;

    let response = ScreenshotRequest::default()
        .interactive(interactive)
        .send()
        .await
        .map_err(portal_err)?;
    let response = response.response().map_err(portal_err)?;

    response
        .uri()
        .to_file_path()
        .map_err(|_| PortalError::Dbus("invalid screenshot URI".into()))
}

fn portal_err(e: impl std::fmt::Display) -> PortalError {
    let msg = e.to_string();
    if msg.contains("cancelled") || msg.contains("Cancelled") {
        PortalError::Cancelled
    } else {
        PortalError::Dbus(msg)
    }
}

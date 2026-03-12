//! Tool dispatch — maps tool IDs to dwnldr library calls.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::SyncSender;
use std::sync::Arc;

use tokio::runtime::Handle;
use tokio::task::AbortHandle;

use crate::app::{JobEvent, JobResult};
use crate::state::{DropZoneState, FormState};
use crate::tools::{self, OptionKind};

/// Dispatch a tool execution. Returns `Ok(Some(AbortHandle))` on success.
pub fn dispatch(
    tool_id: &'static str,
    form: &FormState,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    match tool_id {
        "download" => dispatch_download(tool_id, form, job_id, tx, handle),
        "trim" => dispatch_trim(tool_id, form, drop, job_id, tx, handle),
        "convert" => dispatch_convert(tool_id, form, drop, job_id, tx, handle),
        "resize" => dispatch_resize(tool_id, form, drop, job_id, tx, handle),
        "compress-image" => dispatch_compress_image(tool_id, form, drop, job_id, tx, handle),
        "pdf-split" => dispatch_pdf_split(tool_id, form, drop, job_id, tx, handle),
        "pdf-merge" => dispatch_pdf_merge(tool_id, drop, job_id, tx, handle),
        "pdf-compress" => dispatch_pdf_compress(tool_id, form, drop, job_id, tx, handle),
        "qr-gen" => dispatch_qr_gen(tool_id, form, job_id, tx, handle),
        "qr-decode" => dispatch_qr_decode(tool_id, drop, job_id, tx, handle),
        "zip" => dispatch_zip(tool_id, drop, job_id, tx, handle),
        "unzip" => dispatch_unzip(tool_id, drop, job_id, tx, handle),
        _ => Err(format!("Unknown tool: {tool_id}")),
    }
}

// ── Helpers ──

fn text_val(form: &FormState, tool: &'static str, key: &'static str) -> Option<String> {
    form.get(tool, key)
        .map(|s| s.text.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn select_val(form: &FormState, tool: &'static str, key: &'static str) -> Option<String> {
    let tool_def = tools::find_tool(tool)?;
    let opt = tool_def.options.iter().find(|o| o.key == key)?;
    if let OptionKind::Select { choices } = &opt.kind {
        Some(form.select_value(tool, key, choices).to_string())
    } else {
        None
    }
}

fn first_file(drop: &HashMap<&'static str, DropZoneState>, tool: &str) -> Option<PathBuf> {
    drop.get(tool)
        .and_then(|ds| ds.files.first().cloned())
}

fn all_files(drop: &HashMap<&'static str, DropZoneState>, tool: &str) -> Vec<PathBuf> {
    drop.get(tool)
        .map(|ds| ds.files.clone())
        .unwrap_or_default()
}

fn parse_quality(s: &str) -> dwnldr::core::types::Quality {
    use dwnldr::core::types::Quality;
    match s.to_lowercase().as_str() {
        "best" => Quality::Best,
        "1080p" => Quality::High,
        "720p" => Quality::Medium,
        "480p" => Quality::Low,
        "audio-only" => Quality::Best, // audio_only flag set separately
        other => Quality::Custom(other.to_string()),
    }
}

fn parse_output_format(s: &str) -> dwnldr::core::types::OutputFormat {
    use dwnldr::core::types::OutputFormat;
    match s.to_lowercase().as_str() {
        "mp4" => OutputFormat::Mp4,
        "mkv" => OutputFormat::Mkv,
        "webm" => OutputFormat::Webm,
        "mp3" => OutputFormat::Mp3,
        "flac" => OutputFormat::Flac,
        "wav" => OutputFormat::Wav,
        "ogg" => OutputFormat::Ogg,
        "aac" => OutputFormat::Aac,
        "png" => OutputFormat::Png,
        "jpg" | "jpeg" => OutputFormat::Jpg,
        "webp" => OutputFormat::Webp,
        "gif" => OutputFormat::Gif,
        "opus" => OutputFormat::Opus,
        "av1" => OutputFormat::Av1,
        "hevc" | "h265" => OutputFormat::Hevc,
        "avif" => OutputFormat::Avif,
        _ => OutputFormat::Original,
    }
}

fn parse_page_ranges(s: &str) -> Result<Vec<(u32, u32)>, String> {
    let mut ranges = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: u32 = a.trim().parse().map_err(|_| format!("Invalid page: {a}"))?;
            let end: u32 = b.trim().parse().map_err(|_| format!("Invalid page: {b}"))?;
            if start == 0 || end == 0 || start > end {
                return Err(format!("Invalid range: {part}"));
            }
            ranges.push((start, end));
        } else {
            let page: u32 = part.parse().map_err(|_| format!("Invalid page: {part}"))?;
            if page == 0 {
                return Err("Page numbers start at 1".into());
            }
            ranges.push((page, page));
        }
    }
    if ranges.is_empty() {
        return Err("No page ranges specified".into());
    }
    Ok(ranges)
}

fn is_image_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            matches!(
                e.to_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff" | "avif"
            )
        })
        .unwrap_or(false)
}

fn download_dir() -> PathBuf {
    dirs::download_dir().unwrap_or_else(|| PathBuf::from("."))
}

fn send_done(tx: &SyncSender<JobEvent>, job_id: u64, result: JobResult) {
    let _ = tx.send(JobEvent::Done { job_id, result });
}

fn send_error(tx: &SyncSender<JobEvent>, job_id: u64, message: String) {
    let _ = tx.send(JobEvent::Error { job_id, message });
}

// ── Tool Dispatchers ──

fn dispatch_download(
    tool_id: &'static str,
    form: &FormState,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let url = text_val(form, tool_id, "__input__")
        .ok_or("Enter a URL to download")?;

    let quality_str = select_val(form, tool_id, "quality").unwrap_or_else(|| "best".into());
    let format_str = select_val(form, tool_id, "format").unwrap_or_else(|| "mp4".into());

    let audio_only = quality_str == "audio-only";
    let quality = parse_quality(&quality_str);
    let output_format = parse_output_format(&format_str);

    let opts = dwnldr::core::types::DownloadOptions {
        quality,
        output_format,
        output_dir: download_dir(),
        audio_only,
        ..Default::default()
    };

    let progress_tx = tx.clone();
    let progress_job_id = job_id;
    let cb: dwnldr::core::types::ProgressCallback = Arc::new(move |event| {
        use dwnldr::core::types::ProgressEvent;
        let (message, percent) = match &event {
            ProgressEvent::Started { title } => {
                (format!("Downloading: {title}"), None)
            }
            ProgressEvent::Downloading { percent, speed, eta } => {
                let mut msg = String::from("Downloading");
                if let Some(p) = percent {
                    msg = format!("Downloading {p:.0}%");
                }
                if let Some(s) = speed {
                    msg.push_str(&format!(" — {s}"));
                }
                if let Some(e) = eta {
                    msg.push_str(&format!(" (ETA {e})"));
                }
                (msg, *percent)
            }
            ProgressEvent::Converting => ("Converting...".into(), None),
            ProgressEvent::Finished => ("Done".into(), Some(100.0)),
        };
        let _ = progress_tx.send(JobEvent::Progress {
            job_id: progress_job_id,
            message,
            percent,
        });
    });

    let jh = handle.spawn(async move {
        match dwnldr::pipeline::run(&url, &opts, Some(&cb)).await {
            Ok(files) => {
                let paths: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
                if paths.len() == 1 {
                    send_done(&tx, job_id, JobResult::File(paths.into_iter().next().unwrap()));
                } else {
                    send_done(&tx, job_id, JobResult::Files(paths));
                }
            }
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_trim(
    tool_id: &'static str,
    form: &FormState,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select a file to trim")?;
    let start = text_val(form, tool_id, "start");
    let end = text_val(form, tool_id, "end");
    if start.is_none() && end.is_none() {
        return Err("Specify at least a start or end time".into());
    }

    let jh = handle.spawn(async move {
        match dwnldr::converters::ffmpeg::trim(
            &path,
            start.as_deref(),
            end.as_deref(),
            None,
        )
        .await
        {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_convert(
    tool_id: &'static str,
    form: &FormState,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select a file to convert")?;
    let format_str = select_val(form, tool_id, "format").unwrap_or_else(|| "mp4".into());
    let format = parse_output_format(&format_str);

    let use_image = is_image_ext(&path);

    let jh = handle.spawn(async move {
        let result = if use_image {
            dwnldr::converters::image::convert(&path, &format).await
        } else {
            dwnldr::converters::ffmpeg::convert(&path, &format).await
        };
        match result {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_resize(
    tool_id: &'static str,
    form: &FormState,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select an image to resize")?;
    let width: Option<u32> = text_val(form, tool_id, "width")
        .and_then(|s| s.parse().ok());
    let height: Option<u32> = text_val(form, tool_id, "height")
        .and_then(|s| s.parse().ok());
    let scale: Option<f64> = text_val(form, tool_id, "scale")
        .and_then(|s| s.parse::<f64>().ok())
        .map(|v| v / 100.0);

    if width.is_none() && height.is_none() && scale.is_none() {
        return Err("Specify width, height, or scale".into());
    }

    let jh = handle.spawn(async move {
        match dwnldr::tools::image_ops::resize(&path, width, height, scale, None).await {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_compress_image(
    tool_id: &'static str,
    form: &FormState,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select an image to compress")?;
    let quality: u8 = text_val(form, tool_id, "quality")
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);

    let jh = handle.spawn(async move {
        match dwnldr::tools::image_ops::compress(&path, quality, None).await {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_pdf_split(
    tool_id: &'static str,
    form: &FormState,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select a PDF to split")?;
    let ranges_str = text_val(form, tool_id, "ranges").ok_or("Specify page ranges")?;
    let ranges = parse_page_ranges(&ranges_str)?;
    let output_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(download_dir);

    let jh = handle.spawn_blocking(move || {
        match dwnldr::tools::pdf::split(&path, &ranges, &output_dir) {
            Ok(files) => {
                let paths: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
                send_done(&tx, job_id, JobResult::Files(paths));
            }
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_pdf_merge(
    tool_id: &'static str,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let files = all_files(drop, tool_id);
    if files.len() < 2 {
        return Err("Select at least 2 PDFs to merge".into());
    }

    let output = files[0]
        .parent()
        .unwrap_or(Path::new("."))
        .join("merged.pdf");

    let jh = handle.spawn_blocking(move || {
        match dwnldr::tools::pdf::merge(&files, &output) {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_pdf_compress(
    tool_id: &'static str,
    form: &FormState,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select a PDF to compress")?;
    let quality = select_val(form, tool_id, "quality").unwrap_or_else(|| "ebook".into());

    let jh = handle.spawn(async move {
        match dwnldr::tools::pdf::compress(&path, &quality, None).await {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_qr_gen(
    tool_id: &'static str,
    form: &FormState,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let text = text_val(form, tool_id, "__input__")
        .ok_or("Enter text or URL to encode")?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let output = download_dir().join(format!("qr_{timestamp}.png"));

    let jh = handle.spawn_blocking(move || {
        match dwnldr::tools::qr::generate(&text, &output) {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_qr_decode(
    tool_id: &'static str,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select a QR code image")?;

    let jh = handle.spawn_blocking(move || {
        match dwnldr::tools::qr::decode(&path) {
            Ok(text) => send_done(&tx, job_id, JobResult::Text(text)),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_zip(
    tool_id: &'static str,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let files = all_files(drop, tool_id);
    if files.is_empty() {
        return Err("Select files to zip".into());
    }

    let output = files[0]
        .parent()
        .unwrap_or(Path::new("."))
        .join("archive.zip");

    let jh = handle.spawn_blocking(move || {
        match dwnldr::tools::compress::zip_files(&files, &output) {
            Ok(out) => send_done(&tx, job_id, JobResult::File(out.display().to_string())),
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

fn dispatch_unzip(
    tool_id: &'static str,
    drop: &HashMap<&'static str, DropZoneState>,
    job_id: u64,
    tx: SyncSender<JobEvent>,
    handle: &Handle,
) -> Result<Option<AbortHandle>, String> {
    let path = first_file(drop, tool_id).ok_or("Select a zip file to extract")?;
    let output_dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_else(download_dir);

    let jh = handle.spawn_blocking(move || {
        match dwnldr::tools::compress::unzip(&path, &output_dir) {
            Ok(files) => {
                let paths: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
                send_done(&tx, job_id, JobResult::Files(paths));
            }
            Err(e) => send_error(&tx, job_id, format!("{e}")),
        }
    });

    Ok(Some(jh.abort_handle()))
}

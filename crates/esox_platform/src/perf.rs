//! Live performance monitoring — frame times, RSS, CPU usage.
//!
//! Reads `/proc/self/status` and `/proc/self/stat` on Linux for memory and
//! CPU metrics.  Falls back to zeros on other platforms.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Rolling performance statistics updated each frame.
#[derive(Debug, Clone)]
pub struct PerfMonitor {
    /// Rolling window of frame durations (ms).
    frame_times: VecDeque<f64>,
    /// Maximum number of frame times to keep.
    window_size: usize,
    /// How often to re-read /proc (expensive-ish).
    sample_interval: Duration,
    last_sample: Instant,
    frame_start: Instant,
    /// Previous CPU jiffies reading for delta computation.
    prev_cpu_jiffies: u64,
    prev_cpu_time: Instant,

    // ── Public stats ──

    /// Frames per second (smoothed over the rolling window).
    pub fps: f32,
    /// Average frame time in milliseconds.
    pub frame_time_avg_ms: f32,
    /// 99th-percentile frame time in milliseconds.
    pub frame_time_p99_ms: f32,
    /// Minimum frame time in the window (ms).
    pub frame_time_min_ms: f32,
    /// Maximum frame time in the window (ms).
    pub frame_time_max_ms: f32,
    /// Resident set size in megabytes.
    pub rss_mb: f32,
    /// Virtual memory size in megabytes.
    pub virt_mb: f32,
    /// CPU usage as a percentage (0–100+, can exceed 100 on multi-core).
    pub cpu_percent: f32,
    /// Number of quad instances in the last frame.
    pub instance_count: u32,
    /// Number of draw batches in the last frame.
    pub batch_count: u32,
    /// Total frames rendered.
    pub total_frames: u64,
}

impl PerfMonitor {
    /// Create a new monitor with a rolling window of `window_size` frames.
    pub fn new(window_size: usize) -> Self {
        let now = Instant::now();
        Self {
            frame_times: VecDeque::with_capacity(window_size),
            window_size,
            sample_interval: Duration::from_millis(500),
            last_sample: now,
            frame_start: now,
            prev_cpu_jiffies: read_cpu_jiffies(),
            prev_cpu_time: now,
            fps: 0.0,
            frame_time_avg_ms: 0.0,
            frame_time_p99_ms: 0.0,
            frame_time_min_ms: 0.0,
            frame_time_max_ms: 0.0,
            rss_mb: 0.0,
            virt_mb: 0.0,
            cpu_percent: 0.0,
            instance_count: 0,
            batch_count: 0,
            total_frames: 0,
        }
    }

    /// Call at the start of each frame (before `on_redraw`).
    pub fn begin_frame(&mut self) {
        self.frame_start = Instant::now();
    }

    /// Call at the end of each frame (after GPU submit).
    ///
    /// Pass instance/batch counts from the frame for tracking.
    pub fn end_frame(&mut self, instance_count: u32, batch_count: u32) {
        let elapsed = self.frame_start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;

        if self.frame_times.len() >= self.window_size {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(ms);

        self.instance_count = instance_count;
        self.batch_count = batch_count;
        self.total_frames += 1;

        // Recompute rolling stats every frame (cheap over a small window).
        self.recompute_frame_stats();

        // Sample /proc periodically.
        let now = Instant::now();
        if now.duration_since(self.last_sample) >= self.sample_interval {
            self.sample_proc(now);
            self.last_sample = now;
        }
    }

    fn recompute_frame_stats(&mut self) {
        let n = self.frame_times.len();
        if n == 0 {
            return;
        }

        let sum: f64 = self.frame_times.iter().sum();
        self.frame_time_avg_ms = (sum / n as f64) as f32;
        self.fps = if self.frame_time_avg_ms > 0.0 {
            1000.0 / self.frame_time_avg_ms
        } else {
            0.0
        };

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for &t in &self.frame_times {
            if t < min {
                min = t;
            }
            if t > max {
                max = t;
            }
        }
        self.frame_time_min_ms = min as f32;
        self.frame_time_max_ms = max as f32;

        // p99: sort a copy and pick the 99th percentile index.
        let mut sorted: Vec<f64> = self.frame_times.iter().copied().collect();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((n as f64) * 0.99).ceil() as usize;
        self.frame_time_p99_ms = sorted[p99_idx.min(n - 1)] as f32;
    }

    fn sample_proc(&mut self, now: Instant) {
        let (rss, virt) = read_memory_kb();
        self.rss_mb = rss as f32 / 1024.0;
        self.virt_mb = virt as f32 / 1024.0;

        let jiffies = read_cpu_jiffies();
        let dt = now.duration_since(self.prev_cpu_time).as_secs_f64();
        if dt > 0.0 {
            let djiffies = jiffies.saturating_sub(self.prev_cpu_jiffies);
            let clock_ticks_per_sec = clock_ticks_per_sec();
            let cpu_seconds = djiffies as f64 / clock_ticks_per_sec as f64;
            self.cpu_percent = (cpu_seconds / dt * 100.0) as f32;
        }
        self.prev_cpu_jiffies = jiffies;
        self.prev_cpu_time = now;
    }

    /// Format a compact multi-line summary suitable for an overlay.
    pub fn summary(&self) -> String {
        format!(
            "FPS: {:.0}  frame: {:.2}ms (p99: {:.2}ms)\n\
             CPU: {:.1}%  RSS: {:.1}MB  VIRT: {:.0}MB\n\
             instances: {}  batches: {}  frames: {}",
            self.fps,
            self.frame_time_avg_ms,
            self.frame_time_p99_ms,
            self.cpu_percent,
            self.rss_mb,
            self.virt_mb,
            self.instance_count,
            self.batch_count,
            self.total_frames,
        )
    }
}

// ── Linux /proc helpers ──

#[cfg(target_os = "linux")]
fn read_memory_kb() -> (u64, u64) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (0, 0);
    };
    let mut rss = 0u64;
    let mut virt = 0u64;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = parse_kb(rest);
        } else if let Some(rest) = line.strip_prefix("VmSize:") {
            virt = parse_kb(rest);
        }
    }
    (rss, virt)
}

#[cfg(target_os = "linux")]
fn parse_kb(s: &str) -> u64 {
    s.trim().split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn read_cpu_jiffies() -> u64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    // Fields after the comm (which is in parens): skip to after ')'.
    let Some(after_comm) = stat.rfind(')') else {
        return 0;
    };
    let fields: Vec<&str> = stat[after_comm + 2..].split_whitespace().collect();
    // Field index 11 = utime, 12 = stime (0-indexed after comm).
    let utime: u64 = fields.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stime: u64 = fields.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
    utime + stime
}

#[cfg(target_os = "linux")]
fn clock_ticks_per_sec() -> u64 {
    // SAFETY: sysconf is a standard POSIX call.
    unsafe { libc::sysconf(libc::_SC_CLK_TCK) as u64 }
}

// ── Non-Linux stubs ──

#[cfg(not(target_os = "linux"))]
fn read_memory_kb() -> (u64, u64) {
    (0, 0)
}

#[cfg(not(target_os = "linux"))]
fn read_cpu_jiffies() -> u64 {
    0
}

#[cfg(not(target_os = "linux"))]
fn clock_ticks_per_sec() -> u64 {
    100
}

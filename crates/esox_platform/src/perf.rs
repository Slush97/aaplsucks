//! Live performance monitoring — frame times, RSS, CPU usage.
//!
//! Reads `/proc/self/status` and `/proc/self/stat` on Linux for memory and
//! CPU metrics.  Falls back to zeros on other platforms.
//!
//! On close, [`PerfMonitor::write_report`] writes a summary file with
//! session-wide statistics, histogram, and time-series snapshots.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A periodic snapshot of system metrics (taken every sample interval).
#[derive(Debug, Clone, Copy)]
struct Snapshot {
    /// Seconds since session start.
    elapsed_s: f32,
    fps: f32,
    frame_time_avg_ms: f32,
    frame_time_p99_ms: f32,
    rss_mb: f32,
    cpu_percent: f32,
    instance_count: u32,
    batch_count: u32,
}

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

    // ── Session-level tracking ──

    /// All frame times for the entire session (for histogram / full stats).
    all_frame_times: Vec<f64>,
    /// Periodic snapshots for time-series output.
    snapshots: Vec<Snapshot>,
    /// Session start time.
    session_start: Instant,
    /// Peak RSS observed during the session.
    peak_rss_mb: f32,
    /// Peak CPU% observed.
    peak_cpu_percent: f32,
    /// Sum of all CPU% samples for averaging.
    cpu_sum: f64,
    cpu_sample_count: u64,

    // ── Public stats (rolling window) ──

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
            all_frame_times: Vec::with_capacity(8192),
            snapshots: Vec::with_capacity(256),
            session_start: now,
            peak_rss_mb: 0.0,
            peak_cpu_percent: 0.0,
            cpu_sum: 0.0,
            cpu_sample_count: 0,
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
        self.all_frame_times.push(ms);

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

            // Record snapshot for time-series.
            self.snapshots.push(Snapshot {
                elapsed_s: now.duration_since(self.session_start).as_secs_f32(),
                fps: self.fps,
                frame_time_avg_ms: self.frame_time_avg_ms,
                frame_time_p99_ms: self.frame_time_p99_ms,
                rss_mb: self.rss_mb,
                cpu_percent: self.cpu_percent,
                instance_count,
                batch_count,
            });

            // Track peaks.
            if self.rss_mb > self.peak_rss_mb {
                self.peak_rss_mb = self.rss_mb;
            }
            if self.cpu_percent > self.peak_cpu_percent {
                self.peak_cpu_percent = self.cpu_percent;
            }
            self.cpu_sum += self.cpu_percent as f64;
            self.cpu_sample_count += 1;
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

    /// Write a full session report to `path`.
    ///
    /// Includes session-wide stats, frame time histogram, percentile
    /// breakdown, and a time-series table of periodic snapshots.
    pub fn write_report(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::fmt::Write as _;
        use std::io::Write;

        let session_duration = self.session_start.elapsed();
        let total = self.all_frame_times.len();

        let mut buf = String::with_capacity(4096);

        // ── Header ──
        writeln!(buf, "╔══════════════════════════════════════════════════════════╗").unwrap();
        writeln!(buf, "║              esox performance report                    ║").unwrap();
        writeln!(buf, "╚══════════════════════════════════════════════════════════╝").unwrap();
        writeln!(buf).unwrap();

        // ── Session overview ──
        writeln!(buf, "SESSION").unwrap();
        writeln!(buf, "  duration:       {:.1}s", session_duration.as_secs_f64()).unwrap();
        writeln!(buf, "  total frames:   {}", total).unwrap();
        if session_duration.as_secs_f64() > 0.0 {
            writeln!(buf, "  avg FPS:        {:.1}", total as f64 / session_duration.as_secs_f64()).unwrap();
        }
        writeln!(buf).unwrap();

        if total == 0 {
            writeln!(buf, "(no frames recorded)").unwrap();
            let mut f = std::fs::File::create(path)?;
            f.write_all(buf.as_bytes())?;
            return Ok(());
        }

        // ── Compute session-wide stats ──
        let mut sorted = self.all_frame_times.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sum: f64 = sorted.iter().sum();
        let avg = sum / total as f64;
        let min = sorted[0];
        let max = sorted[total - 1];
        let median = sorted[total / 2];
        let p50 = percentile(&sorted, 0.50);
        let p90 = percentile(&sorted, 0.90);
        let p95 = percentile(&sorted, 0.95);
        let p99 = percentile(&sorted, 0.99);
        let p999 = percentile(&sorted, 0.999);

        // Variance / stdev.
        let variance: f64 = sorted.iter().map(|&t| (t - avg).powi(2)).sum::<f64>() / total as f64;
        let stdev = variance.sqrt();

        // Jank: frames > 2x the median.
        let jank_threshold = median * 2.0;
        let jank_count = sorted.iter().filter(|&&t| t > jank_threshold).count();
        let jank_pct = jank_count as f64 / total as f64 * 100.0;

        writeln!(buf, "FRAME TIMES (ms)").unwrap();
        writeln!(buf, "  avg:            {avg:.3}").unwrap();
        writeln!(buf, "  stdev:          {stdev:.3}").unwrap();
        writeln!(buf, "  min:            {min:.3}").unwrap();
        writeln!(buf, "  max:            {max:.3}").unwrap();
        writeln!(buf, "  median:         {median:.3}").unwrap();
        writeln!(buf, "  p50:            {p50:.3}").unwrap();
        writeln!(buf, "  p90:            {p90:.3}").unwrap();
        writeln!(buf, "  p95:            {p95:.3}").unwrap();
        writeln!(buf, "  p99:            {p99:.3}").unwrap();
        writeln!(buf, "  p99.9:          {p999:.3}").unwrap();
        writeln!(buf).unwrap();

        writeln!(buf, "JANK (>{:.2}ms = 2x median)", jank_threshold).unwrap();
        writeln!(buf, "  count:          {jank_count}").unwrap();
        writeln!(buf, "  percent:        {jank_pct:.2}%").unwrap();
        writeln!(buf).unwrap();

        // ── CPU & memory ──
        let avg_cpu = if self.cpu_sample_count > 0 {
            self.cpu_sum / self.cpu_sample_count as f64
        } else {
            0.0
        };
        writeln!(buf, "CPU & MEMORY").unwrap();
        writeln!(buf, "  avg CPU:        {avg_cpu:.1}%").unwrap();
        writeln!(buf, "  peak CPU:       {:.1}%", self.peak_cpu_percent).unwrap();
        writeln!(buf, "  final RSS:      {:.1} MB", self.rss_mb).unwrap();
        writeln!(buf, "  peak RSS:       {:.1} MB", self.peak_rss_mb).unwrap();
        writeln!(buf, "  final VIRT:     {:.0} MB", self.virt_mb).unwrap();
        writeln!(buf).unwrap();

        // ── Histogram ──
        writeln!(buf, "FRAME TIME HISTOGRAM").unwrap();
        let buckets: &[(f64, &str)] = &[
            (1.0,   "  < 1ms   "),
            (2.0,   "  1-2ms   "),
            (4.0,   "  2-4ms   "),
            (8.0,   "  4-8ms   "),
            (16.0,  "  8-16ms  "),
            (33.3,  "  16-33ms "),
            (f64::MAX, "  33ms+   "),
        ];
        let mut bucket_counts = vec![0u64; buckets.len()];
        for &t in &self.all_frame_times {
            for (i, &(upper, _)) in buckets.iter().enumerate() {
                let lower = if i == 0 { 0.0 } else { buckets[i - 1].0 };
                if t >= lower && t < upper {
                    bucket_counts[i] += 1;
                    break;
                }
            }
        }
        let bar_max = *bucket_counts.iter().max().unwrap_or(&1);
        for (i, &(_, label)) in buckets.iter().enumerate() {
            let count = bucket_counts[i];
            let pct = count as f64 / total as f64 * 100.0;
            let bar_len = if bar_max > 0 {
                (count as f64 / bar_max as f64 * 30.0) as usize
            } else {
                0
            };
            let bar: String = "█".repeat(bar_len);
            writeln!(buf, "{label} {bar:<30} {count:>6} ({pct:>5.1}%)").unwrap();
        }
        writeln!(buf).unwrap();

        // ── Time series ──
        if !self.snapshots.is_empty() {
            writeln!(buf, "TIME SERIES (sampled every {:.0}ms)", self.sample_interval.as_millis()).unwrap();
            writeln!(buf, "  {:>8}  {:>6}  {:>8}  {:>8}  {:>8}  {:>6}  {:>5}  {:>5}",
                "time(s)", "FPS", "avg(ms)", "p99(ms)", "RSS(MB)", "CPU%", "inst", "batch").unwrap();
            writeln!(buf, "  {:─>8}  {:─>6}  {:─>8}  {:─>8}  {:─>8}  {:─>6}  {:─>5}  {:─>5}",
                "", "", "", "", "", "", "", "").unwrap();
            for s in &self.snapshots {
                writeln!(buf, "  {:>8.1}  {:>6.0}  {:>8.3}  {:>8.3}  {:>8.1}  {:>6.1}  {:>5}  {:>5}",
                    s.elapsed_s, s.fps, s.frame_time_avg_ms, s.frame_time_p99_ms,
                    s.rss_mb, s.cpu_percent, s.instance_count, s.batch_count).unwrap();
            }
        }

        let mut f = std::fs::File::create(path)?;
        f.write_all(buf.as_bytes())?;
        tracing::info!("perf report written to {}", path.display());
        Ok(())
    }
}

/// Compute a percentile from a pre-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (sorted.len() as f64 * p).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
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

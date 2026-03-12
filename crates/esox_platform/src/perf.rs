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
    cpu_time_avg_ms: f32,
    cpu_time_p99_ms: f32,
    rss_mb: f32,
    cpu_percent: f32,
    instance_count: u32,
    batch_count: u32,
}

/// Rolling performance statistics updated each frame.
#[derive(Debug, Clone)]
pub struct PerfMonitor {
    /// Rolling window of CPU frame durations (ms) — time spent in on_redraw + encode.
    cpu_frame_times: VecDeque<f64>,
    /// Rolling window of wall-clock intervals between frames (ms) — actual frame period.
    wall_intervals: VecDeque<f64>,
    /// Maximum number of frame times to keep.
    window_size: usize,
    /// How often to re-read /proc (expensive-ish).
    sample_interval: Duration,
    last_sample: Instant,
    frame_start: Instant,
    /// Timestamp of the previous begin_frame call (for wall-clock FPS).
    prev_frame_start: Option<Instant>,
    /// Previous CPU jiffies reading for delta computation.
    prev_cpu_jiffies: u64,
    prev_cpu_time: Instant,

    // ── Session-level tracking ──

    /// All CPU frame times for the entire session (for histogram / full stats).
    all_cpu_times: Vec<f64>,
    /// All wall-clock intervals for the entire session.
    all_wall_intervals: Vec<f64>,
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

    /// Actual frames per second (wall-clock, accounting for vsync waits).
    pub fps: f32,
    /// Average CPU frame time in milliseconds (time spent rendering, not waiting).
    pub cpu_time_avg_ms: f32,
    /// 99th-percentile CPU frame time in milliseconds.
    pub cpu_time_p99_ms: f32,
    /// Minimum CPU frame time in the window (ms).
    pub cpu_time_min_ms: f32,
    /// Maximum CPU frame time in the window (ms).
    pub cpu_time_max_ms: f32,
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
    /// Frames skipped due to no damage.
    pub frames_skipped: u64,
}

impl PerfMonitor {
    /// Create a new monitor with a rolling window of `window_size` frames.
    pub fn new(window_size: usize) -> Self {
        let now = Instant::now();
        Self {
            cpu_frame_times: VecDeque::with_capacity(window_size),
            wall_intervals: VecDeque::with_capacity(window_size),
            window_size,
            sample_interval: Duration::from_millis(500),
            last_sample: now,
            frame_start: now,
            prev_frame_start: None,
            prev_cpu_jiffies: read_cpu_jiffies(),
            prev_cpu_time: now,
            all_cpu_times: Vec::with_capacity(8192),
            all_wall_intervals: Vec::with_capacity(8192),
            snapshots: Vec::with_capacity(256),
            session_start: now,
            peak_rss_mb: 0.0,
            peak_cpu_percent: 0.0,
            cpu_sum: 0.0,
            cpu_sample_count: 0,
            fps: 0.0,
            cpu_time_avg_ms: 0.0,
            cpu_time_p99_ms: 0.0,
            cpu_time_min_ms: 0.0,
            cpu_time_max_ms: 0.0,
            rss_mb: 0.0,
            virt_mb: 0.0,
            cpu_percent: 0.0,
            instance_count: 0,
            batch_count: 0,
            total_frames: 0,
            frames_skipped: 0,
        }
    }

    /// Call at the start of each frame (before `on_redraw`).
    pub fn begin_frame(&mut self) {
        let now = Instant::now();
        // Track wall-clock interval between frames (actual FPS).
        if let Some(prev) = self.prev_frame_start {
            let wall_ms = now.duration_since(prev).as_secs_f64() * 1000.0;
            if self.wall_intervals.len() >= self.window_size {
                self.wall_intervals.pop_front();
            }
            self.wall_intervals.push_back(wall_ms);
            self.all_wall_intervals.push(wall_ms);
        }
        self.prev_frame_start = Some(now);
        self.frame_start = now;
    }

    /// Call at the end of each frame (after GPU submit).
    ///
    /// Pass instance/batch counts from the frame for tracking.
    pub fn end_frame(&mut self, instance_count: u32, batch_count: u32) {
        let elapsed = self.frame_start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;

        if self.cpu_frame_times.len() >= self.window_size {
            self.cpu_frame_times.pop_front();
        }
        self.cpu_frame_times.push_back(ms);
        self.all_cpu_times.push(ms);

        self.instance_count = instance_count;
        self.batch_count = batch_count;
        self.total_frames += 1;

        // Recompute rolling stats every frame.
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
                cpu_time_avg_ms: self.cpu_time_avg_ms,
                cpu_time_p99_ms: self.cpu_time_p99_ms,
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
        // CPU frame time stats.
        let n = self.cpu_frame_times.len();
        if n > 0 {
            let sum: f64 = self.cpu_frame_times.iter().sum();
            self.cpu_time_avg_ms = (sum / n as f64) as f32;

            let mut min = f64::MAX;
            let mut max = f64::MIN;
            for &t in &self.cpu_frame_times {
                if t < min { min = t; }
                if t > max { max = t; }
            }
            self.cpu_time_min_ms = min as f32;
            self.cpu_time_max_ms = max as f32;

            let mut sorted: Vec<f64> = self.cpu_frame_times.iter().copied().collect();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p99_idx = ((n as f64) * 0.99).ceil() as usize;
            self.cpu_time_p99_ms = sorted[p99_idx.min(n - 1)] as f32;
        }

        // Actual FPS from wall-clock intervals.
        let wn = self.wall_intervals.len();
        if wn > 0 {
            let wall_sum: f64 = self.wall_intervals.iter().sum();
            let wall_avg_ms = wall_sum / wn as f64;
            self.fps = if wall_avg_ms > 0.0 { 1000.0 / wall_avg_ms as f32 } else { 0.0 };
        }
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
            "FPS: {:.0}  cpu: {:.2}ms (p99: {:.2}ms)\n\
             CPU: {:.1}%  RSS: {:.1}MB  VIRT: {:.0}MB\n\
             instances: {}  batches: {}  frames: {}",
            self.fps,
            self.cpu_time_avg_ms,
            self.cpu_time_p99_ms,
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
        let total = self.all_cpu_times.len();

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
        writeln!(buf, "  frames skipped: {}", self.frames_skipped).unwrap();
        if session_duration.as_secs_f64() > 0.0 {
            writeln!(buf, "  actual FPS:     {:.1}", total as f64 / session_duration.as_secs_f64()).unwrap();
        }
        writeln!(buf).unwrap();

        if total == 0 {
            writeln!(buf, "(no frames recorded)").unwrap();
            let mut f = std::fs::File::create(path)?;
            f.write_all(buf.as_bytes())?;
            return Ok(());
        }

        // ── CPU frame time stats ──
        let mut sorted_cpu = self.all_cpu_times.clone();
        sorted_cpu.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cpu_sum: f64 = sorted_cpu.iter().sum();
        let cpu_avg = cpu_sum / total as f64;
        let cpu_min = sorted_cpu[0];
        let cpu_max = sorted_cpu[total - 1];
        let cpu_median = sorted_cpu[total / 2];
        let cpu_p90 = percentile(&sorted_cpu, 0.90);
        let cpu_p95 = percentile(&sorted_cpu, 0.95);
        let cpu_p99 = percentile(&sorted_cpu, 0.99);
        let cpu_p999 = percentile(&sorted_cpu, 0.999);

        let cpu_variance: f64 = sorted_cpu.iter().map(|&t| (t - cpu_avg).powi(2)).sum::<f64>() / total as f64;
        let cpu_stdev = cpu_variance.sqrt();

        writeln!(buf, "CPU FRAME TIME (ms) — time spent rendering each frame").unwrap();
        writeln!(buf, "  avg:            {cpu_avg:.3}").unwrap();
        writeln!(buf, "  stdev:          {cpu_stdev:.3}").unwrap();
        writeln!(buf, "  min:            {cpu_min:.3}").unwrap();
        writeln!(buf, "  max:            {cpu_max:.3}").unwrap();
        writeln!(buf, "  median:         {cpu_median:.3}").unwrap();
        writeln!(buf, "  p90:            {cpu_p90:.3}").unwrap();
        writeln!(buf, "  p95:            {cpu_p95:.3}").unwrap();
        writeln!(buf, "  p99:            {cpu_p99:.3}").unwrap();
        writeln!(buf, "  p99.9:          {cpu_p999:.3}").unwrap();
        writeln!(buf).unwrap();

        // ── Wall-clock frame interval stats ──
        if !self.all_wall_intervals.is_empty() {
            let wn = self.all_wall_intervals.len();
            let mut sorted_wall = self.all_wall_intervals.clone();
            sorted_wall.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let wall_sum: f64 = sorted_wall.iter().sum();
            let wall_avg = wall_sum / wn as f64;
            let wall_min = sorted_wall[0];
            let wall_max = sorted_wall[wn - 1];
            let wall_median = sorted_wall[wn / 2];
            let wall_p99 = percentile(&sorted_wall, 0.99);

            writeln!(buf, "WALL-CLOCK FRAME INTERVAL (ms) — actual time between frames").unwrap();
            writeln!(buf, "  avg:            {wall_avg:.3}  ({:.1} FPS)", 1000.0 / wall_avg).unwrap();
            writeln!(buf, "  min:            {wall_min:.3}  ({:.1} FPS)", 1000.0 / wall_min).unwrap();
            writeln!(buf, "  max:            {wall_max:.3}  ({:.1} FPS)", 1000.0 / wall_max).unwrap();
            writeln!(buf, "  median:         {wall_median:.3}  ({:.1} FPS)", 1000.0 / wall_median).unwrap();
            writeln!(buf, "  p99:            {wall_p99:.3}  ({:.1} FPS)", 1000.0 / wall_p99).unwrap();
            writeln!(buf).unwrap();
        }

        // ── Jank ──
        let jank_threshold = cpu_median * 2.0;
        let jank_count = sorted_cpu.iter().filter(|&&t| t > jank_threshold).count();
        let jank_pct = jank_count as f64 / total as f64 * 100.0;

        writeln!(buf, "JANK (cpu time >{:.2}ms = 2x median)", jank_threshold).unwrap();
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

        // ── Memory breakdown (Linux) ──
        let smaps = read_smaps_rollup();
        if !smaps.is_empty() {
            writeln!(buf, "MEMORY BREAKDOWN (from /proc/self/smaps_rollup)").unwrap();
            for (key, kb) in &smaps {
                writeln!(buf, "  {:<18}{:.1} MB", format!("{key}:"), *kb as f32 / 1024.0).unwrap();
            }
            writeln!(buf).unwrap();
        }

        // ── Top memory mappings ──
        let top_maps = read_top_mappings(20);
        if !top_maps.is_empty() {
            writeln!(buf, "TOP MEMORY MAPPINGS BY RSS").unwrap();
            writeln!(buf, "  {:>8}  {:>8}  {}", "RSS(kB)", "PSS(kB)", "mapping").unwrap();
            writeln!(buf, "  {:─>8}  {:─>8}  {:─>40}", "", "", "").unwrap();
            for (rss, pss, name) in &top_maps {
                writeln!(buf, "  {:>8}  {:>8}  {}", rss, pss, name).unwrap();
            }
            writeln!(buf).unwrap();
        }

        // ── Histogram (CPU frame time) ──
        writeln!(buf, "CPU FRAME TIME HISTOGRAM").unwrap();
        let buckets: &[(f64, &str)] = &[
            (0.5,   "  < 0.5ms "),
            (1.0,   "  0.5-1ms "),
            (2.0,   "  1-2ms   "),
            (4.0,   "  2-4ms   "),
            (8.0,   "  4-8ms   "),
            (16.0,  "  8-16ms  "),
            (33.3,  "  16-33ms "),
            (f64::MAX, "  33ms+   "),
        ];
        let mut bucket_counts = vec![0u64; buckets.len()];
        for &t in &self.all_cpu_times {
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
                "time(s)", "FPS", "cpu(ms)", "p99(ms)", "RSS(MB)", "CPU%", "inst", "batch").unwrap();
            writeln!(buf, "  {:─>8}  {:─>6}  {:─>8}  {:─>8}  {:─>8}  {:─>6}  {:─>5}  {:─>5}",
                "", "", "", "", "", "", "", "").unwrap();
            for s in &self.snapshots {
                writeln!(buf, "  {:>8.1}  {:>6.0}  {:>8.3}  {:>8.3}  {:>8.1}  {:>6.1}  {:>5}  {:>5}",
                    s.elapsed_s, s.fps, s.cpu_time_avg_ms, s.cpu_time_p99_ms,
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

/// Read key fields from /proc/self/smaps_rollup for memory breakdown.
#[cfg(target_os = "linux")]
fn read_smaps_rollup() -> Vec<(String, u64)> {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps_rollup") else {
        return Vec::new();
    };
    let keys = [
        "Rss", "Pss", "Shared_Clean", "Shared_Dirty",
        "Private_Clean", "Private_Dirty", "Swap",
    ];
    let mut result = Vec::new();
    for line in smaps.lines() {
        for &key in &keys {
            if let Some(rest) = line.strip_prefix(key) {
                if let Some(rest) = rest.strip_prefix(':') {
                    let kb = parse_kb(rest);
                    result.push((key.to_string(), kb));
                }
            }
        }
    }
    result
}

#[cfg(not(target_os = "linux"))]
fn read_smaps_rollup() -> Vec<(String, u64)> {
    Vec::new()
}

/// Parse /proc/self/smaps and return the top N mappings by RSS.
/// Returns (rss_kb, pss_kb, mapping_name).
#[cfg(target_os = "linux")]
fn read_top_mappings(n: usize) -> Vec<(u64, u64, String)> {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return Vec::new();
    };

    struct Mapping {
        name: String,
        rss: u64,
        pss: u64,
    }

    let mut mappings: Vec<Mapping> = Vec::new();
    let mut current_name = String::new();
    let mut current_rss = 0u64;
    let mut current_pss = 0u64;
    let mut in_mapping = false;

    for line in smaps.lines() {
        if line.starts_with(|c: char| c.is_ascii_hexdigit()) {
            // New mapping header — flush previous.
            if in_mapping && current_rss > 0 {
                mappings.push(Mapping {
                    name: current_name.clone(),
                    rss: current_rss,
                    pss: current_pss,
                });
            }
            // Parse mapping name from the end of the header line.
            // Format: addr perms offset dev inode pathname
            let parts: Vec<&str> = line.splitn(6, ' ').collect();
            current_name = if parts.len() >= 6 {
                parts[5].trim().to_string()
            } else {
                "[anon]".to_string()
            };
            if current_name.is_empty() {
                current_name = "[anon]".to_string();
            }
            current_rss = 0;
            current_pss = 0;
            in_mapping = true;
        } else if in_mapping {
            if let Some(rest) = line.strip_prefix("Rss:") {
                current_rss = parse_kb(rest);
            } else if let Some(rest) = line.strip_prefix("Pss:") {
                current_pss = parse_kb(rest);
            }
        }
    }
    // Flush last mapping.
    if in_mapping && current_rss > 0 {
        mappings.push(Mapping {
            name: current_name,
            rss: current_rss,
            pss: current_pss,
        });
    }

    // Merge by name (aggregate RSS/PSS for same library).
    let mut merged: std::collections::HashMap<String, (u64, u64)> = std::collections::HashMap::new();
    for m in mappings {
        let entry = merged.entry(m.name).or_insert((0, 0));
        entry.0 += m.rss;
        entry.1 += m.pss;
    }

    let mut sorted: Vec<(u64, u64, String)> = merged
        .into_iter()
        .map(|(name, (rss, pss))| (rss, pss, name))
        .collect();
    sorted.sort_by(|a, b| b.0.cmp(&a.0));
    sorted.truncate(n);
    sorted
}

#[cfg(not(target_os = "linux"))]
fn read_top_mappings(_n: usize) -> Vec<(u64, u64, String)> {
    Vec::new()
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
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks <= 0 {
        tracing::warn!("sysconf(_SC_CLK_TCK) failed, defaulting to 100");
        100
    } else {
        ticks as u64
    }
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

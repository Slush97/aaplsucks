//! Fixed timestep and time state.

use std::time::Instant;

/// Time state exposed to the game each frame.
pub struct TimeState {
    /// Fixed tick delta (e.g. 1/60 for 60Hz).
    pub tick_dt: f32,
    /// Wall-clock delta since last frame (variable).
    pub frame_dt: f32,
    /// Total elapsed time since engine start.
    pub elapsed: f32,
    /// Number of fixed ticks executed this frame.
    pub tick_count: u32,
    /// Total number of fixed ticks since engine start.
    pub total_ticks: u64,
}

/// Manages a fixed-rate timestep with accumulator.
pub(crate) struct FixedTimestep {
    #[allow(dead_code)]
    pub tick_rate: f32,
    pub tick_dt: f32,
    accumulator: f32,
    last_instant: Option<Instant>,
    start_instant: Instant,
    frame_dt: f32,
    total_ticks: u64,
    /// Cached time state for the current frame (avoids lifetime issues).
    pub time_state_cache: TimeState,
}

impl FixedTimestep {
    pub fn new(tick_rate: f32) -> Self {
        let tick_dt = 1.0 / tick_rate;
        Self {
            tick_rate,
            tick_dt,
            accumulator: 0.0,
            last_instant: None,
            start_instant: Instant::now(),
            frame_dt: 0.0,
            total_ticks: 0,
            time_state_cache: TimeState {
                tick_dt,
                frame_dt: 0.0,
                elapsed: 0.0,
                tick_count: 0,
                total_ticks: 0,
            },
        }
    }

    /// Call at the start of each frame. Returns (number_of_ticks, alpha).
    pub fn advance(&mut self) -> (u32, f32) {
        let now = Instant::now();
        let dt = match self.last_instant {
            Some(prev) => (now - prev).as_secs_f32(),
            None => self.tick_dt, // first frame: assume one tick
        };
        self.last_instant = Some(now);
        self.frame_dt = dt;

        // Cap to prevent spiral of death (max 250ms worth of ticks).
        self.accumulator += dt.min(0.25);

        let mut ticks = 0u32;
        while self.accumulator >= self.tick_dt {
            self.accumulator -= self.tick_dt;
            ticks += 1;
            self.total_ticks += 1;
        }

        let alpha = self.accumulator / self.tick_dt;
        (ticks, alpha)
    }

    pub fn time_state(&self, tick_count: u32) -> TimeState {
        TimeState {
            tick_dt: self.tick_dt,
            frame_dt: self.frame_dt,
            elapsed: self.start_instant.elapsed().as_secs_f32(),
            tick_count,
            total_ticks: self.total_ticks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_frame_produces_one_tick() {
        let mut ts = FixedTimestep::new(60.0);
        let (ticks, _alpha) = ts.advance();
        assert_eq!(ticks, 1);
    }

    #[test]
    fn spiral_of_death_capped() {
        let mut ts = FixedTimestep::new(60.0);
        // Simulate first frame
        ts.advance();
        // Force a huge gap
        ts.last_instant = Some(Instant::now() - std::time::Duration::from_secs(2));
        let (ticks, _) = ts.advance();
        // 250ms cap / (1/60) ≈ 15 ticks max
        assert!(ticks <= 15, "got {ticks} ticks, expected <= 15");
    }
}

//! Widget ID utilities — FNV-1a hashing for zero-allocation u64 IDs.

/// FNV-1a 64-bit hash. const fn — computable at compile time.
pub const fn fnv1a(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x00000100000001b3);
        i += 1;
    }
    hash
}

/// Runtime version for dynamically-constructed strings.
pub fn fnv1a_runtime(s: &str) -> u64 {
    fnv1a(s)
}

/// Mix a u64 (e.g. job_id) into an existing hash seed.
/// Use: `fnv1a_mix(id!("open_"), job_id)` for per-job button IDs.
pub fn fnv1a_mix(seed: u64, val: u64) -> u64 {
    let mut h = seed;
    for byte in val.to_le_bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    h
}

/// XOR salt used to derive hover-animation IDs from widget IDs.
/// Chosen so collisions with any plausible widget ID string are negligible.
pub const HOVER_SALT: u64 = 0x9e3779b97f4a7c15;

/// Compile-time widget ID from a string literal.
/// `id!("my_widget")` → a `u64` constant, zero runtime cost.
#[macro_export]
macro_rules! id {
    ($s:literal) => {{
        const _ID: u64 = $crate::id::fnv1a($s);
        _ID
    }};
}

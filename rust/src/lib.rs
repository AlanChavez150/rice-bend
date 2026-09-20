//! rice-bend-core: Rust implementation of the rice-bend MGS solver hot path.
//!
//! The Python implementation (src/rice_bend/mgs.py, rs.py) is THE reference —
//! byte-hash pinned by scripts/characterize.sh and never modified on this
//! crate's account. This crate is a second, tolerance-validated engine for the
//! grid sweep's compute: per-candidate final losses must agree within rtol 1e-3
//! and the sweep-level results (argmin cell, top-candidate set) must match
//! exactly. Bit-identity is out of reach by design (rustfft != pocketfft) and
//! is NOT the bar. See rust/README.md for the work split and gates.
//!
//! Module map (one work package per pair):
//!   fft.rs + conv.rs        — FFT/convolution layer (scipy-exact "same" conv,
//!                             c64 quantization)   [gate: tests/conv_parity.rs]
//!   solve.rs + warm_start.rs — solver core (descent + backtracking +
//!                             warm start)         [gate: tests/solve_parity.rs]
//!   types.rs + py.rs        — data interface + PyO3 boundary
//!                                                 [gate: rust/python/parity.py]
//!
//! `cargo test` / `cargo bench` build only the pure-Rust core; the `python`
//! feature (enabled by maturin, see pyproject.toml) adds the extension module.

pub mod conv;
pub mod fft;
pub mod solve;
pub mod types;
pub mod warm_start;

#[cfg(feature = "python")]
mod py;

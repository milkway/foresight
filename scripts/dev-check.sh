#!/usr/bin/env bash
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# The same gates with the optional tsorder feature.
cargo clippy --all-targets --features tsorder -- -D warnings
cargo test --features tsorder
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features tsorder

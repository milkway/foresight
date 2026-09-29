#!/usr/bin/env bash
set -euo pipefail

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

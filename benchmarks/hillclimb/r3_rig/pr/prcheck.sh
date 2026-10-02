#!/bin/bash
# usage: prcheck.sh PATCH  -> ~/hc/prcheck.log ; mirrors CI's Rust legs on this box
exec > ~/hc/prcheck.log 2>&1
source ~/.cargo/env
cd ~/turbovec
git checkout -f -q main && git reset -q --hard ccab9f32
git clean -fdq turbovec/examples turbovec/src benchmarks
git apply "$1" || { echo PATCH_FAILED; echo ALLDONE; exit 1; }
sum() { grep -E "^test result|FAILED|panicked|^error|^warning" | sort | uniq -c | sort -rn | head -25; }
echo "== release, toggle off"; cargo test -p turbovec --release --locked 2>&1 | sum
echo "== release, toggle on";  TURBOVEC_2BIT_PLANES=1 cargo test -p turbovec --release --locked 2>&1 | sum
echo "== debug lib+suites"
args=(); for f in turbovec/tests/*.rs; do n=$(basename "$f" .rs); [ "$n" = io_v6 ] && continue; args+=(--test "$n"); done
cargo test -p turbovec --locked --lib "${args[@]}" 2>&1 | sum
echo "== clippy"
rustup toolchain install 1.97.0 --profile minimal --component clippy >/dev/null 2>&1
FLAGS=(-D warnings -A clippy::assertions_on_constants -A clippy::doc_lazy_continuation -A clippy::empty_line_after_doc_comments -A clippy::excessive_precision -A clippy::explicit_counter_loop -A clippy::items_after_test_module -A clippy::len_zero -A clippy::manual_checked_ops -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of -A clippy::manual_repeat_n -A clippy::needless_range_loop -A clippy::needless_return -A clippy::neg_cmp_op_on_partial_ord -A clippy::ptr_arg -A clippy::redundant_locals -A clippy::too_many_arguments -A clippy::type_complexity -A clippy::unusual_byte_groupings -A clippy::useless_vec)
cargo +1.97.0 clippy --workspace --all-targets --locked -- "${FLAGS[@]}" 2>&1 | grep -E "^(error|warning)" -A6 | head -80
echo "clippy rc=${PIPESTATUS[0]}"
echo ALLDONE

#!/bin/bash
# CI's Rust legs for a candidate, with each planes switch on, then a smoke.
# usage: checks.sh ARCH TAG  -> ~/hc/r4/TAG_checks.log
ARCH=$1; TAG=$2; O=~/hc/r4; exec > $O/${TAG}_checks.log 2>&1
cd ~/hc; bash build_so2.sh $TAG ~/hc/$TAG.patch; tail -1 ~/hc_build_so2.log
source ~/.cargo/env; cd ~/turbovec
sum() { grep -E "^test result|FAILED|panicked|^error" $1 | sort | uniq -c | sort -rn | grep -v "ok\. " | head; echo "  ok results: $(grep -c "test result: ok" $1)"; }
for mode in "X=0" "TURBOVEC_4BIT_PLANES=1" "TURBOVEC_2BIT_PLANES=1"; do env $mode cargo test -p turbovec --release --locked > /tmp/t.log 2>&1; echo "== release $mode exit=$?"; sum /tmp/t.log; done
args=(); for f in turbovec/tests/*.rs; do n=$(basename "$f" .rs); [ "$n" = io_v6 ] && continue; args+=(--test "$n"); done
cargo test -p turbovec --locked --lib "${args[@]}" > /tmp/t.log 2>&1; echo "== debug exit=$?"; sum /tmp/t.log
grep -E "cache bytes" /tmp/t.log | head -3
FLAGS=(-D warnings -A clippy::assertions_on_constants -A clippy::doc_lazy_continuation -A clippy::empty_line_after_doc_comments -A clippy::excessive_precision -A clippy::explicit_counter_loop -A clippy::items_after_test_module -A clippy::len_zero -A clippy::manual_checked_ops -A clippy::manual_div_ceil -A clippy::manual_is_multiple_of -A clippy::manual_repeat_n -A clippy::needless_range_loop -A clippy::needless_return -A clippy::neg_cmp_op_on_partial_ord -A clippy::ptr_arg -A clippy::redundant_locals -A clippy::too_many_arguments -A clippy::type_complexity -A clippy::unusual_byte_groupings -A clippy::useless_vec)
rustup toolchain install 1.97.0 --profile minimal --component clippy >/dev/null 2>&1
cargo +1.97.0 clippy --workspace --all-targets --locked -- "${FLAGS[@]}" 2>&1 | grep -E "^(error|warning)" -A7 | head -40; echo "clippy rc=${PIPESTATUS[0]}"
cd ~/hc; bash smoke.sh $ARCH base4 $TAG:planes | tail -1
echo CHECKS_DONE

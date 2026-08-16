//! Stale-temp sweep, driven through a real save (#299, #488).
//!
//! These live in their own test binary, apart from the rest of the I/O
//! hardening suite, because the sweep is **best-effort by design** and
//! silently declines under contention.
//!
//! `claim_first_sweep` takes `SWEPT` with `try_lock`, not `lock`: a fork
//! landing while another thread holds it would leave the child with a
//! permanently locked mutex and hang its first `write`. Declining just
//! skips an opportunistic sweep, which is the right trade for the
//! product — but it means any *other* thread saving at that instant can
//! make a sweep assertion fail for a reason that has nothing to do with
//! the sweep's logic. In the shared `io_hardening` binary, where ~40
//! tests run in parallel and many of them save, that is a live race:
//! running that suite 40 times reproduced a spurious failure, and the
//! test that lost varied between runs.
//!
//! `SWEPT` is a process-global static, and each integration test file is
//! its own process, so isolating these here means the only saves in this
//! process are the ones below — and `SERIAL` keeps even those from
//! overlapping. Nothing here waits on the product's mutex; the tests
//! simply stop competing for it.

use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;


/// Serializes the saves in this file. See the module comment: two
/// concurrent saves let one lose the `try_lock` and skip its sweep.
static SERIAL: Mutex<()> = Mutex::new(());

/// Take [`SERIAL`], ignoring poisoning — a panicking test has already
/// failed on its own assertion, and poisoning the rest of the file on top
/// of it just replaces a real failure with a confusing one.
///
/// Private on purpose: the guard is taken by the two save helpers below,
/// never by a test. Serialization that each test has to remember is one
/// forgotten line away from the flakiness this file exists to remove, so
/// the only ways to save here both hold it. (Also why it must not be
/// taken twice on one thread — `Mutex` is not reentrant.)
fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-{}-{}", nonce, name));
    std::fs::create_dir(&p).unwrap();
    p
}


/// Write a small valid index to `path`, serialized against every other
/// save in this file (see [`serial`]).
fn write_good_tv(path: &PathBuf) {
    let _guard = serial();
    let mut idx = turbovec::TurboQuantIndex::new(32, 4).unwrap();
    idx.add(&vec![0.25f32; 32 * 2]);
    idx.write(path).unwrap();
}

#[cfg(unix)]
fn set_mtime(p: &std::path::Path, t: std::time::SystemTime) {
    use std::os::unix::ffi::OsStrExt;
    // `utimensat` takes `struct timespec`, whose members are both `long`
    // — i64 on every 64-bit unix. `utimes`/`struct timeval` would need
    // `suseconds_t`, which is i32 on macOS and i64 on 64-bit glibc; a
    // fixed i32 there leaves four bytes of padding the kernel reads as
    // the high half of tv_usec, and the call fails with EINVAL.
    #[repr(C)]
    struct Ts {
        sec: i64,
        nsec: i64,
    }
    const AT_FDCWD: i32 = -100;
    unsafe extern "C" {
        fn utimensat(dirfd: i32, path: *const std::ffi::c_char, times: *const Ts, flags: i32)
            -> i32;
    }
    let secs = t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let times = [Ts { sec: secs, nsec: 0 }, Ts { sec: secs, nsec: 0 }];
    let c = std::ffi::CString::new(p.as_os_str().as_bytes()).unwrap();
    let rc = unsafe { utimensat(AT_FDCWD, c.as_ptr(), times.as_ptr(), 0) };
    assert_eq!(rc, 0, "utimensat failed: {}", std::io::Error::last_os_error());
}

/// Save a real index through `TurboQuantIndex::write`, serialized the
/// same way — the sweep runs inside the save either way, so every save in
/// this file has to hold [`SERIAL`], not just the `io::write` ones.
fn save_index(idx: &turbovec::TurboQuantIndex, path: &PathBuf) {
    let _guard = serial();
    idx.write(path).unwrap();
}

/// Save through `sync()` — the v7 container — serialized like the rest.
/// Only the full-rewrite path (a first sync, a sync after `calibrate`, a
/// new path, or a change set too big for one header) stages through a
/// temp and sweeps; an incremental sync writes in place and never does.
fn sync_index(idx: &mut turbovec::TurboQuantIndex, path: &PathBuf) {
    let _guard = serial();
    idx.sync(path).unwrap();
}

/// `sweep_stale_tmps` has good unit coverage in `io.rs`, but every one of
/// those tests calls it directly. Deleting the `sweep_stale_tmps(path)`
/// call from `write_atomic` / `write_atomic_parallel` — the whole fix —
/// leaves all of them green: the sweep would still be correct, just never
/// invoked. This drives it the way a user does, through `write`.
///
/// The planted leftovers mimic what a SIGKILL between temp creation and
/// rename leaves behind: a full-size `<dest>.tmp.{pid}.{seq}.{rand}`
/// sibling nothing else ever removes.
#[test]
fn a_real_save_sweeps_a_leaked_temp_sibling() {
    use std::time::{Duration, SystemTime};
    let dir = temp_dir("sweep-on-save");
    let dest = dir.join("index.tv");

    // Two hours old — past the sweep's one-hour staleness bar, which
    // exists so a live writer's in-flight temp is never touched.
    let old = SystemTime::now() - Duration::from_secs(2 * 60 * 60);
    let plant = |name: &str, age: Option<SystemTime>| {
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(&[0u8; 4096]).unwrap();
        if let Some(t) = age {
            f.set_times(std::fs::FileTimes::new().set_modified(t)).unwrap();
        }
        p
    };
    let leaked = plant("index.tv.tmp.4242.0.deadbeef", Some(old));
    let leaked_legacy = plant("index.tv.tmp.4242.1", Some(old));
    let in_flight = plant("index.tv.tmp.4242.2.deadbeef", None);
    let other_dest = plant("other.tv.tmp.4242.0.deadbeef", Some(old));
    let unrelated = plant("index.tv.tmp.notes", Some(old));

    let mut idx = turbovec::TurboQuantIndex::new(32, 4).unwrap();
    let v: Vec<f32> = (0..64 * 32).map(|i| (i % 97) as f32 / 97.0 - 0.5).collect();
    idx.add(&v);
    save_index(&idx, &dest);

    assert!(!leaked.exists(), "save did not sweep the leaked temp sibling");
    assert!(!leaked_legacy.exists(), "save did not sweep the legacy-pattern leftover");
    assert!(in_flight.exists(), "save swept a fresh temp — a live writer's could be next");
    assert!(other_dest.exists(), "save swept another destination's temp");
    assert!(unrelated.exists(), "save swept a name it did not create");
    // And the save itself worked.
    turbovec::TurboQuantIndex::load(&dest).unwrap();
    std::fs::remove_dir_all(&dir).ok();
}

/// #488: `tmp_sibling` truncates the destination's basename when
/// `base + suffix` would exceed NAME_MAX, but the sweep matched on the
/// untruncated basename — so past ~234 bytes the #299 reclaim was a
/// permanent no-op and a crash-looping writer's temps accumulated
/// forever.
///
/// The sweep runs once per destination per process, so the leaked temp
/// has to be planted BEFORE the first save to that path — saving first
/// claims the memo and the later save sweeps nothing, which is what made
/// an earlier version of this test pass for the wrong reason.
///
/// Unix-only: backdating an mtime needs a syscall std does not expose.
#[cfg(unix)]
#[test]
fn a_leaked_temp_is_swept_for_a_long_destination_name() {
    fn plant_then_save(tag: &str, base_len: usize) -> (bool, bool) {
        let dir = temp_dir(&format!("sweep-{tag}"));
        let name = format!("{}.tv", "x".repeat(base_len.saturating_sub(3)));
        let dest = dir.join(&name);

        // The exact name `tmp_sibling` produces for a dead pid, using the
        // same NAME_MAX rule — so this is what a killed writer leaves.
        const NAME_MAX: usize = 255;
        let suffix = ".tmp.99999.7.deadbeef";
        let stem = if name.len() + suffix.len() <= NAME_MAX {
            name.clone()
        } else {
            name[..NAME_MAX - suffix.len()].to_string()
        };
        let leaked = dir.join(format!("{stem}{suffix}"));
        std::fs::write(&leaked, vec![0u8; 4096]).unwrap();
        let truncated = stem != name;

        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
        set_mtime(&leaked, old);

        // First save to this destination: the sweep's only trigger.
        write_good_tv(&dest);

        let survived = leaked.exists();
        let _ = std::fs::remove_dir_all(&dir);
        (truncated, survived)
    }

    let (trunc_short, survived_short) = plant_then_save("short", 80);
    assert!(!trunc_short, "80-byte base should not truncate");
    assert!(!survived_short, "a short-name leak must be swept (control)");

    let (trunc_long, survived_long) = plant_then_save("long", 240);
    assert!(trunc_long, "240-byte base must truncate, or the test proves nothing");
    assert!(
        !survived_long,
        "a leaked temp for a long destination survived a real save (#488)"
    );
}

/// The name a long destination's *truncated* temp takes is a legal
/// filename in its own right, so it is byte-identical to the temp a
/// shorter destination — one whose whole basename is that prefix —
/// creates untruncated. Both land on NAME_MAX, so nothing about the
/// name separates them, and reclaiming it would delete a live staged
/// index rather than a leak.
#[cfg(unix)]
#[test]
fn the_sweep_spares_a_shorter_destinations_temp_that_looks_truncated() {
    use std::time::{Duration, SystemTime};
    const NAME_MAX: usize = 255;
    let suffix = ".tmp.99999.7.deadbeef";
    let rival_base = "x".repeat(NAME_MAX - suffix.len());
    let long_dest = format!("{}.tv", "x".repeat(rival_base.len() + 6));
    let tmp_name = format!("{rival_base}{suffix}");

    // `plant` returns whether the temp survived the save. `rival` decides
    // whether the shorter destination exists alongside it.
    let plant = |tag: &str, rival: bool| -> bool {
        let dir = temp_dir(&format!("sweep-rival-{tag}"));
        if rival {
            std::fs::write(dir.join(&rival_base), b"a real index").unwrap();
        }
        let leaked = dir.join(&tmp_name);
        std::fs::write(&leaked, vec![0u8; 4096]).unwrap();
        set_mtime(&leaked, SystemTime::now() - Duration::from_secs(2 * 3600));
        write_good_tv(&dir.join(&long_dest));
        let survived = leaked.exists();
        let _ = std::fs::remove_dir_all(&dir);
        survived
    };

    // Control: with no rival destination the sweep must reclaim this
    // name. This is what makes the case below meaningful — it proves the
    // planted name really is one the sweep matches for `long_dest`,
    // rather than being derived from the NAME_MAX copy above.
    assert!(
        !plant("control", false),
        "control: this name must be swept for the long destination, or the case below proves nothing"
    );
    assert!(
        plant("rival", true),
        "the sweep deleted a shorter destination's temp (it is not ours to reclaim)"
    );
}

/// A truncation whose cut lands inside a multi-byte character emits a
/// name 1-3 bytes short of NAME_MAX, because `tmp_sibling` walks back to
/// a char boundary. Matching on `len == NAME_MAX` missed exactly those,
/// so #488 survived for non-ASCII destination names.
#[cfg(unix)]
#[test]
fn a_leaked_temp_is_swept_for_a_long_non_ascii_destination_name() {
    let dir = temp_dir("sweep-utf8");
    // 3-byte chars offset by one ASCII byte, so the 234-byte budget
    // lands *inside* a character (boundaries sit at 1 + 3k) and the cut
    // walks back to 232 — a 253-byte name, three short of NAME_MAX.
    // 83 not 84: at 84 the destination itself is 256 bytes, one past
    // NAME_MAX, and the save fails with ENAMETOOLONG before the sweep is
    // ever reached. macOS counts characters and let that pass locally;
    // Linux counts bytes and did not.
    let name = format!("a{}.tv", "€".repeat(83));
    assert!(name.len() > 240, "need a base past the truncation point");
    assert!(
        name.len() <= 255,
        "the destination must itself be creatable: {} bytes",
        name.len()
    );
    let dest = dir.join(&name);

    const NAME_MAX: usize = 255;
    let suffix = ".tmp.99999.7.deadbeef";
    let mut cut = NAME_MAX - suffix.len();
    while !name.is_char_boundary(cut) {
        cut -= 1;
    }
    let leaked = dir.join(format!("{}{}", &name[..cut], suffix));
    assert!(
        leaked.file_name().unwrap().len() < NAME_MAX,
        "this case only bites when the cut lands mid-character"
    );
    std::fs::write(&leaked, vec![0u8; 4096]).unwrap();
    set_mtime(&leaked, std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600));

    write_good_tv(&dest);
    assert!(
        !leaked.exists(),
        "a truncated temp whose cut landed mid-character survived the sweep"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The widened match must not reach another destination's temps.
#[cfg(unix)]
#[test]
fn the_sweep_does_not_touch_a_different_destinations_temp() {
    let dir = temp_dir("sweep-other");
    let dest = dir.join("index.tv");

    let other = dir.join("unrelated.tv.tmp.99999.7.deadbeef");
    std::fs::write(&other, vec![0u8; 128]).unwrap();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600);
    set_mtime(&other, old);

    write_good_tv(&dest);
    assert!(other.exists(), "swept a temp belonging to another destination");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same long-name reclaim, driven through `sync()` instead of
/// `write()`.
///
/// `io_v7::write_full` stages through the same `create_tmp` and calls the
/// same `sweep_stale_tmps`, so the #488 fix should cover it — but "shares
/// a function" is a claim about the code, not a test of it, and sync is
/// the path a repeatedly-saved index actually takes. A first sync is a
/// full rewrite, which is the sync that stages through a temp at all.
#[cfg(unix)]
#[test]
fn a_leaked_temp_is_swept_for_a_long_destination_name_on_the_sync_path() {
    let dir = temp_dir("sweep-sync-long");
    let name = format!("{}.tv", "y".repeat(240 - 3));
    let dest = dir.join(&name);

    const NAME_MAX: usize = 255;
    let suffix = ".tmp.99999.7.deadbeef";
    let stem = &name[..NAME_MAX - suffix.len()];
    assert_ne!(stem, name.as_str(), "this case needs a truncated temp name");
    let leaked = dir.join(format!("{stem}{suffix}"));
    std::fs::write(&leaked, vec![0u8; 4096]).unwrap();
    set_mtime(&leaked, std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 3600));

    let mut idx = turbovec::TurboQuantIndex::new(32, 4).unwrap();
    let v: Vec<f32> = (0..64 * 32).map(|i| (i % 97) as f32 / 97.0 - 0.5).collect();
    idx.add(&v);
    sync_index(&mut idx, &dest);

    assert!(
        !leaked.exists(),
        "sync() left a leaked temp for a long destination name (#488)"
    );
    turbovec::TurboQuantIndex::load(&dest).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

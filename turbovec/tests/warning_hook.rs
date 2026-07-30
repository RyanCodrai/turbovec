//! The post-commit durability shortfall (#365) must reach the caller
//! through a sink the caller controls, not an unconditional `eprintln!`.
//!
//! A save that got as far as the rename has committed: the new file is
//! the one readers see, so the operation is a success and cannot return
//! `Err`. But if the follow-up parent-directory fsync fails, the rename
//! itself is not on stable storage, and that must stay visible. This
//! test drives the real failure — a destination directory the process
//! may write and traverse but not open for reading, so `File::open(dir)`
//! fails with `EACCES` after the rename has already happened.

#![cfg(unix)]

use std::path::PathBuf;
use std::sync::Mutex;

use turbovec::TurboQuantIndex;

/// Messages the installed hook has seen. A `fn` hook has no captures, so
/// the sink has to be reachable statically.
static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The hook is process-global, so the tests in this file must not
/// install over each other.
static SERIAL: Mutex<()> = Mutex::new(());

fn capture(message: &str) {
    CAPTURED.lock().unwrap().push(message.to_string());
}

fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-{nonce}-{name}"));
    std::fs::create_dir(&p).unwrap();
    p
}

fn chmod(dir: &PathBuf, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// Set this to make an unprovokable durability shortfall fatal instead of
/// skipped. See [`shortfall_unprovokable`].
const REQUIRE_ENV: &str = "TURBOVEC_REQUIRE_DURABILITY_TEST";

/// True when the `0o300` mode on `dir` is not enforced — running as root,
/// or a filesystem that ignores the mode — so the post-rename directory
/// fsync cannot be made to fail and there is nothing to assert. Cleans the
/// directory up on that path, since the caller returns immediately.
///
/// The `eprintln!` below is a debugging aid, not the signal: libtest
/// captures each test's stderr and prints it only when that test *fails*,
/// so under a plain `cargo test` (no `--nocapture`, no `--show-output`)
/// the line is swallowed and a skip is indistinguishable from a pass.
/// That is where this differs from `pytest.skip`, which the Python sibling
/// uses for the same condition (turbovec-python/tests/test_index.py) and
/// which does surface in the default report. The signal here is opt-in
/// instead: an environment that knows the shortfall must be provokable —
/// a non-root job on a mode-enforcing filesystem — sets
/// `TURBOVEC_REQUIRE_DURABILITY_TEST=1` and turns the skip into a hard
/// failure, so the test cannot go vacuous unnoticed.
fn shortfall_unprovokable(dir: &PathBuf, test: &str) -> bool {
    if std::fs::read_dir(dir).is_err() {
        return false;
    }
    chmod(dir, 0o700);
    std::fs::remove_dir_all(dir).ok();

    let message = format!(
        "{test}: SKIPPED — directory mode not enforced (running as root?), \
         the durability shortfall cannot be provoked here"
    );
    let required = std::env::var(REQUIRE_ENV).map(|v| v != "0" && !v.is_empty());
    if required.unwrap_or(false) {
        panic!("{message}, but {REQUIRE_ENV} says it must be");
    }
    eprintln!("{message}");
    true
}

fn small_index() -> TurboQuantIndex {
    let dim = 32;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let vectors: Vec<f32> = (0..64 * dim).map(|i| (i % 17) as f32 - 8.0).collect();
    idx.add_2d(&vectors, dim).unwrap();
    idx
}

#[test]
fn a_durability_shortfall_reaches_an_installed_warning_hook() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = temp_dir("durability-warning");
    // Write + execute but not read: creating the temp, renaming it over
    // the destination and fsyncing the file all still work; opening the
    // directory itself for the post-rename fsync does not.
    chmod(&dir, 0o300);
    if shortfall_unprovokable(
        &dir,
        "a_durability_shortfall_reaches_an_installed_warning_hook",
    ) {
        return;
    }

    CAPTURED.lock().unwrap().clear();
    turbovec::set_warning_hook(Some(capture));

    let path = dir.join("index.tv");
    let result = small_index().write(&path);

    turbovec::set_warning_hook(None);
    let seen = CAPTURED.lock().unwrap().clone();
    chmod(&dir, 0o700);

    assert!(
        result.is_ok(),
        "a save that committed must not be reported as a failure: {result:?}",
    );
    assert!(
        path.exists(),
        "the rename committed, so the destination must exist",
    );
    let warned = seen
        .iter()
        .find(|m| m.contains(&path.display().to_string()))
        .unwrap_or_else(|| {
            panic!(
                "the durability shortfall was not delivered to the installed hook \
                 (captured: {seen:?}) — it is going somewhere the caller cannot \
                 redirect or suppress",
            )
        });
    assert!(
        warned.contains("power loss"),
        "the warning must say what was lost, got: {warned}",
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The default sink must stay noisy: with no hook installed the warning
/// still goes to stderr, so a caller that does nothing does not silently
/// lose it (#365). Asserting on stderr from inside the test process is
/// not possible without capturing the fd, so this asserts the contract
/// that makes the default reachable — installing `None` restores it, and
/// a previously installed hook stops receiving.
#[test]
fn clearing_the_hook_restores_the_default_sink() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    static AFTER_CLEAR: Mutex<Vec<String>> = Mutex::new(Vec::new());
    fn record(message: &str) {
        AFTER_CLEAR.lock().unwrap().push(message.to_string());
    }

    turbovec::set_warning_hook(Some(record));
    turbovec::set_warning_hook(None);

    let dir = temp_dir("durability-default");
    chmod(&dir, 0o300);
    if shortfall_unprovokable(&dir, "clearing_the_hook_restores_the_default_sink") {
        return;
    }
    let path = dir.join("index.tv");
    let result = small_index().write(&path);
    chmod(&dir, 0o700);
    assert!(result.is_ok(), "{result:?}");
    assert!(
        AFTER_CLEAR.lock().unwrap().is_empty(),
        "a cleared hook must stop receiving warnings",
    );
    std::fs::remove_dir_all(&dir).ok();
}

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
    if std::fs::read_dir(&dir).is_ok() {
        // Running as root (or on a filesystem that ignores the mode) —
        // the shortfall cannot be provoked, so there is nothing to assert.
        chmod(&dir, 0o700);
        std::fs::remove_dir_all(&dir).ok();
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
    if std::fs::read_dir(&dir).is_ok() {
        chmod(&dir, 0o700);
        std::fs::remove_dir_all(&dir).ok();
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

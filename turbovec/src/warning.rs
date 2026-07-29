//! Non-fatal diagnostics: conditions a caller must be able to see but
//! that are not failures of the operation that produced them.
//!
//! There is exactly one today — the post-commit durability shortfall in
//! [`io`](crate::io) (#365): the rename that publishes a saved index has
//! already succeeded, so the save is not an error, but the caller is
//! entitled to know that the rename may not survive power loss.
//!
//! A library must not decide unilaterally that stderr is the right place
//! for that. A service that captures its logs structurally never sees a
//! bare `eprintln!`, and a caller who does not want the line has no way
//! to turn it off. So the sink is a process-global hook the embedder
//! installs — the shape `std::panic::set_hook` uses for the same problem
//! — with a stderr default so that doing nothing still shows the
//! warning rather than dropping it.
//!
//! Why not the `log`/`tracing` facade: turbovec has no logging
//! dependency today, and a facade with no logger installed *discards*
//! the record silently, which is the one outcome #365 rules out. A hook
//! forwards into whichever facade the embedder actually uses in three
//! lines, and costs downstreams nothing.
//!
//! The slot is a single [`AtomicPtr`], not a `Mutex`/`OnceLock`: reading
//! it is one relaxed load that can never block, so a warning emitted in
//! a process that has forked behaves the same as in one that has not
//! (see [`codebook`](crate::codebook) for the same requirement stated at
//! length). It is also replaceable, which set-once cells are not.

use std::sync::atomic::{AtomicPtr, Ordering};

/// Sink for a non-fatal diagnostic. Receives the message body with no
/// trailing newline and no `turbovec:` prefix.
///
/// It may be called from any thread, including a rayon worker inside a
/// save, and it must not panic or unwind into the caller.
pub type WarningHook = fn(&str);

/// Null means "no hook installed"; any other value is a `WarningHook`
/// that was cast to a data pointer by [`set_warning_hook`].
static HOOK: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// Route non-fatal diagnostics to `hook`, replacing any previous one;
/// `None` restores the stderr default.
///
/// Install it once during startup, before other threads exist: a hook
/// swapped concurrently with an in-flight warning may still see the old
/// one deliver that message. Passing a hook that suppresses everything
/// (`|_| {}`) is the supported way to silence the library.
///
/// ```
/// fn to_my_log(message: &str) {
///     eprintln!("[turbovec] {message}");
/// }
/// turbovec::set_warning_hook(Some(to_my_log));
/// turbovec::set_warning_hook(None); // back to the default
/// ```
pub fn set_warning_hook(hook: Option<WarningHook>) {
    let ptr = match hook {
        Some(f) => f as *const () as *mut (),
        None => std::ptr::null_mut(),
    };
    HOOK.store(ptr, Ordering::Release);
}

/// Deliver `message` to the installed hook, or to stderr if there is
/// none.
pub(crate) fn warn(message: &str) {
    let ptr = HOOK.load(Ordering::Acquire);
    if ptr.is_null() {
        eprintln!("turbovec: warning: {message}");
        return;
    }
    // SAFETY: `HOOK` is only ever written by `set_warning_hook`, which
    // stores either null (handled above) or a `WarningHook` cast to
    // `*mut ()`. Function and data pointers are the same width on every
    // target this crate compiles for (64-bit only, enforced in lib.rs),
    // so the round trip recovers exactly the pointer that was stored.
    let hook: WarningHook = unsafe { std::mem::transmute::<*mut (), WarningHook>(ptr) };
    hook(message);
}

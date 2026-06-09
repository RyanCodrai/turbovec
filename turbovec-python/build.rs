fn main() {
    // Emit the platform-correct linker arguments for a Python extension
    // module. On macOS this passes `-undefined dynamic_lookup` so symbols
    // from the Python interpreter (e.g. `Py_True`) resolve at load time
    // instead of failing the link step. Without it, a plain `cargo build`
    // on macOS fails with "symbol(s) not found for architecture arm64"
    // (issue #92). Building via maturin already injects these args; this
    // makes a bare `cargo build` work too.
    pyo3_build_config::add_extension_module_link_args();
}

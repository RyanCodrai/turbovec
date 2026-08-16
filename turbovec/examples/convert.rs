//! Convert an index file between format versions.
//!
//! ```text
//! cargo run --example convert -- <input> <output> <v5|v6|v7>
//! cargo run --example convert -- <input>            # report the version
//! ```
//!
//! The version of the input is detected; any version converts to any
//! other, for both `.tv` and `.tvim`.

use std::path::Path;
use std::process::ExitCode;

use turbovec::convert::{self, Kind, Version};

fn parse(v: &str) -> Option<Version> {
    match v.trim().to_ascii_lowercase().as_str() {
        "v5" | "5" => Some(Version::V5),
        "v6" | "6" => Some(Version::V6),
        "v7" | "7" => Some(Version::V7),
        _ => None,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.len() {
        1 => match convert::version_of(Path::new(&args[0])) {
            Ok((v, k)) => {
                let kind = match k {
                    Kind::Plain => "positional (.tv)",
                    Kind::IdMapped => "id-mapped (.tvim)",
                };
                println!("{}: {v} {kind}", args[0]);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{}: {e}", args[0]);
                ExitCode::FAILURE
            }
        },
        3 => {
            let Some(to) = parse(&args[2]) else {
                eprintln!("unknown target version {:?} (expected v5, v6 or v7)", args[2]);
                return ExitCode::FAILURE;
            };
            let (src, dst) = (Path::new(&args[0]), Path::new(&args[1]));
            let from = convert::version_of(src);
            match convert::convert_file(src, dst, to) {
                Ok(()) => {
                    let from = from.map(|(v, _)| v.to_string()).unwrap_or_default();
                    println!("{} ({from}) -> {} ({to})", args[0], args[1]);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("converting {} -> {}: {e}", args[0], args[1]);
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("usage: convert <input> [<output> <v5|v6|v7>]");
            ExitCode::FAILURE
        }
    }
}

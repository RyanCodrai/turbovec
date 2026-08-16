//! Every version to every version, both index kinds.
//!
//! The bar is not "it parses": a conversion must preserve what the file
//! is *for*, so each case compares search results against the original
//! index, and the id-mapped cases compare resolved ids too.

use turbovec::convert::{self, Image, Kind, Version};
use turbovec::{IdMapIndex, TurboQuantIndex};

const ALL: [Version; 3] = [Version::V5, Version::V6, Version::V7];

fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * dim];
    let mut s = seed | 1;
    for x in v.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = ((s >> 40) as f32 / (1u64 << 23) as f32) - 0.5;
    }
    v
}

fn build(dim: usize, n: usize, calibrated: bool) -> TurboQuantIndex {
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    if calibrated {
        idx.calibrate(&rows(1024, dim, 3)).unwrap();
    }
    idx.add(&rows(n, dim, 1));
    idx
}

/// Convert `bytes` to `to`, then read it back as an index and compare
/// search results with `want`.
fn assert_plain_parity(bytes: &[u8], to: Version, want: &TurboQuantIndex, dim: usize) {
    let image = convert::read(bytes).expect("read source");
    let out = convert::write(&image, to).unwrap_or_else(|e| panic!("write {to}: {e}"));
    let (v, k) = convert::detect(&out).expect("detect output");
    assert_eq!(v, to, "output is not {to}");
    assert_eq!(k, Kind::Plain);

    // Round the output back through the reader: for v7 that is the real
    // loader, for v5/v6 it is this module, which is the only thing that
    // still reads them.
    let back = convert::read(&out).expect("read back");
    assert_eq!(back.n_vectors, want.len());
    assert_eq!(back.dim, dim);

    let idx = TurboQuantIndex::from_parts(
        Some(back.dim),
        back.bit_width,
        back.n_vectors,
        back.packed_codes.clone(),
        back.scales.clone(),
        back.tqplus_shift.clone(),
        back.tqplus_scale.clone(),
    )
    .expect("rebuild index");

    let q = rows(4, dim, 99);
    let got = idx.search(&q, 10);
    let expect = want.search(&q, 10);
    assert_eq!(got.indices, expect.indices, "indices differ after -> {to}");
    assert_eq!(got.scores, expect.scores, "scores differ after -> {to}");
}

#[test]
fn every_version_converts_to_every_other_for_tv() {
    let dim = 64;
    let want = build(dim, 200, false);
    // One source file per version, produced by the converter itself, so
    // the matrix does not depend on having fixtures for retired formats.
    let base = convert::read(&want.to_bytes()).unwrap();
    for from in ALL {
        let src = convert::write(&base, from).unwrap_or_else(|e| panic!("seed {from}: {e}"));
        assert_eq!(convert::detect(&src).unwrap().0, from);
        for to in ALL {
            assert_plain_parity(&src, to, &want, dim);
        }
    }
}

#[test]
fn every_version_converts_to_every_other_for_tvim() {
    let dim = 64;
    let n = 200;
    let ids: Vec<u64> = (0..n as u64).map(|i| i * 7 + 11).collect();
    let mut want = IdMapIndex::new(dim, 4).unwrap();
    want.add_with_ids(&rows(n, dim, 1), &ids).unwrap();

    let base = convert::read(&want.to_bytes()).unwrap();
    assert_eq!(base.ids.as_deref(), Some(&ids[..]));

    let q = rows(4, dim, 99);
    let expect = want.search(&q, 10);

    for from in ALL {
        let src = convert::write(&base, from).unwrap_or_else(|e| panic!("seed {from}: {e}"));
        assert_eq!(convert::detect(&src).unwrap(), (from, Kind::IdMapped));
        for to in ALL {
            let image = convert::read(&src).unwrap();
            let out = convert::write(&image, to).unwrap();
            assert_eq!(convert::detect(&out).unwrap(), (to, Kind::IdMapped));

            let back = convert::read(&out).unwrap();
            assert_eq!(back.ids.as_deref(), Some(&ids[..]), "{from} -> {to} lost ids");

            let inner = TurboQuantIndex::from_parts(
                Some(back.dim),
                back.bit_width,
                back.n_vectors,
                back.packed_codes,
                back.scales,
                back.tqplus_shift,
                back.tqplus_scale,
            )
            .unwrap();
            let bytes = convert::write(
                &convert::read(&inner.to_bytes()).map(|mut i| {
                    i.ids = Some(ids.clone());
                    i
                })
                .unwrap(),
                Version::V7,
            )
            .unwrap();
            let m = IdMapIndex::from_bytes(&bytes).unwrap();
            assert_eq!(m.search(&q, 10), expect, "{from} -> {to} changed results");
        }
    }
}

#[test]
fn calibration_survives_every_conversion() {
    let dim = 64;
    let want = build(dim, 128, true);
    assert!(!want.tqplus_shift().is_empty(), "fixture must be calibrated");
    let base = convert::read(&want.to_bytes()).unwrap();
    for from in ALL {
        let src = convert::write(&base, from).unwrap();
        for to in ALL {
            let out = convert::write(&convert::read(&src).unwrap(), to).unwrap();
            let back = convert::read(&out).unwrap();
            assert_eq!(back.tqplus_shift, base.tqplus_shift, "{from} -> {to} shift");
            assert_eq!(back.tqplus_scale, base.tqplus_scale, "{from} -> {to} scale");
        }
    }
}

/// A v7 file can hold a lazy index; v5 and v6 have no sentinel for one,
/// so the converter must say that rather than write something a reader
/// would misinterpret.
#[test]
fn a_lazy_index_cannot_be_written_as_v5_or_v6() {
    let lazy = TurboQuantIndex::new_lazy(4).unwrap();
    let image = convert::read(&lazy.to_bytes()).unwrap();
    assert_eq!(image.dim, 0);
    assert!(convert::write(&image, Version::V7).is_ok());
    for v in [Version::V5, Version::V6] {
        let e = convert::write(&image, v).expect_err("must refuse");
        assert!(
            e.to_string().contains("no committed dimension"),
            "unhelpful: {e}"
        );
    }
}

#[test]
fn pre_v5_files_and_junk_are_named_rather_than_guessed() {
    let mut v4 = b"TVPI".to_vec();
    v4.push(4);
    v4.extend_from_slice(&[0u8; 32]);
    let e = convert::read(&v4).expect_err("v4 must be refused");
    assert!(e.to_string().contains("pre-v5 rotation"), "got: {e}");
    assert!(e.to_string().contains("Rebuild"), "got: {e}");

    let mut v1 = b"TVPI".to_vec();
    v1.push(1);
    v1.extend_from_slice(&[0u8; 32]);
    let e = convert::read(&v1).expect_err("v1 must be refused");
    assert!(e.to_string().contains("0.4.3"), "got: {e}");

    let e = convert::read(b"\x7fELF\x02\x01\x01\x00").expect_err("junk must be refused");
    assert!(e.to_string().contains("not a turbovec index"), "got: {e}");
}

#[test]
fn convert_file_writes_atomically_and_detects_the_version() {
    let dir = std::env::temp_dir().join(format!("tvconv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("in.tv");
    let dst = dir.join("out.tv");

    let idx = build(64, 96, false);
    idx.write(&src).unwrap();
    assert_eq!(convert::version_of(&src).unwrap(), (Version::V7, Kind::Plain));

    convert::convert_file(&src, &dst, Version::V6).unwrap();
    assert_eq!(convert::version_of(&dst).unwrap(), (Version::V6, Kind::Plain));
    // No temp left behind.
    let strays: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp."))
        .collect();
    assert!(strays.is_empty(), "temp files left: {strays:?}");

    // And back again.
    convert::convert_file(&dst, &src, Version::V7).unwrap();
    assert_eq!(convert::version_of(&src).unwrap(), (Version::V7, Kind::Plain));
    let reloaded = TurboQuantIndex::load(&src).unwrap();
    let q = rows(2, 64, 5);
    assert_eq!(reloaded.search(&q, 8).indices, idx.search(&q, 8).indices);
    std::fs::remove_dir_all(&dir).ok();
}

/// Converting is a re-container, not a re-quantize: the stored codes
/// must come out byte-identical, whatever route they took.
#[test]
fn codes_are_never_re_encoded() {
    let dim = 128;
    let want = build(dim, 64, true);
    let base: Image = convert::read(&want.to_bytes()).unwrap();
    for from in ALL {
        for to in ALL {
            let a = convert::write(&base, from).unwrap();
            let b = convert::write(&convert::read(&a).unwrap(), to).unwrap();
            let back = convert::read(&b).unwrap();
            assert_eq!(
                back.packed_codes, base.packed_codes,
                "{from} -> {to} changed the stored codes"
            );
            assert_eq!(back.scales, base.scales, "{from} -> {to} changed scales");
        }
    }
}

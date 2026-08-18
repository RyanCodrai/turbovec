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

/// The lazy sentinel is not a v7 invention: v5 and v6 carry `dim == 0`
/// with no rows too, and the previous release's `write()` emitted one
/// for a store saved before its first add. So it has to convert both
/// ways, or those files cannot be brought forward.
#[test]
fn a_lazy_index_converts_in_every_direction() {
    let lazy = TurboQuantIndex::new_lazy(4).unwrap();
    let base = convert::read(&lazy.to_bytes()).unwrap();
    assert_eq!(base.dim, 0);
    assert_eq!(base.n_vectors, 0);

    for from in ALL {
        let src = convert::write(&base, from).unwrap_or_else(|e| panic!("write {from}: {e}"));
        assert_eq!(convert::detect(&src).unwrap().0, from);
        for to in ALL {
            let out = convert::write(&convert::read(&src).unwrap(), to)
                .unwrap_or_else(|e| panic!("{from} -> {to}: {e}"));
            let back = convert::read(&out).unwrap();
            assert_eq!(back.dim, 0, "{from} -> {to} lost the sentinel");
            assert_eq!(back.n_vectors, 0);
        }
    }

    // And it still loads as a lazy index through the real v7 loader.
    let v7 = convert::write(&base, Version::V7).unwrap();
    let idx = TurboQuantIndex::from_bytes(&v7).unwrap();
    assert_eq!(idx.dim_opt(), None);
}

/// `dim == 0` is only legal with no rows, whichever version claims it.
#[test]
fn the_sentinel_is_refused_when_it_claims_rows() {
    let mut idx = TurboQuantIndex::new(64, 4).unwrap();
    idx.add(&rows(32, 64, 2));
    let mut image = convert::read(&idx.to_bytes()).unwrap();
    image.dim = 0;
    for v in ALL {
        let e = convert::write(&image, v).expect_err("dim 0 with rows must be refused");
        assert!(
            e.to_string().contains("sentinel")
                || e.to_string().contains("dim 0")
                || e.to_string().contains("n_vectors=0"),
            "unhelpful for {v}: {e}"
        );
    }
}

/// A header can claim any row count; the reader must bound it against
/// the file before any size arithmetic runs on it.
#[test]
fn an_absurd_row_count_is_refused_not_multiplied_out() {
    let mut v6 = b"TVPI".to_vec();
    v6.push(6);
    v6.push(4); // bit_width
    v6.extend_from_slice(&64u32.to_le_bytes()); // dim
    v6.extend_from_slice(&u64::MAX.to_le_bytes()); // n_vectors
    v6.extend_from_slice(&[0u8; 32]);

    let e = convert::read(&v6).expect_err("an absurd row count must be refused");
    assert!(
        e.to_string().contains("rows"),
        "should name the row count: {e}"
    );
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

/// `Image` is public with public fields, so it can arrive hand-built
/// rather than from `read`. Every version must check the geometry —
/// the v7 arm inherits `from_parts`' validation, the legacy arms have
/// none of their own, and a short code buffer would otherwise be
/// written straight into a file that no reader can make sense of.
#[test]
fn a_hand_built_image_is_validated_for_every_version() {
    let good = |n: usize| Image {
        bit_width: 4,
        dim: 64,
        n_vectors: n,
        packed_codes: vec![0u8; n * 64 * 4 / 8],
        scales: vec![1.0; n],
        tqplus_shift: vec![],
        tqplus_scale: vec![],
        ids: None,
    };
    for v in ALL {
        assert!(convert::write(&good(2), v).is_ok(), "{v}: the valid case must write");

        // A row is 32 bytes at dim 64, 4-bit; supply one.
        let mut short = good(1);
        short.packed_codes = vec![0u8; 1];
        let e = convert::write(&short, v).expect_err("short codes must be refused");
        assert!(e.to_string().contains("packed_codes"), "{v}: {e}");

        let mut bad_bits = good(1);
        bad_bits.bit_width = 7;
        assert!(
            convert::write(&bad_bits, v).is_err(),
            "{v}: bit_width 7 must be refused"
        );

        let mut bad_dim = good(1);
        bad_dim.dim = 65;
        assert!(convert::write(&bad_dim, v).is_err(), "{v}: dim 65 must be refused");

        let mut huge = good(1);
        huge.dim = 1 << 24;
        assert!(convert::write(&huge, v).is_err(), "{v}: an absurd dim must be refused");
    }
}

/// A mismatched TQ+ pair must not reach the legacy writers.
///
/// They take `n_calib` from the shift array but emit both, so a pair of
/// unequal length writes a header saying "uncalibrated" followed by
/// stray floats — which land where the id table starts and come back as
/// ids, with no error anywhere. v7 caught it through `from_parts`; v5
/// and v6 had nothing.
#[test]
fn a_mismatched_calibration_pair_is_refused_before_it_corrupts_ids() {
    let dim = 64;
    let n = 2;
    let ids: Vec<u64> = vec![111, 222];
    let mut img = Image {
        bit_width: 4,
        dim,
        n_vectors: n,
        packed_codes: vec![0u8; n * dim * 4 / 8],
        scales: vec![1.0; n],
        tqplus_shift: vec![],
        tqplus_scale: vec![7.5; dim],
        ids: Some(ids.clone()),
    };
    for v in ALL {
        let e = convert::write(&img, v).expect_err("a half-empty pair must be refused");
        assert!(e.to_string().contains("tqplus"), "{v}: {e}");
    }

    // A pair of equal but wrong length is refused too.
    img.tqplus_shift = vec![0.0; dim / 2];
    img.tqplus_scale = vec![1.0; dim / 2];
    for v in ALL {
        let e = convert::write(&img, v).expect_err("a short pair must be refused");
        assert!(e.to_string().contains("calibration length"), "{v}: {e}");
    }
}

/// Calibration and ids in the same file, through every version.
///
/// The two features are written adjacently — the TQ+ trailer then the id
/// table — so an offset error in one silently reads the other. Nothing
/// covered both at once: the id-mapped matrix was uncalibrated and the
/// calibration matrix had no ids.
#[test]
fn a_calibrated_id_mapped_index_converts_in_every_direction() {
    let dim = 64;
    let n = 128;
    let ids: Vec<u64> = (0..n as u64).map(|i| i * 5 + 3).collect();
    let mut m = IdMapIndex::new(dim, 4).unwrap();
    m.calibrate(&rows(1024, dim, 8)).unwrap();
    m.add_with_ids(&rows(n, dim, 9), &ids).unwrap();

    let base = convert::read(&m.to_bytes()).unwrap();
    assert_eq!(base.tqplus_shift.len(), dim, "fixture must be calibrated");
    assert_eq!(base.ids.as_deref(), Some(&ids[..]));

    for from in ALL {
        let src = convert::write(&base, from).unwrap();
        for to in ALL {
            let out = convert::write(&convert::read(&src).unwrap(), to).unwrap();
            let back = convert::read(&out).unwrap();
            assert_eq!(back.ids.as_deref(), Some(&ids[..]), "{from} -> {to}: ids");
            assert_eq!(back.tqplus_shift, base.tqplus_shift, "{from} -> {to}: shift");
            assert_eq!(back.tqplus_scale, base.tqplus_scale, "{from} -> {to}: scale");
            assert_eq!(back.packed_codes, base.packed_codes, "{from} -> {to}: codes");
        }
    }
}

/// Too short to hold a magic and a version byte: an error, not a panic
/// from slicing past the end.
#[test]
fn a_truncated_prefix_is_an_error_not_a_panic() {
    for len in 0..5usize {
        let bytes = vec![b'T'; len];
        let e = convert::detect(&bytes).expect_err("{len} bytes must be refused");
        assert!(e.to_string().contains("too short"), "{len}: {e}");
        assert!(convert::read(&bytes).is_err());
    }
}

/// `dim > MAX_DIM` is the bound; at `>=` the largest legal index cannot
/// be written at all.
#[test]
fn an_image_at_exactly_max_dim_can_be_written() {
    let dim = turbovec::MAX_DIM;
    let img = Image {
        bit_width: 2,
        dim,
        n_vectors: 0,
        packed_codes: Vec::new(),
        scales: Vec::new(),
        tqplus_shift: Vec::new(),
        tqplus_scale: Vec::new(),
        ids: None,
    };
    for v in ALL {
        assert!(convert::write(&img, v).is_ok(), "{v}: dim {dim} is legal");
    }
}

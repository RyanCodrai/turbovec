//! Round-3 Stage 0 probe (not a test of behaviour): dump real codes, the
//! rotated/calibrated queries and the exact top-k so the sign-plane
//! shortlist can be measured offline. Run with
//! `PLANE_NPY=... PLANE_OUT=... cargo test --release -p turbovec plane_probe -- --ignored --nocapture`.
use crate::{pack, TurboQuantIndex};
use std::io::{Read, Seek, SeekFrom, Write};

fn read_rows(path: &str, start: usize, n: usize, dim: usize) -> Vec<f32> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut head = [0u8; 10];
    f.read_exact(&mut head).unwrap();
    assert_eq!(&head[..6], b"\x93NUMPY");
    assert_eq!(head[6], 1, "npy v1 header expected");
    let hlen = u16::from_le_bytes([head[8], head[9]]) as u64;
    f.seek(SeekFrom::Start(10 + hlen + (start * dim * 4) as u64)).unwrap();
    let mut bytes = vec![0u8; n * dim * 4];
    f.read_exact(&mut bytes).unwrap();
    let mut v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    for row in v.chunks_exact_mut(dim) {
        let nrm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
        for x in row.iter_mut() { *x /= nrm; }
    }
    v
}

fn dump<T: Copy>(path: &str, data: &[T]) {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
    std::fs::File::create(path).unwrap().write_all(bytes).unwrap();
}

#[test]
#[ignore]
fn plane_probe_dump() {
    let npy = std::env::var("PLANE_NPY").unwrap();
    let out = std::env::var("PLANE_OUT").unwrap();
    let env = |k: &str| std::env::var(k).unwrap().parse::<usize>().unwrap();
    let (dim, ndb, nq, qstart, k) = (env("PLANE_DIM"), env("PLANE_NDB"), env("PLANE_NQ"), env("PLANE_QSTART"), 100usize);
    let db = read_rows(&npy, 0, ndb, dim);
    let queries = read_rows(&npy, qstart, nq, dim);
    for bits in [2usize, 4] {
        for calib in [false, true] {
            let mut ix = TurboQuantIndex::new(dim, bits).unwrap();
            if calib {
                let step = ndb / 1024;
                let sample: Vec<f32> = (0..1024).flat_map(|i| db[i * step * dim..(i * step + 1) * dim].to_vec()).collect();
                ix.calibrate(&sample).unwrap();
            }
            ix.add(&db);
            let t = std::time::Instant::now();
            let res = ix.search(&queries, k);
            eprintln!("bits={bits} calib={calib} search {:?}", t.elapsed());
            let codes = pack::extract_codes_flat(ix.packed(), ndb, bits, dim);
            let rot = ix.rotation.get().unwrap();
            let mut q = queries.clone();
            let mut scratch = vec![0.0f32; dim];
            let mut bias = vec![0.0f32; nq];
            for (qi, row) in q.chunks_exact_mut(dim).enumerate() {
                rot.apply_with_scratch(row, &mut scratch);
                if !ix.tqplus_shift.is_empty() {
                    let mut bc = 0.0f64;
                    for d in 0..dim {
                        bc -= (row[d] as f64) * (ix.tqplus_shift[d] as f64);
                        row[d] /= ix.tqplus_scale[d];
                    }
                    bias[qi] = bc as f32;
                }
            }
            let tag = format!("{out}/b{bits}_c{}", calib as u8);
            dump(&format!("{tag}_codes.u8"), &codes);
            dump(&format!("{tag}_scales.f32"), &ix.scales);
            dump(&format!("{tag}_centroids.f32"), ix.centroids.get().unwrap());
            dump(&format!("{tag}_q.f32"), &q);
            dump(&format!("{tag}_bias.f32"), &bias);
            dump(&format!("{tag}_ids.i64"), &res.indices);
            dump(&format!("{tag}_scores.f32"), &res.scores);
        }
    }
}

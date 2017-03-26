#![feature(test)]
extern crate test;
extern crate ocl;

extern crate geogrid;

use std::time::{Duration, Instant};
use geogrid::util::*;
use ocl::builders::DeviceSpecifier;

#[cfg(test)]
use test::{Bencher, black_box};

const M: usize = 9000;
const N: usize = M;
const MID: usize = M * N / 2;
const SHAPE_DIM: usize = 40;
fn mock_shape(sz: usize) -> Vec<Vec<bool>> {
    let mut sp = vec![vec![false; sz]; sz];
    // Set some of them to true
    for i in 0..(sz - 1) / 2 {
        for j in 0..(sz - 1) / 2 {
            sp[i * 2][j * 2] = true;
        }
    }
    sp
}

fn dur_as_ms(dur: Duration) -> f64 {
    dur.as_secs() as f64 * 1000.0 + dur.subsec_nanos() as f64 / 1000_000.0
}
// test match of 40x40 shape on 9000x9000 maps.
#[bench]
fn bench_singlecore(_: &mut Bencher) {
    println!();
    let sp = mock_shape(SHAPE_DIM);
    let dt = vec![2; M * N];
    let start = Instant::now();
    let res = black_box(match_shape_slow(&dt, (M, N), &sp));
    println!("Time elapsed: {:.2} ms", dur_as_ms(start.elapsed()));
    println!("Verification: {:?}", &res[MID..MID + 8]);
}

#[bench]
fn bench_multicore(_: &mut Bencher) {
    println!();
    let sp = mock_shape(SHAPE_DIM);
    let dt = vec![2; M * N];
    let start = Instant::now();
    let res = black_box(match_shape(&dt, (M, N), &sp));
    black_box(match_shape(&dt, (M, N), &sp));
    println!("Time elapsed: {:.2} ms", dur_as_ms(start.elapsed()));
    println!("Verification: {:?}", &res[MID..MID + 8]);
}

#[bench]
fn bench_ocl(_: &mut Bencher) {
    println!();
    let sp = mock_shape(SHAPE_DIM);
    let dt = vec![2; M * N];
    if let Ok(all_devices) = (DeviceSpecifier::All).to_device_list(None) {
        for device in all_devices {
            let start = Instant::now();
            println!("{}", device.name());
            let res = black_box(match_shape_ocl(&dt, (M, N), &sp, &device, Some(1024)));
            println!("Time elapsed: {:.2} ms", dur_as_ms(start.elapsed()));
            println!("Verification: {:?}", &res[MID..MID + 8]);
        }
    }
}

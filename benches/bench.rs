#![feature(test)]
extern crate test;
#[cfg(feature="opencl")]
extern crate ocl;

extern crate geogrid;

use std::time::{Duration, Instant};
use geogrid::util::*;
use geogrid::util::Processor::*;

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

fn mock_dt() -> Vec<i32> {
    // Compute distance transform of an identity matrix.
    let grid = (0..M * N).map(|x| if x / N == x % N { 1 } else { 0 }).collect::<Vec<_>>();
    l1dist_transform(&grid, (M, N))
}

fn dur_as_ms(dur: Duration) -> f64 {
    dur.as_secs() as f64 * 1000.0 + dur.subsec_nanos() as f64 / 1000_000.0
}

#[bench]
fn match_singlecore(_: &mut Bencher) {
    println!();
    let sp = mock_shape(SHAPE_DIM);
    let dt = mock_dt();
    let start = Instant::now();
    let res = black_box(match_shape(&dt, (M, N), &sp, 2, SingleCore));
    println!("Time elapsed: {:.2} ms", dur_as_ms(start.elapsed()));
    println!("Verification 1: {:?}", &res[..8]);
    println!("Verification 2: {:?}", &res[MID..MID + 8]);
}

#[bench]
fn match_multicore(_: &mut Bencher) {
    println!();
    let sp = mock_shape(SHAPE_DIM);
    let dt = mock_dt();
    let start = Instant::now();
    let res = black_box(match_shape(&dt, (M, N), &sp, 2, MultiCore));
    println!("Time elapsed: {:.2} ms", dur_as_ms(start.elapsed()));
    println!("Verification 1: {:?}", &res[..8]);
    println!("Verification 2: {:?}", &res[MID..MID + 8]);
}

#[cfg(feature="opencl")]
#[bench]
fn match_ocl(_: &mut Bencher) {
    use ocl::builders::DeviceSpecifier;
    println!();
    let sp = mock_shape(SHAPE_DIM);
    let dt = mock_dt();
    if let Ok(all_devices) = (DeviceSpecifier::All).to_device_list(None) {
        for device in all_devices {
            let start = Instant::now();
            println!("{}", device.name());
            let res = black_box(match_shape(&dt, (M, N), &sp, 2, GPU(&device, 256)));
            println!("Time elapsed: {:.2} ms", dur_as_ms(start.elapsed()));
            println!("Verification 1: {:?}", &res[..8]);
            println!("Verification 2: {:?}", &res[MID..MID + 8]);
        }
    }
}

#[bench]
fn l1_dist(_: &mut Bencher) {
    println!();
    let start = Instant::now();
    let res = black_box(mock_dt());
    println!("Time elapsed: {:.2} ms", dur_as_ms(start.elapsed()));
    println!("Verification 1: {:?}", &res[..8]);
    println!("Verification 2: {:?}", &res[MID..MID + 8]);
}

extern crate geogrid;
extern crate imagefmt;
extern crate num;
extern crate quick_xml;
extern crate rayon;
extern crate stopwatch;

mod osm_parse;

use std::path::Path;

use imagefmt::ColFmt;
use imagefmt::ColType;
use num::{Num, ToPrimitive};
use rayon::prelude::*;
use stopwatch::Stopwatch;

/// Make some square shape of given side length.
pub fn build_square(s: usize) -> Vec<Vec<bool>> {
    let mut sq = vec![vec![true; s]];
    for _ in 1..s-1 {
        let mut row = vec![false; s];
        row[0] = true;
        row[s-1] = true;
        sq.push(row);
    }
    sq.push(vec![true; s]);
    sq
}

/// Write given 2D numerical matrix to a grayscale image at requested path.
pub fn mat_to_img<T: Clone+Ord+Num+ToPrimitive, P: AsRef<Path>>
                 (t: &[T], dim: (usize, usize), p: P, clip: Option<(T, T)>) {
    // Normalize to range 0, 255.
    let (m, n) = dim;
    let (min, max) = {
        let min = t.iter().min().expect("Could not find minimum value.");
        let max = t.iter().max().expect("Could not find maximum value.");
        if let Some((lo, hi)) = clip {
            (std::cmp::max(min, &lo).to_f64().unwrap(),
             std::cmp::min(max, &hi).to_f64().unwrap())
        } else {
            (min.to_f64().unwrap(), max.to_f64().unwrap())
        }
    };
    // make sure range >= 1 to avoid divide by zero later.
    let range = if max > min { max - min } else { 1.0 };
    let bytes = t.iter().map(|v| v.to_f64().expect("Cast to f64 failed."))
                        .map(|v| if v < min { min } else if v > max { max } else { v })
                        .map(|v| (255.0 * (v + min) / range) as u8)
                        .collect::<Vec<u8>>();
    imagefmt::write(p, n, m, ColFmt::Y, &bytes, ColType::Auto).expect("Error writing image file");
}

/// Match provided mask matrix on provided distance transform matrix.
/// dt and mask matrices are read from in parallel.
/// Return score of match at each square.
pub fn match_shape<BMatrix: AsRef<[BRow]>+Sync, BRow: AsRef<[bool]>+Sync>
                   (dt: &[i32], dim: (usize, usize), mask: BMatrix) -> Vec<i32>
{
    let (m, n) = dim;
    let s = mask.as_ref().len();
    let t = mask.as_ref()[0].as_ref().len();
    let cm = vec![std::i32::MAX; n*m];
    let row_slice: Vec<usize> = (0..m-s).collect();
    row_slice.par_iter().for_each(|&i| {
        // Get pointer to start of the i-th row.
        unsafe {
            let ptr = cm.as_ptr().offset((i * n) as isize) as *mut i32;
            for j in 0..n-t {
                *(ptr).offset(j as isize) = mask.as_ref().iter().enumerate()
                    .flat_map(|(x, row)| row.as_ref().iter().enumerate()
                                .map(move |(y, v)| (x, y, v)))
                    .filter(|&(_, _, v)| *v).map(|(x, y, _)| dt[(i+x)*n + j+y]).sum();
            }
        }
    });
    cm
}

pub fn min_idx<T: Ord+Num>(t: &[T]) -> Option<(usize, &T)> {
    t.iter().enumerate().min_by_key(|&(_, v)| v)
}

fn main() {
    let mut s = Stopwatch::start_new();
    let nodes = osm_parse::osm_to_nodes(&std::env::args().nth(1).unwrap());
    println!("XML parse took {} ms, found {} nodes", s.elapsed_ms(), nodes.len());
    s.restart();
    let grid = geogrid::new(&nodes, (15.0, 15.0));
    println!("{:?} pixel grid construction took {} ms", grid.size(), s.elapsed_ms());
    mat_to_img(grid.grid(), grid.size(), "grid_mat.png", None);
    s.restart();
    let dt = grid.dist_transform();
    println!("Distance transform took {} ms", s.elapsed_ms());
    mat_to_img(&dt, grid.size(), "dt_mat.png", Some((0, 100)));
    println!("Max distance: {}", dt.iter().max().unwrap());
    s.restart();
    let mask = build_square(15);
    let cm = match_shape(&dt, grid.size(), &mask);
    println!("Shape match took {} ms", s.elapsed_ms());
    mat_to_img(&cm, grid.size(), "cm_mat.png", Some((0, 200)));
    if let Some((idx, v)) = min_idx(&cm) {
        println!("Found match {} at {}", v, idx);
        for (x, y) in grid.trace_shape(&mask, idx).iter().map(|&i| grid.to_lat_lon(i)) {
            print!("({}, {}) ", x, y);
        }
    }
    println!();
}

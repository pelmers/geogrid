use std::f32;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::str::from_utf8;

extern crate quick_xml;
extern crate stopwatch;
extern crate imagefmt;
extern crate rayon;

use quick_xml::reader::Reader;
use rayon::prelude::*;
use quick_xml::events::Event;
use stopwatch::Stopwatch;
use imagefmt::ColFmt;
use imagefmt::ColType;

#[derive(Debug)]
pub struct Node {
    lat: f32,
    lon: f32
}

/// Compute the length in meters of one degree latitude and longitude at given latitude degree.
pub fn lat_lon(lat: f32) -> (f32, f32) {
    // Port of http://msi.nga.mil/MSISiteContent/StaticFiles/Calculators/degree.html
    let lat = lat * std::f32::consts::PI * 2.0 / 360.0;
    let m1 = 111132.92;
    let m2 = -559.82;
    let m3 = 1.175;
    let m4 = -0.0023;
    let p1 = 111412.84;
    let p2 = -93.5;
    let p3 = 0.118;

    // Calculate the length of a degree of latitude and longitude in meters
    let latlen = m1 + (m2 * (2.0 * lat).cos()) + (m3 * (4.0 * lat).cos()) + (m4 * (6.0 * lat).cos());
    let longlen = (p1 * lat.cos()) + (p2 * (3.0 * lat).cos()) + (p3 * (5.0 * lat).cos());
    (latlen, longlen)
}

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

/// Write given i32 matrix to an image at requested path.
/// If scale, then interpolate all display values linearly to the range [0, 255].
/// Otherwise just truncate excess values to range [0, 255].
pub fn bmat_to_img<P: AsRef<Path>, Matrix: AsRef<[Row]>, Row: AsRef<[bool]>>(t: Matrix, p: P) {
    // Normalize to range 0, 255.
    let t = t.as_ref();
    let m = t.len();
    let n = t[0].as_ref().len();
    // make sure range >= 1 to avoid divide by zero later.
    let mut bytes = vec![0u8; m*n];
    for i in 0..m {
        for j in 0..n {
            bytes[i*n + j] = if t[i].as_ref()[j] { 255 } else { 0 };
        }
    }
    imagefmt::write(p, n, m, ColFmt::Y, &bytes, ColType::Auto).expect("Error writing image file");
}


/// Write given i32 matrix to an image at requested path.
/// If scale, then interpolate all display values linearly to the range [0, 255].
/// Otherwise just truncate excess values to range [0, 255].
pub fn mat_to_img<P: AsRef<Path>, Matrix: AsRef<[Row]>, Row: AsRef<[i32]>>(t: Matrix, p: P, scale: bool) {
    // Normalize to range 0, 255.
    let t = t.as_ref();
    let m = t.len();
    let n = t[0].as_ref().len();
    let min = t.iter().map(|r| r.as_ref().iter().min().unwrap()).min().unwrap();
    let max = t.iter().map(|r| r.as_ref().iter().max().unwrap()).max().unwrap();
    // make sure range >= 1 to avoid divide by zero later.
    let range = std::cmp::min(1, max - min) as f64;
    let mut bytes = vec![0u8; m*n];
    for i in 0..m {
        for j in 0..n {
            bytes[i*n + j] = if scale {
                (((t[i].as_ref()[j] + min) as f64) / range * 255.0) as u8
            } else {
                std::cmp::min(std::cmp::max(t[i].as_ref()[j], 0), 255) as u8
            };
        }
    }
    imagefmt::write(p, n, m, ColFmt::Y, &bytes, ColType::Auto).expect("Error writing image file");
}

/// Compute L1 distance transform of t matrix.
pub fn dist_transform<Matrix: AsRef<[Row]>, Row: AsRef<[bool]>>(t: Matrix) -> Vec<Vec<i32>> {
    let t = t.as_ref();
    let m = t.len();
    let n = t[0].as_ref().len();
    let mut dt = vec![vec![(m*n+1) as i32; n]; m];
    for i in 0..m {
        for j in 0..n {
            if t[i].as_ref()[j] {
                dt[i][j] = 0;
            } else {
                // Let val be min of current, (left, and above) + 1
                let mut val = dt[i][j];
                if j > 0 {
                    val = std::cmp::min(val, dt[i][j-1] + 1);
                }
                if i > 0 {
                    val = std::cmp::min(val, dt[i-1][j] + 1);
                }
                dt[i][j] = val;
            }
        }
    }
    // Second pass, reverse order
    for i in (0..m).rev() {
        for j in (0..n).rev() {
            // take min of current, (right, and below) + 1
            let mut val = dt[i][j];
            if j < n - 1 {
                val = std::cmp::min(val, dt[i][j+1] + 1);
            }
            if i < m - 1 {
                val = std::cmp::min(val, dt[i+1][j] + 1);
            }
            dt[i][j] = val;
        }
    }
    dt
}

/// Match provided mask matrix on provided distance transform matrix.
/// dt and mask matrices are read from in parallel.
/// Return score of match at each square.
pub fn match_shape<IMatrix: AsRef<[IRow]>+Sync, IRow: AsRef<[i32]>+Sync,
                   BMatrix: AsRef<[BRow]>+Sync, BRow: AsRef<[bool]>+Sync>
                   (dt: IMatrix, mask: BMatrix) -> Vec<Vec<i32>>
{
    let dt = dt.as_ref();
    let m = dt.len();
    let n = dt[0].as_ref().len();
    let s = mask.as_ref().len();
    let t = mask.as_ref()[0].as_ref().len();
    let cm = vec![vec![std::i32::MAX; n]; m];
    let row_slice: Vec<usize> = (0..m-s).collect();
    row_slice.par_iter().for_each(|&i| {
        let ptr = cm[i].as_ptr() as *mut i32;
        for j in 0..n-t {
            unsafe {
                *(ptr).offset(j as isize) = mask.as_ref().iter().enumerate()
                    .flat_map(|(x, row)| row.as_ref().iter().enumerate()
                              .map(move |(y, v)| (x, y, v)))
                    .filter(|&(_, _, v)| *v).map(|(x, y, _)| dt[i+x].as_ref()[j+y]).sum();
            }
        }
    });
    cm
}

pub fn trace_shape<BMatrix: AsRef<[BRow]>+Sync, BRow: AsRef<[bool]>+Sync>(shape: BMatrix, topleft: (usize, usize)) -> Vec<(usize, usize)> {
    let (a, b) = topleft;
    shape.as_ref().iter().enumerate().flat_map(|(x, r)| r.as_ref().iter().enumerate()
                                                         .map(move |(y, v)| (x, y, v)))
         .filter(|&(_, _, v)| *v).map(|(x, y, _)| (x + a, y + b)).collect()
}

/// Return i, j index pair in integer matrix t with lowest value.
pub fn min_idx<Matrix: AsRef<[Row]>, Row: AsRef<[i32]>>(t: Matrix) -> (usize, usize) {
    t.as_ref().iter().enumerate().flat_map(|(x, row)| row.as_ref().iter().enumerate()
                                                         .map(move |(y, v)| (x, y, v)))
     .min_by_key(|&(_, _, v)| *v).map(|(x, y, _)| (x, y)).unwrap()
}

fn main() {
    let mut r = Reader::from_file(&std::env::args().nth(1).unwrap()).unwrap();
    let mut nodes = HashMap::new();
    let mut ways = HashMap::new();
    let mut nd_vec = Vec::new();
    let mut current_way = -1i64;
    let mut buf = Vec::with_capacity(2048);
    let mut s = Stopwatch::start_new();
    loop {
        match r.read_event(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                match e.name() {
                    b"node" => {
                        let mut o_id: Option<i64> = Option::None;
                        let mut o_lat: Option<f32> = Option::None;
                        let mut o_lon: Option<f32> = Option::None;
                        for attr in e.attributes() {
                            if let Ok(a) = attr {
                                match a.key {
                                    b"id" => o_id = FromStr::from_str(from_utf8(a.value).unwrap()).ok(),
                                    b"lat" => o_lat = FromStr::from_str(from_utf8(a.value).unwrap()).ok(),
                                    b"lon" => o_lon = FromStr::from_str(from_utf8(a.value).unwrap()).ok(),
                                    _ => ()
                                }
                            }
                        }
                        if o_id.is_some() && o_lat.is_some() && o_lon.is_some() {
                            nodes.insert(o_id.unwrap(), Node{lat: o_lat.unwrap(), lon: o_lon.unwrap()});
                        }
                    },
                    b"way" => {
                        for attr in e.attributes() {
                            if let Ok(a) = attr {
                                match a.key {
                                    b"id" => current_way = FromStr::from_str(from_utf8(a.value).unwrap()).unwrap_or(current_way),
                                    _ => ()
                                }
                            }
                        }
                    },
                    b"nd" => {
                        for attr in e.attributes() {
                            if let Ok(a) = attr {
                                match a.key {
                                    b"ref" => {
                                        let r_ref: Option<i64> = FromStr::from_str(from_utf8(a.value).unwrap()).ok();
                                        if r_ref.is_some() {
                                            nd_vec.push(r_ref.unwrap());
                                        }
                                    },
                                    _ => ()
                                }
                            }
                        }
                    },
                    _ => ()
                }
            },
            Ok(Event::End(ref e)) => {
                match e.name() {
                    b"way" => {
                        ways.insert(current_way, nd_vec.clone());
                        nd_vec.clear();
                    },
                    _ =>()
                }
            },
            Ok(Event::Eof) => break,
            _ => ()
        };
        buf.clear();
    }
    println!("XML parse took {} ms", s.elapsed_ms());
    // Now compute grid of width x height to quantize the space.
    let lat_min = nodes.values().map(|n| n.lat).fold(f32::MAX, f32::min);
    let lat_max = nodes.values().map(|n| n.lat).fold(f32::MIN, f32::max);
    let lon_min = nodes.values().map(|n| n.lon).fold(f32::MAX, f32::min);
    let lon_max = nodes.values().map(|n| n.lon).fold(f32::MIN, f32::max);
    // Desired size of lat/lon divisions, in meters.
    let lat_div_size = 20.0;
    let lon_div_size = 20.0;
    let (lat_len, lon_len) = lat_lon((lat_min + lat_max) / 2.0);
    let height = lat_len * (lat_max - lat_min);
    let width = lon_len * (lon_max - lon_min);
    println!("Processed {} nodes", nodes.len());
    println!("Processed {} ways", ways.len());
    println!("Found given bounds: [{}, {}] x [{}, {}]", lat_min, lat_max, lon_min, lon_max);
    println!("{} x {} km", height / 1000.0, width / 1000.0);
    s.restart();
    let mut grid = vec![vec![false; (height / lat_div_size).round() as usize];
                        (width / lon_div_size).round() as usize];
    let mut count = 0;
    for (_, nds) in ways {
        for ref nd in nds {
            let grid_lat = (nodes[nd].lat - lat_min) / (lat_max - lat_min) * ((grid.len() - 1) as f32);
            let grid_lon = (nodes[nd].lon - lon_min) / (lon_max - lon_min) * ((grid[0].len() - 1)  as f32);
            let xx = grid.len() - 1;
            grid[xx - (grid_lat.round() as usize)][grid_lon.round() as usize] = true;
            count += 1;
        }
    }
    println!("Grid construction took {} ms", s.elapsed_ms());
    println!("Grid size: {} x {} = {}", grid.len(), grid[0].len(), grid.len() * grid[0].len());
    println!("Grid density: {}%", 100.0*(count as f32) / (grid.len() * grid[0].len()) as f32);
    bmat_to_img(&grid, "grid_mat.png");
    s.restart();
    let dt = dist_transform(&grid);
    println!("Distance transform took {} ms", s.elapsed_ms());
    mat_to_img(&dt, "dt_mat.png", true);
    println!("Max distance: {}", dt.iter().map(|r| r.iter().max().unwrap()).max().unwrap());
    s.restart();
    let mask = build_square(63);
    let cm = match_shape(&dt, &mask);
    println!("Shape match took {} ms", s.elapsed_ms());
    mat_to_img(&cm, "cm_mat.png", false);
    let (xp, yp) = min_idx(&cm);
    println!("Found match {} at ({}, {})", cm[xp][yp], xp, yp);
    // The number of degrees each lat/lon on the grid spans in the real map.
    let lat_per_grid = lat_div_size / lat_len;
    let lon_per_grid = lon_div_size / lon_len;
    for (x, y) in trace_shape(&mask, (xp, yp)) {
        print!("({}, {}), ", lat_max - lat_per_grid * (x as f32), lon_per_grid * (y as f32) + lon_min);
    }
    println!("");
}

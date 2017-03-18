use std::{f32, i32, cmp};
use std::io::Read;
use std::path::Path;

use geojson::GeoJson;
use num::{Num, ToPrimitive};
use rayon::prelude::*;


use imagefmt::{ColFmt, ColType};
use types::{Node, Bounds};


/// Compute the length in meters of one degree latitude and longitude at given latitude degree.
pub fn lat_lon(lat: f32) -> (f32, f32) {
    // Port of http://msi.nga.mil/MSISiteContent/StaticFiles/Calculators/degree.html
    let lat = lat * f32::consts::PI * 2.0 / 360.0;
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


/// Find the bounds over an iterator of nodes.
pub fn node_bounds<'a, I: Iterator<Item=&'a Node>>(iter: I) -> Bounds {
    iter.fold(Bounds{north: f32::MIN, south: f32::MAX, east: f32::MIN, west: f32::MAX},
              |b, n|  Bounds{north: f32::max(b.north, n.lat), south: f32::min(b.south, n.lat),
                             east: f32::max(b.east, n.lon), west: f32::min(b.west, n.lon)})
}


/// Return vector of vector of nodes where each vector represents the nodes defined on that road.
/// If bounds given, only return coordinates within bounds.
pub fn roads_from_json<R: Read>(reader: R, b: Option<Bounds>) -> Vec<Vec<Node>> {
    let mut nodes = Vec::with_capacity(2000);
    let b = match b {
        Some(b) => b,
        None => Bounds{north: f32::MAX, south: f32::MIN, east: f32::MAX, west: f32::MIN}
    };
    if let Ok(json) = ::serde_json::from_reader::<_, GeoJson>(reader) {
        if let GeoJson::FeatureCollection(ref roads) = json {
            for road in &roads.features {
                if let &Some(ref geometry) = &road.geometry {
                    if let ::geojson::Value::LineString(ref positions) = geometry.value {
                        nodes.push(positions.iter()
                                   .map(|pos| Node{lat: pos[1] as f32, lon: pos[0] as f32})
                                   .filter(|n| (b.north > n.lat) && (b.south < n.lat) && (b.east > n.lon) && (b.west < n.lon))
                                   .collect())
                    }
                }
            }
        }
    }
    nodes
}


/// Write given 2D numerical matrix to a scaled grayscale image at requested path.
/// Clip specifies lower, upper bounds of values that will be clipped to black/white.
pub fn mat_to_img<T: Clone+Ord+Num+ToPrimitive, P: AsRef<Path>>
                 (t: &[T], dim: (usize, usize), p: P, clip: Option<(T, T)>) {
    // Normalize to range 0, 255.
    let (m, n) = dim;
    let (min, max) = {
        let min = t.iter().min().expect("Could not find minimum value.");
        let max = t.iter().max().expect("Could not find maximum value.");
        if let Some((lo, hi)) = clip {
            (cmp::max(min, &lo).to_f64().unwrap(),
             cmp::min(max, &hi).to_f64().unwrap())
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
    ::imagefmt::write(p, n, m, ColFmt::Y, &bytes, ColType::Auto).expect("Error writing image file");
}


fn stride_mask<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>
               (mask: BMatrix) -> Vec<Vec<i32>>
{
    let mask = mask.as_ref();
    let m = mask.len();
    if m == 0 {
        return vec![vec![]];
    }
    let n = mask[0].as_ref().len();
    let mut stride = vec![vec![0; n]; m];
    for (r, row) in mask.iter().enumerate() {
        let mut last_set = row.as_ref().len();
        for (c, val) in row.as_ref().iter().enumerate().rev() {
            if *val {
                last_set = c;
            } else {
                stride[r][c] = (last_set - c) as i32;
            }
        }
    }
    stride
}


/// Set to large value any cell that is not the minimum of its 8-neighborhood.
fn non_min_suppress(mat: &mut [i32], dim: (usize, usize)) {
    // At each pixel, suppress any neighbor pixels greater than it.
    let (m, n) = dim;
    for i in 0..m {
        for j in 0..n {
            for x in cmp::max(0, i-1)..cmp::min(m, i+2) {
                for y in cmp::max(0, j-1)..cmp::min(n, j+2) {
                    if mat[i * n + j] < mat[x * n + y] {
                        mat[x * n + y] = i32::MAX;
                    }
                }
            }
        }
    }
}


/// Match provided mask matrix on provided distance transform matrix.
/// dt and mask matrices are read from in parallel.
/// Return score of match at each square.
pub fn match_shape<BMatrix: AsRef<[BRow]>+Sync, BRow: AsRef<[bool]>+Sync>
                   (dt: &[i32], dim: (usize, usize), mask: BMatrix) -> Vec<i32>
{
    let (m, n) = dim;
    let sm = stride_mask(mask);
    let s = sm.len();
    let t = sm[0].len();
    let mut cm = vec![i32::MAX; n*m];
    let row_vec: Vec<usize> = (0..m-s).collect();
    row_vec.par_iter().for_each(|&i| {
        // Get pointer to start of the i-th row.
        unsafe {
            let ptr = cm.as_ptr().offset((i * n) as isize) as *mut i32;
            for j in 0..n-t {
                let mut acc = 0;
                for x in 0..s {
                    let ref s_row = sm[x];
                    let mut y = 0;
                    while y < t {
                        let val = *s_row.get_unchecked(y);
                        if val == 0 {
                            acc += dt[(i + x)*n + j+y];
                            y += 1;
                        } else {
                            y += val as usize;
                        }
                    }
                }
                if acc < 0 {
                    // Then I suppose it overflowed, flip it back around.
                    acc = i32::MAX;
                }
                *(ptr).offset(j as isize) = acc;
            }
        }
    });
    non_min_suppress(&mut cm, dim);
    cm
}


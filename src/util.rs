use std::{f32, cmp, i32};
use std::io::Read;
use std::path::Path;

use geojson::GeoJson;
use num::{Num, ToPrimitive};

use imagefmt::{ColFmt, ColType};
use types::{Node, Bounds};

pub use match_shape::{match_shape, Processor};


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
    let latlen = m1 + (m2 * (2.0 * lat).cos()) + (m3 * (4.0 * lat).cos()) +
                 (m4 * (6.0 * lat).cos());
    let longlen = (p1 * lat.cos()) + (p2 * (3.0 * lat).cos()) + (p3 * (5.0 * lat).cos());
    (latlen, longlen)
}


/// Find the bounds over an iterator of nodes.
pub fn node_bounds<'a, I: Iterator<Item = &'a Node>>(iter: I) -> Bounds {
    iter.fold(Bounds {
                  north: f32::MIN,
                  south: f32::MAX,
                  east: f32::MIN,
                  west: f32::MAX,
              },
              |b, n| {
        Bounds {
            north: f32::max(b.north, n.lat),
            south: f32::min(b.south, n.lat),
            east: f32::max(b.east, n.lon),
            west: f32::min(b.west, n.lon),
        }
    })
}


/// Return vector of vector of nodes where each vector represents the nodes defined on that road.
/// If bounds given, only return coordinates within bounds.
pub fn roads_from_json<R: Read>(reader: R, b: Option<Bounds>) -> Vec<Vec<Node>> {
    let mut nodes = Vec::with_capacity(2000);
    let b = match b {
        Some(b) => b,
        None => {
            Bounds {
                north: f32::MAX,
                south: f32::MIN,
                east: f32::MAX,
                west: f32::MIN,
            }
        }
    };
    if let Ok(json) = ::serde_json::from_reader::<_, GeoJson>(reader) {
        if let GeoJson::FeatureCollection(ref roads) = json {
            for road in &roads.features {
                if let Some(ref geometry) = road.geometry {
                    if let ::geojson::Value::LineString(ref positions) = geometry.value {
                        nodes.push(positions.iter()
                            .map(|pos| {
                                Node {
                                    lat: pos[1] as f32,
                                    lon: pos[0] as f32,
                                }
                            })
                            .filter(|n| {
                                (b.north > n.lat) && (b.south < n.lat) && (b.east > n.lon) &&
                                (b.west < n.lon)
                            })
                            .collect());
                    }
                }
            }
        }
    }
    nodes
}


/// Write given 2D numerical matrix to a scaled grayscale image at requested path.
/// Clip specifies lower, upper bounds of values that will be clipped to black/white.
pub fn mat_to_img<T: Clone + Ord + Num + ToPrimitive, P: AsRef<Path>>(t: &[T],
                                                                      dim: (usize, usize),
                                                                      p: P,
                                                                      clip: Option<(T, T)>) {
    // Normalize to range 0, 255.
    let (m, n) = dim;
    let (min, max) = {
        let min = t.iter().min().expect("Could not find minimum value.");
        let max = t.iter().max().expect("Could not find maximum value.");
        if let Some((lo, hi)) = clip {
            (cmp::max(min, &lo).to_f64().unwrap(), cmp::min(max, &hi).to_f64().unwrap())
        } else {
            (min.to_f64().unwrap(), max.to_f64().unwrap())
        }
    };
    // make sure range >= 1 to avoid divide by zero later.
    let range = if max > min { max - min } else { 1.0 };
    let bytes = t.iter()
        .map(|v| v.to_f64().expect("Cast to f64 failed."))
        .map(|v| if v < min {
            min
        } else if v > max {
            max
        } else {
            v
        })
        .map(|v| (255.0 * (v + min) / range) as u8)
        .collect::<Vec<u8>>();
    ::imagefmt::write(p, n, m, ColFmt::Y, &bytes, ColType::Auto).expect("Error writing image file");
}

/// Compute L1 distance transform of t matrix with given dimensions.
/// Output is linearized matrix of same size as grid.
pub fn l1dist_transform(t: &[u8], dim: (usize, usize)) -> Vec<i32> {
    let (m, n) = dim;
    let mut dt = vec![(m*n+1) as i32; n*m];
    for i in 0..m {
        let off = i * n;
        for j in 0..n {
            if t[off + j] > 0 {
                dt[off + j] = 0;
            } else {
                // Let val be min of current, (left, and above) + 1
                let mut val = dt[off + j];
                if j > 0 {
                    val = cmp::min(val, dt[off + j - 1] + 1);
                }
                if i > 0 {
                    val = cmp::min(val, dt[off - n + j] + 1);
                }
                dt[off + j] = val;
            }
        }
    }
    // Second pass, reverse order
    for i in (0..m).rev() {
        let off = i * n;
        for j in (0..n).rev() {
            // take min of current, (right, and below) + 1
            let mut val = dt[off + j];
            if j < n - 1 {
                val = cmp::min(val, dt[off + j + 1] + 1);
            }
            if i < m - 1 {
                val = cmp::min(val, dt[off + j + n] + 1);
            }
            dt[off + j] = val;
        }
    }
    dt
}

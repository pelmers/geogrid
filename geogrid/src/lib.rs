extern crate num;
extern crate imagefmt;

use std::f32;
use std::path::Path;

use imagefmt::ColFmt;
use imagefmt::ColType;
use num::{Num, ToPrimitive};

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

#[derive(Debug, Copy, Clone)]
pub struct Node {
    pub lat: f32,
    pub lon: f32
}

pub struct GeoGrid {
    min_lat: f32,
    max_lat: f32,
    min_lon: f32,
    max_lon: f32,
    res_lat: f32,
    res_lon: f32,
    grid_height: usize,
    grid_width: usize,
    grid: Vec<u8>
}

// Recall: max latitude goes to row 0.
pub fn new(nodes: &[Node], resolution: (f32, f32)) -> GeoGrid {
    let (res_lat, res_lon) = resolution;
    let min_lat = nodes.iter().map(|n| n.lat).fold(f32::MAX, f32::min);
    let max_lat = nodes.iter().map(|n| n.lat).fold(f32::MIN, f32::max);
    let min_lon = nodes.iter().map(|n| n.lon).fold(f32::MAX, f32::min);
    let max_lon = nodes.iter().map(|n| n.lon).fold(f32::MIN, f32::max);
    let range_lat = max_lat - min_lat;
    let range_lon = max_lon - min_lon;

    let (lat_len, lon_len) = lat_lon((min_lat + max_lat) / 2.0);
    let real_height = lat_len * range_lat;
    let real_width = lon_len * range_lon;
    let grid_height = (real_height / res_lat).round() as usize;
    let grid_width = (real_width / res_lon).round() as usize;
    let mut grid = vec![0; grid_height*grid_width];
    for n in nodes {
        let grid_lat = (max_lat - n.lat) / range_lat * ((grid_height - 1) as f32);
        let grid_lon = (n.lon - min_lon) / range_lon * ((grid_width - 1)  as f32);
        grid[(grid_lat.round() as usize) * grid_width + (grid_lon.round() as usize)] = 1;
    }

    GeoGrid{ min_lat, max_lat, min_lon, max_lon, res_lat, res_lon, grid_height, grid_width, grid }
}

// TODO: impl indexing trait

impl GeoGrid {
    /// Grid resolution, in meters per row and meters per column.
    pub fn resolution(&self) -> (f32, f32) {
        (self.res_lat, self.res_lon)
    }

    /// Return a pair of latitude/longitude pairs that represent the northwest and southeast
    /// corners of the bounding box covered by the grid.
    pub fn bbox(&self) -> ((f32, f32), (f32, f32)) {
        ((self.max_lat, self.min_lon), (self.min_lat, self.max_lon))
    }

    /// Grid dimensions.
    pub fn size(&self) -> (usize, usize) {
        (self.grid_height, self.grid_width)
    }

    /// Grid length.
    pub fn len(&self) -> usize {
        self.grid.len()
    }

    /// Latitude and longitude covered, in meters. resolution * size.
    pub fn real_size(&self) -> (f32, f32) {
        (self.res_lat * self.grid_height as f32, self.res_lon * self.grid_width as f32)
    }

    /// Access the underlying grid.
    pub fn grid(&self) -> &[u8] {
        &self.grid[..]
    }

    /// Trace provided shape from given top left index, returning locations of all the true cells.
    pub fn trace_shape<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(&self, shape: BMatrix, topleft: usize) -> Vec<usize> {
        shape.as_ref().iter().enumerate().flat_map(|(x, r)| r.as_ref().iter().enumerate()
                                                   .map(move |(y, v)| (x, y, v)))
            .filter(|&(_, _, v)| *v).map(|(x, y, _)| topleft + x * self.grid_width + y).collect()
    }

    /// Return lat, lon coordinates of given index in the grid.
    pub fn to_lat_lon(&self, idx: usize) -> (f32, f32) {
        let (lat_len, lon_len) = lat_lon((self.min_lat + self.max_lat) / 2.0);
        let row = (idx / self.grid_width) as f32;
        let col = (idx % self.grid_width) as f32;
        (self.max_lat - row * self.res_lat / lat_len, self.min_lon + col * self.res_lon / lon_len)
    }

    /// Compute L1 distance transform of t matrix.
    /// Output is linearized matrix of same size as grid.
    pub fn dist_transform(&self) -> Vec<i32> {
        let m = self.grid_height;
        let n = self.grid_width;
        let mut dt = vec![(m*n+1) as i32; n*m];
        for i in 0..m {
            let off = i * n;
            for j in 0..n {
                if self.grid[off+j] > 0 {
                    dt[off+j] = 0;
                } else {
                    // Let val be min of current, (left, and above) + 1
                    let mut val = dt[off + j];
                    if j > 0 {
                        val = std::cmp::min(val, dt[off + j - 1] + 1);
                    }
                    if i > 0 {
                        val = std::cmp::min(val, dt[off - n + j] + 1);
                    }
                    dt[off+j] = val;
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
                    val = std::cmp::min(val, dt[off + j + 1] + 1);
                }
                if i < m - 1 {
                    val = std::cmp::min(val, dt[off + j + n] + 1);
                }
                dt[off + j] = val;
            }
        }
        dt
    }
}


extern crate geojson;
extern crate num;
extern crate imagefmt;
extern crate rayon;
extern crate serde_json;
#[macro_use] extern crate serde_derive;

use std::f32;
use std::io::Read;
use std::path::Path;

use geojson::GeoJson;
use imagefmt::ColFmt;
use imagefmt::ColType;
use num::{Num, ToPrimitive};
use rayon::prelude::*;


#[derive(Debug, Copy, Clone)]
pub struct Node {
    pub lat: f32,
    pub lon: f32
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct Bounds {
    pub north: f32,
    pub south: f32,
    pub east: f32,
    pub west: f32
}

#[derive(Clone)]
pub struct GeoGrid {
    bounds: Bounds,
    res_lat: f32,
    res_lon: f32,
    grid_height: usize,
    grid_width: usize,
    grid: Vec<u8>
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


/// Write given 2D numerical matrix to a scaled grayscale image at requested path.
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


/// Set near max int any cell that is not the minimum of its 8-neighborhood.
fn min_suppress(mat: &mut [i32], dim: (usize, usize)) {
    //TODO...
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
    let mut cm = vec![std::i32::MAX; n*m];
    let row_slice: Vec<usize> = (0..m-s).collect();
    row_slice.par_iter().for_each(|&i| {
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
                *(ptr).offset(j as isize) = acc;
                //*(ptr).offset(j as isize) = mask.as_ref().iter().enumerate()
                //    .flat_map(|(x, row)| row.as_ref().iter().enumerate()
                //                .map(move |(y, v)| (x, y, v)))
                //    .filter(|&(_, _, v)| *v).map(|(x, y, _)| dt[(i+x)*n + j+y]).sum();
            }
        }
    });
    // TODO: non-max suppression
    min_suppress(&mut cm, dim);
    cm
}


/// Return vector of vector of nodes where each vector represents the nodes defined on that road.
/// If bounds given, only return coordinates within bounds.
pub fn nodes_from_json<R: Read>(reader: R, b: Option<Bounds>) -> Vec<Vec<Node>> {
    let mut nodes = Vec::with_capacity(2000);
    let b = match b {
        Some(b) => b,
        None => Bounds{north: f32::MAX, south: f32::MIN, east: f32::MAX, west: f32::MIN}
    };
    if let Ok(json) = serde_json::from_reader::<_, GeoJson>(reader) {
        if let GeoJson::FeatureCollection(ref roads) = json {
            for road in &roads.features {
                if let &Some(ref geometry) = &road.geometry {
                    if let geojson::Value::LineString(ref positions) = geometry.value {
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

/// Find the bounds over an iterator of nodes.
fn node_bounds<'a, I: Iterator<Item=&'a Node>>(iter: I) -> Bounds {
    iter.fold(Bounds{north: f32::MIN, south: f32::MAX, east: f32::MIN, west: f32::MAX},
              |b, n|  Bounds{north: f32::max(b.north, n.lat), south: f32::min(b.south, n.lat),
                             east: f32::max(b.east, n.lon), west: f32::min(b.west, n.lon)})
}

impl Bounds {
    pub fn range_lat(&self) -> f32 {
        self.north - self.south
    }
    pub fn range_lon(&self) -> f32 {
        self.east - self.west
    }
}


#[inline]
unsafe fn unsafe_mark_point(buf: &[u8], row: f32, col: f32, width: usize) {
    let buf_offset = row.round() as usize * width + col.round() as usize;
    *(buf.as_ptr().offset(buf_offset as isize) as *mut u8) = 1;
}

// TODO: impl indexing trait
impl GeoGrid {
    // Similar to from_nodes but interpolate linearly between nodes in a row.
    pub fn from_roads<NMatrix: AsRef<[NRow]>+Sync, NRow: AsRef<[Node]>+Sync>(roads: NMatrix, resolution: (f32, f32)) -> GeoGrid {
        let roads = roads.as_ref();

        // TODO: factor out with from_nodes
        let (res_lat, res_lon) = resolution;
        let bounds = node_bounds(roads.iter().flat_map(|r| r.as_ref().iter()));

        let (lat_len, lon_len) = lat_lon((bounds.south + bounds.north) / 2.0);
        let real_height = lat_len * bounds.range_lat();
        let real_width = lon_len * bounds.range_lon();
        let grid_height = (real_height / res_lat).round() as usize;
        let grid_width = (real_width / res_lon).round() as usize;
        let grid = vec![0; grid_height*grid_width];
        roads.par_iter().for_each(|r| {
            let r = r.as_ref();
            for (a, b) in r.iter().zip(r.iter().skip(1)) {
                let a_row = (bounds.north - a.lat) / bounds.range_lat() * ((grid_height - 1) as f32);
                let a_col = (a.lon - bounds.west) / bounds.range_lon() * ((grid_width - 1)  as f32);
                let b_row = (bounds.north - b.lat) / bounds.range_lat() * ((grid_height - 1) as f32);
                let b_col = (b.lon - bounds.west) / bounds.range_lon() * ((grid_width - 1)  as f32);
                let max_abs_diff = f32::max((b_row - a_row).abs(), (b_col - a_col).abs());
                if max_abs_diff == 0.0 {
                    // Sometimes the data puts two nodes at the same point.
                    // In this case only mark the first one and continue.
                    unsafe {
                        unsafe_mark_point(&grid, a_row, a_col, grid_width);
                    }
                    continue;
                }
                let lat_step = (b_row - a_row) / max_abs_diff;
                let lon_step = (b_col - a_col) / max_abs_diff;
                // Step point a toward point b and write grid cell.
                // By construction each step will fill exactly one cell and we will not skip any.
                let mut step_row = a_row;
                let mut step_col = a_col;
                while (step_row > b_row) == (a_row > b_row) &&
                      (step_col > b_col) == (a_col > b_col) {
                    // Because we set to 1 no matter what, data race on this cell will not affect
                    // the overall outcome.
                    unsafe {
                        unsafe_mark_point(&grid, step_row, step_col, grid_width);
                    }
                    step_row += lat_step;
                    step_col += lon_step;
                }
            }
            // Now mark the final point in the road, which is skipped in the zip iteration above.
            if r.len() > 0 {
                let last = r[r.len() - 1];
                let l_row = (bounds.north - last.lat) / bounds.range_lat() * ((grid_height - 1) as f32);
                let l_col = (last.lon - bounds.west) / bounds.range_lon() * ((grid_width - 1)  as f32);
                unsafe {
                    unsafe_mark_point(&grid, l_row, l_col, grid_width);
                }
            }
        });
        GeoGrid{ bounds, res_lat, res_lon, grid_height, grid_width, grid }
    }
    // Recall: max latitude goes to row 0.
    pub fn from_nodes(nodes: &[Node], resolution: (f32, f32)) -> GeoGrid {
        let (res_lat, res_lon) = resolution;
        let bounds = node_bounds(nodes.iter());

        let (lat_len, lon_len) = lat_lon((bounds.south + bounds.north) / 2.0);
        let real_height = lat_len * bounds.range_lat();
        let real_width = lon_len * bounds.range_lon();
        let grid_height = (real_height / res_lat).round() as usize;
        let grid_width = (real_width / res_lon).round() as usize;
        let mut grid = vec![0; grid_height*grid_width];
        for n in nodes {
            let grid_lat = (bounds.north - n.lat) / bounds.range_lat() * ((grid_height - 1) as f32);
            let grid_lon = (n.lon - bounds.west) / bounds.range_lon() * ((grid_width - 1)  as f32);
            grid[(grid_lat.round() as usize) * grid_width + (grid_lon.round() as usize)] = 1;
        }

        GeoGrid{ bounds, res_lat, res_lon, grid_height, grid_width, grid }
    }

    /// Grid resolution, in meters per row and meters per column.
    pub fn resolution(&self) -> (f32, f32) {
        (self.res_lat, self.res_lon)
    }

    /// Grid resolution, in degrees latitude per row and degrees longitude per column.
    pub fn degree_resolution(&self) -> (f32, f32) {
        ((self.bounds.north - self.bounds.south) / self.grid_height as f32,
         (self.bounds.east - self.bounds.west) / self.grid_width as f32)
    }

    /// Return a pair of latitude/longitude pairs that represent the northwest and southeast
    /// corners of the bounding box covered by the grid.
    pub fn bbox(&self) -> Bounds {
        self.bounds
    }

    /// Grid dimensions.
    pub fn size(&self) -> (usize, usize) {
        (self.grid_height, self.grid_width)
    }

    /// Grid length.
    pub fn len(&self) -> usize {
        self.grid.len()
    }

    /// Clears the grid, but leaves all other fields intact.
    pub fn clear_grid(&mut self) {
        self.grid.clear();
        self.grid.shrink_to_fit();
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
        let (lat_len, lon_len) = lat_lon((self.bounds.south + self.bounds.north) / 2.0);
        let row = (idx / self.grid_width) as f32;
        let col = (idx % self.grid_width) as f32;
        (self.bounds.north - row * self.res_lat / lat_len, self.bounds.west + col * self.res_lon / lon_len)
    }

    /// Return closest index in grid to given latitude, longitude.
    pub fn from_lat_lon(&self, lat: f32, lon: f32) -> usize {
        let (ra, ri) = self.degree_resolution();
        let lat_offset = ((self.bounds.north - lat) / ra).round() as usize;
        let lon_offset = ((lon - self.bounds.west) / ri).round() as usize;
        lat_offset * self.grid_width + lon_offset
    }

    /// Return start index and a new size of a bounded subgrid specified by given rectangle in
    /// degrees.
    pub fn bounded_subgrid(&self, north: f32, south: f32, east: f32, west: f32) -> (usize, (usize, usize)) {
        let (ra, ri) = self.degree_resolution();
        let lat_start = ((self.bounds.north - north) / ra).round() as usize;
        let lon_start = ((west - self.bounds.west) / ri).round() as usize;
        let start = lat_start * self.grid_width + lon_start;
        let lat_end = ((self.bounds.north - south) / ra).round() as usize;
        let lon_end = ((east - self.bounds.west) / ri).round() as usize;
        (start, (lat_end - lat_start, lon_end - lon_start))
    }

    /// Compute L1 distance transform of t matrix.
    /// Output is linearized matrix of same size as grid.
    pub fn dist_transform(&self) -> Vec<i32> {
        let (m, n) = self.size();
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
        // Square it just for funsies.
        for el in dt.iter_mut() {
            *el *= *el;
        }
        dt
    }
}


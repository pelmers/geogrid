extern crate geojson;
extern crate num;
extern crate rayon;
extern crate serde_json;
extern crate imagefmt;
#[macro_use]
extern crate serde_derive;

#[cfg(feature="opencl")]
extern crate ocl;

use std::{f32, i32};

mod types;
pub use types::*;
mod match_shape;
pub mod util;
use util::{lat_lon, node_bounds};


/// `GeoGrid` represents some portion of a geographic map as a grid. The grid computes step size
/// from the center of the bounds. Because earth is not flat there will be some distance distortion
/// farther from the central latitude if the grid is spread over a large area.
#[derive(Clone)]
pub struct GeoGrid {
    bounds: Bounds,
    res_lat: f32,
    res_lon: f32,
    grid_height: usize,
    grid_width: usize,
    grid: Vec<u8>,
}

#[inline]
fn mark_point(buf: &mut [u8], row: f32, col: f32, width: usize) {
    let buf_offset = row.round() as usize * width + col.round() as usize;
    buf[buf_offset] = 1;
}

// TODO: impl indexing trait
impl GeoGrid {
    /// Construct a GeoGrid from a slice of roads, where each road is a slice of nodes. This
    /// constructor will interpolate linearly between nodes within each row to set to 1. The grid's
    /// bounds will be exactly large enough to contain every given map node. Resolution is given as
    /// pair of (height, width) of the grid in pixels. If scale is true, then the aspect ratio of
    /// the original map will be maintained, potentially reducing one of the resolution dimensions.
    /// Otherwise, with scale false, the grid will stretch to the provided resolution.
    pub fn from_roads<NMatrix: AsRef<[NRow]>+Sync, NRow: AsRef<[Node]>+Sync>
        (roads: NMatrix, resolution: (usize, usize), scale: bool) -> GeoGrid {
        let roads = roads.as_ref();

        // TODO: factor out with from_nodes
        let bounds = node_bounds(roads.iter().flat_map(|r| r.as_ref().iter()));
        let (lat_len, lon_len) = lat_lon((bounds.south + bounds.north) / 2.0);
        let real_height = lat_len * bounds.range_lat();
        let real_width = lon_len * bounds.range_lon();

        let (expected_height, expected_width) = resolution;
        let mut res_lat = real_height / expected_height as f32;
        let mut res_lon = real_width / expected_width as f32;
        if scale {
            res_lat = f32::max(res_lat, res_lon);
            res_lon = res_lat;
        }

        let grid_height = (real_height / res_lat).round() as usize;
        let grid_width = (real_width / res_lon).round() as usize;
        let mut grid = vec![0; grid_height*grid_width];
        for r in roads {
            let r = r.as_ref();
            for (a, b) in r.iter().zip(r.iter().skip(1)) {
                let a_row = (bounds.north - a.lat) / bounds.range_lat() *
                            ((grid_height - 1) as f32);
                let a_col = (a.lon - bounds.west) / bounds.range_lon() * ((grid_width - 1) as f32);
                let b_row = (bounds.north - b.lat) / bounds.range_lat() *
                            ((grid_height - 1) as f32);
                let b_col = (b.lon - bounds.west) / bounds.range_lon() * ((grid_width - 1) as f32);
                let max_abs_diff = f32::max((b_row - a_row).abs(), (b_col - a_col).abs());
                if max_abs_diff == 0.0 {
                    // Sometimes the data puts two nodes at the same point.
                    // In this case only mark the first one and continue.
                    mark_point(&mut grid, a_row, a_col, grid_width);
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
                    mark_point(&mut grid, step_row, step_col, grid_width);
                    step_row += lat_step;
                    step_col += lon_step;
                }
            }
            // Now mark the final point in the road, which is skipped in the zip iteration above.
            if !r.is_empty() {
                let last = r[r.len() - 1];
                let l_row = (bounds.north - last.lat) / bounds.range_lat() *
                            ((grid_height - 1) as f32);
                let l_col = (last.lon - bounds.west) / bounds.range_lon() *
                            ((grid_width - 1) as f32);
                mark_point(&mut grid, l_row, l_col, grid_width);
            }
        }
        GeoGrid {
            bounds: bounds,
            res_lat: res_lat,
            res_lon: res_lon,
            grid_height: grid_height,
            grid_width: grid_width,
            grid: grid,
        }
    }
    /// Construct a GeoGrid from provided slice of nodes using given resolution, expressed in units
    /// of (meters latitude per grid unit, meters longitude per grid unit).
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
            let grid_lon = (n.lon - bounds.west) / bounds.range_lon() * ((grid_width - 1) as f32);
            grid[(grid_lat.round() as usize) * grid_width + (grid_lon.round() as usize)] = 1;
        }

        GeoGrid {
            bounds: bounds,
            res_lat: res_lat,
            res_lon: res_lon,
            grid_height: grid_height,
            grid_width: grid_width,
            grid: grid,
        }
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

    /// Return the lat/lon boundaries of the grid.
    pub fn bbox(&self) -> Bounds {
        self.bounds
    }

    /// Grid dimensions.
    pub fn size(&self) -> (usize, usize) {
        (self.grid_height, self.grid_width)
    }

    /// Grid length, product of width and height.
    pub fn len(&self) -> usize {
        self.grid.len()
    }

    /// Grid is 0x0.
    pub fn is_empty(&self) -> bool {
        self.grid.is_empty()
    }


    /// Clears the grid, but leaves all other fields intact. Maybe useful to save memory if you
    /// only need the distance transform and not the original grid.
    pub fn clear_grid(&mut self) {
        self.grid.clear();
        self.grid.shrink_to_fit();
    }

    /// Latitude and longitude covered, in meters. resolution * size.
    pub fn real_size(&self) -> (f32, f32) {
        (self.res_lat * self.grid_height as f32, self.res_lon * self.grid_width as f32)
    }

    /// Immutable access to the underlying grid.
    pub fn grid(&self) -> &[u8] {
        &self.grid[..]
    }

    /// Mutable access to the underlying grid.
    pub fn grid_mut(&mut self) -> &mut [u8] {
        &mut self.grid[..]
    }


    /// Trace provided shape from given top left index, returning locations of all the true cells.
    pub fn trace_shape<BMatrix: AsRef<[BRow]>, BRow: AsRef<[bool]>>(&self,
                                                                    shape: BMatrix,
                                                                    topleft: usize)
                                                                    -> Vec<usize> {
        shape.as_ref()
            .iter()
            .enumerate()
            .flat_map(|(x, r)| {
                r.as_ref()
                    .iter()
                    .enumerate()
                    .map(move |(y, v)| (x, y, v))
            })
            .filter(|&(_, _, v)| *v)
            .map(|(x, y, _)| topleft + x * self.grid_width + y)
            .collect()
    }

    /// Return lat, lon coordinates of given index in the grid.
    pub fn to_lat_lon(&self, idx: usize) -> (f32, f32) {
        // TODO: account for latitude rounding error
        let (lat_len, lon_len) = lat_lon((self.bounds.south + self.bounds.north) / 2.0);
        let row = (idx / self.grid_width) as f32;
        let col = (idx % self.grid_width) as f32;
        (self.bounds.north - row * self.res_lat / lat_len,
         self.bounds.west + col * self.res_lon / lon_len)
    }

    /// Return closest index in grid to given latitude, longitude.
    pub fn near_lat_lon(&self, lat: f32, lon: f32) -> usize {
        // TODO: account for latitude rounding error
        let (ra, ri) = self.degree_resolution();
        let lat_offset = ((self.bounds.north - lat) / ra).round() as usize;
        let lon_offset = ((lon - self.bounds.west) / ri).round() as usize;
        lat_offset * self.grid_width + lon_offset
    }

    /// Return start index and a new size of a bounded subgrid specified by given rectangle in
    /// degrees.
    pub fn bounded_subgrid(&self,
                           north: f32,
                           south: f32,
                           east: f32,
                           west: f32)
                           -> (usize, (usize, usize)) {
        // TODO: guarantee valid subgrid boundaries
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
    pub fn l1dist_transform(&self) -> Vec<i32> {
        util::l1dist_transform(&self.grid, self.size())
    }
}

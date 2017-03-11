extern crate stopwatch;

extern crate geogrid;
extern crate osmlib;
extern crate bzip2;

use stopwatch::Stopwatch;
use std::fs::File;
use std::io::BufReader;

use bzip2::read::BzDecoder;
use geogrid::mat_to_img;
use osmlib::*;

fn main() {
    let mut s = Stopwatch::start_new();
    let filename = std::env::args().nth(1).unwrap();
    let file = File::open(&filename).unwrap();
    let nodes = if filename.ends_with(".bz2") {
        osm_to_nodes(BufReader::new(BzDecoder::new(BufReader::new(file))))
    } else {
        osm_to_nodes(BufReader::new(file))
    };
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

extern crate geogrid;
extern crate num;
extern crate quick_xml;
extern crate rayon;

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::str::from_utf8;

use num::Num;
use quick_xml::reader::Reader;
use quick_xml::events::Event;
use rayon::prelude::*;

use geogrid::Node;

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

pub fn osm_to_nodes<P: AsRef<Path>>(f: P) -> Vec<Node> {
    let mut r = Reader::from_file(f).unwrap();
    let mut nodes = HashMap::new();
    let mut ways = HashMap::new();
    let mut nd_vec = Vec::new();
    let mut current_way = -1i64;
    let mut buf = Vec::with_capacity(2048);
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
    ways.values().flat_map(|nds| nds.iter()).map(|nd| nodes[nd]).collect()
}

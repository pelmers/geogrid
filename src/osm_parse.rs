use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::str::from_utf8;

use geogrid::Node;
use quick_xml::reader::Reader;
use quick_xml::events::Event;

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

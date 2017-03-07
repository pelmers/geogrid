use std::f32;
use std::str::FromStr;
use std::io::BufReader;
use std::fs::File;
use std::collections::HashMap;

extern crate xml;
use xml::EventReader;
use xml::reader::XmlEvent;

#[derive(Debug)]
pub struct Node {
    lat: f32,
    lon: f32
}

fn main() {
    let f = File::open("geo.xml").unwrap();
    let r = EventReader::new(BufReader::new(f));
    let mut nodes = HashMap::new();
    let mut ways = HashMap::new();
    let mut nd_vec = Vec::new();
    let mut current_way = -1i64;
    for event_result in r {
        if !event_result.is_ok() {
            println!("XML event is not ok.");
            break;
        }
        let event = event_result.unwrap();
        match event {
            XmlEvent::StartElement{
                name: n, attributes: attrs, namespace: _
            } => {
                match n.local_name.as_ref() {
                    "node" => {
                        let mut o_id: Option<i64> = Option::None;
                        let mut o_lat: Option<f32> = Option::None;
                        let mut o_lon: Option<f32> = Option::None;
                        for attr in attrs {
                            match attr.name.local_name.as_ref() {
                                "id" => o_id = FromStr::from_str(attr.value.as_ref()).ok(),
                                "lat" => o_lat = FromStr::from_str(attr.value.as_ref()).ok(),
                                "lon" => o_lon = FromStr::from_str(attr.value.as_ref()).ok(),
                                _ => ()
                            }
                        }
                        if o_id.is_some() && o_lat.is_some() && o_lon.is_some() {
                            nodes.insert(o_id.unwrap(), Node{lat: o_lat.unwrap(), lon: o_lon.unwrap()});
                        }
                    },
                    "way" => {
                        for attr in attrs {
                            match attr.name.local_name.as_ref() {
                                "id" => current_way = FromStr::from_str(attr.value.as_ref()).unwrap_or(current_way),
                                _ => ()
                            }
                        }
                    },
                    "nd" => {
                        for attr in attrs {
                            match attr.name.local_name.as_ref() {
                                "ref" => {
                                    if let r = Ok(FromStr::from_str(attr.value.as_ref())) {
                                        nd_vec.push(r);
                                    }
                                },
                                _ => ()
                            }
                        }
                    },
                    _ => ()
                }
            },
            XmlEvent::EndElement{ name: n } => {
                if n.local_name == "way" {
                    ways.insert(current_way, nd_vec.clone());
                    nd_vec.clear();
                }
            },
            _ => ()
        };
    }
    // Now compute grid of width x height to quantize the space.
    let lat_min = nodes.values().map(|n| n.lat).fold(f32::MAX, f32::min);
    let lat_max = nodes.values().map(|n| n.lat).fold(f32::MIN, f32::max);
    let lon_min = nodes.values().map(|n| n.lon).fold(f32::MAX, f32::min);
    let lon_max = nodes.values().map(|n| n.lon).fold(f32::MIN, f32::max);
    println!("Processed {} nodes", nodes.len());
    println!("Found given bounds: [{}, {}] x [{}, {}]", lat_min, lat_max, lon_min, lon_max);
}

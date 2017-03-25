/// Represent some map coordinate, generally in degrees.
#[derive(Debug, Copy, Clone)]
pub struct Node {
    pub lat: f32,
    pub lon: f32,
}

/// Represent some map bounds, generally in degrees.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct Bounds {
    pub north: f32,
    pub south: f32,
    pub east: f32,
    pub west: f32,
}

impl Bounds {
    pub fn range_lat(&self) -> f32 {
        self.north - self.south
    }
    pub fn range_lon(&self) -> f32 {
        self.east - self.west
    }
}

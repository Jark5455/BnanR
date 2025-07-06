use cgmath::*;

use crate::core::ArcMut;
use crate::core::bnan_window::{BnanWindow, WindowObserver};

pub struct BnanCamera {
    pub window: ArcMut<BnanWindow>,
    pub position: Vector3<f32>,
    pub rotation: Vector3<f32>,

    pub move_sense: f32,
    pub look_sense: f32,
}

impl WindowObserver<(f32, f32)> for BnanCamera {
    fn update(&mut self, data: (f32, f32)) {
        let (x, y) = data;
        let vec = self.look_sense * Vector3::new(x, y, 0.0);
        self.rotation += vec;
        // self.rotation.y = self.rotation.y.clamp(-std::f32::consts::PI / 2.0, std::f32::consts::PI /  2.0);
    }
}

impl BnanCamera {
    pub fn new(window: ArcMut<BnanWindow>) -> BnanCamera {
        BnanCamera {
            window,
            position: Vector3::zero(),
            rotation: Vector3::zero(),
            move_sense: 10.0,
            look_sense: 0.0025,
        }
    }
    
    pub fn set_view_direction(&mut self, position: Vector3<f32>, direction: Vector3<f32>, up: Vector3<f32>) {
        self.position = position;
        self.rotation = direction;
    }
}
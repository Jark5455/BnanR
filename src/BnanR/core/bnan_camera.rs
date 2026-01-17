use cgmath::*;

use crate::core::ArcMut;
use crate::core::bnan_window::{BnanWindow, WindowObserver};

pub struct BnanCamera {
    pub position: Vector3<f32>,
    pub rotation: Vector3<f32>,

    pub projection_matrix: Matrix4<f32>,
    pub inverse_projection_matrix: Matrix4<f32>,
    pub view_matrix: Matrix4<f32>,
    pub inverse_view_matrix: Matrix4<f32>,

    pub move_sense: f32,
    pub look_sense: f32,
}

impl WindowObserver<(f32, f32)> for BnanCamera {
    fn update(&mut self, data: (f32, f32)) {
        let (x, y) = data;
        let vec = self.look_sense * Vector3::new(x, y, 0.0);

        self.set_view(self.position, self.rotation + vec);
    }
}

impl BnanCamera {

    pub fn new() -> BnanCamera {
        BnanCamera {
            position: Vector3::zero(),
            rotation: Vector3::zero(),

            projection_matrix: Matrix4::identity(),
            inverse_projection_matrix: Matrix4::identity(),
            view_matrix: Matrix4::identity(),
            inverse_view_matrix: Matrix4::identity(),

            move_sense: 10.0,
            look_sense: 0.0025,
        }
    }

    pub fn set_orthographic_projection(&mut self, left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) {
        self.projection_matrix = Matrix4::identity();
        self.projection_matrix[0][0] = 2.0 / (right - left);
        self.projection_matrix[1][1] = 2.0 / (bottom - top);
        self.projection_matrix[2][2] = 1.0 / (far - near);
        self.projection_matrix[3][0] = -(right + left) / (right - left);
        self.projection_matrix[3][1] = -(bottom + top) / (bottom - top);
        self.projection_matrix[3][2] = -near / (far - near);

        self.inverse_projection_matrix = Matrix4::zero();
        self.inverse_projection_matrix[0][0] = (right - left) / 2.0;
        self.inverse_projection_matrix[1][1] = (bottom - top) / 2.0;
        self.inverse_projection_matrix[2][2] = far - near;
        self.inverse_projection_matrix[3][0] = (left + right) / 2.0;
        self.inverse_projection_matrix[3][1] = (bottom + top) / 2.0;
        self.inverse_projection_matrix[3][2] = near;
    }

    pub fn set_perspective_projection(&mut self, fovy: f32, aspect: f32, near: f32, far: f32) {
        assert!((aspect - 10e-10).abs() > 0.0);
        let tan_half_fovy = (fovy / 2.0).tan();

        self.projection_matrix = Matrix4::zero();
        self.projection_matrix[0][0] = 1.0 / (aspect * tan_half_fovy);
        self.projection_matrix[1][1] = 1.0 / (tan_half_fovy);
        self.projection_matrix[2][2] = far / (far - near);
        self.projection_matrix[2][3] = 1.0;
        self.projection_matrix[3][2] = -(far * near) / (far - near);

        self.inverse_projection_matrix = Matrix4::zero();
        self.inverse_projection_matrix[0][0] = aspect * tan_half_fovy;
        self.inverse_projection_matrix[1][1] = tan_half_fovy;
        self.inverse_projection_matrix[2][2] = 0.0;
        self.inverse_projection_matrix[3][2] = 1.0;
        self.inverse_projection_matrix[2][3] = (near - far) / (near * far);
        self.inverse_projection_matrix[3][3] = 1.0 / near;
    }

    pub fn set_view(&mut self, position: Vector3<f32>, rotation: Vector3<f32>) {
        self.position = position;
        self.rotation = rotation;

        let c3 = rotation.z.cos();
        let s3 = rotation.z.sin();
        let c2 = rotation.x.cos();
        let s2 = rotation.x.sin();
        let c1 = rotation.y.cos();
        let s1 = rotation.y.sin();

        let u = Vector3 {x: c1 * c3 + s1 * s2 * s3, y: c2 * s3, z: c1 * s2 * s3 - c3 * s1};
        let v = Vector3 {x: c3 * s1 * s2 - c1 * s3, y: c2 * c3, z: c1 * c3 * s2 + s1 * s3};
        let w = Vector3 {x: c2 * s1, y: -s2, z: c1 * c2};

        self.view_matrix = Matrix4::identity();

        self.view_matrix[0][0] = u.x;
        self.view_matrix[1][0] = u.y;
        self.view_matrix[2][0] = u.z;
        self.view_matrix[0][1] = v.x;
        self.view_matrix[1][1] = v.y;
        self.view_matrix[2][1] = v.z;
        self.view_matrix[0][2] = w.x;
        self.view_matrix[1][2] = w.y;
        self.view_matrix[2][2] = w.z;
        self.view_matrix[3][0] = -u.dot(position);
        self.view_matrix[3][1] = -v.dot(position);
        self.view_matrix[3][2] = -w.dot(position);

        self.inverse_view_matrix = Matrix4::identity();

        self.inverse_view_matrix[0][0] = u.x;
        self.inverse_view_matrix[0][1] = u.y;
        self.inverse_view_matrix[0][2] = u.z;
        self.inverse_view_matrix[1][0] = v.x;
        self.inverse_view_matrix[1][1] = v.y;
        self.inverse_view_matrix[1][2] = v.z;
        self.inverse_view_matrix[2][0] = w.x;
        self.inverse_view_matrix[2][1] = w.y;
        self.inverse_view_matrix[2][2] = w.z;
        self.inverse_view_matrix[3][0] = position.x;
        self.inverse_view_matrix[3][1] = position.y;
        self.inverse_view_matrix[3][2] = position.z;
    }
}
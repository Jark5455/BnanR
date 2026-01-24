use cgmath::*;
use cgmath::num_traits::FloatConst;
use sdl3_sys::everything::{SDL_Scancode, SDL_SCANCODE_A, SDL_SCANCODE_D, SDL_SCANCODE_LCTRL, SDL_SCANCODE_S, SDL_SCANCODE_SPACE, SDL_SCANCODE_W};
use sdl3_sys::keyboard::SDL_GetKeyboardState;
use crate::core::bnan_window::{WindowObserver};

pub struct InternalKeyboardState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

enum Projection {
    Orthographic { aspect: f32, top: f32, bottom: f32, near: f32, far: f32 },
    Perspective { fovy: f32, aspect: f32, near: f32, far: f32 },
}

impl Default for Projection {
    fn default() -> Self {
        Projection::Orthographic {
            aspect: 1.0,
            top: -1.0,
            bottom: 0.0,
            near: -1.0,
            far: 1.0,
        }
    }
}

pub struct BnanCamera {
    pub position: Vector3<f32>,
    pub rotation: Vector3<f32>,

    pub projection: Projection,
    pub projection_matrix: Matrix4<f32>,
    pub inverse_projection_matrix: Matrix4<f32>,
    pub view_matrix: Matrix4<f32>,
    pub inverse_view_matrix: Matrix4<f32>,

    pub keyboard_state: InternalKeyboardState,
    pub move_sense: f32,
    pub look_sense: f32,
}

impl WindowObserver<(i32, i32)> for BnanCamera {
    fn update(&mut self, data: (i32, i32)) {

        let (width, height) = data;
        let aspect = width as f32 / height as f32;

        match self.projection {
            Projection::Orthographic { top, bottom, near, far, .. } => {
                self.set_orthographic_projection(aspect, top, bottom, near, far);
            },

            Projection::Perspective { fovy, near, far, .. } => {
                self.set_perspective_projection(fovy, aspect, near, far);
            },
        }
    }
}

impl WindowObserver<(f32, f32)> for BnanCamera {
    fn update(&mut self, data: (f32, f32)) {
        let (x, y) = data;
        let vec = self.look_sense * Vector3::new(-y, x, 0.0);

        let mut rot = self.rotation + vec;
        rot.x = rot.x.clamp(-1.5, 1.5);
        rot.y = rot.y % (f32::PI() * 2.0);

        self.set_view(self.position, rot);
    }
}

impl WindowObserver<()> for BnanCamera {

    fn update(&mut self, _data: ()) {

        const MOVE_FORWARD: SDL_Scancode = SDL_SCANCODE_W;
        const MOVE_BACK: SDL_Scancode = SDL_SCANCODE_S;

        const MOVE_LEFT: SDL_Scancode = SDL_SCANCODE_A;
        const MOVE_RIGHT: SDL_Scancode = SDL_SCANCODE_D;

        const MOVE_UP: SDL_Scancode = SDL_SCANCODE_SPACE;
        const MOVE_DOWN: SDL_Scancode = SDL_SCANCODE_LCTRL;

        unsafe {
            let state = SDL_GetKeyboardState(std::ptr::null_mut());

            self.keyboard_state.forward = *state.add(MOVE_FORWARD.0 as usize);
            self.keyboard_state.backward = *state.add(MOVE_BACK.0 as usize);
            self.keyboard_state.left = *state.add(MOVE_LEFT.0 as usize);
            self.keyboard_state.right = *state.add(MOVE_RIGHT.0 as usize);
            self.keyboard_state.up = *state.add(MOVE_UP.0 as usize);
            self.keyboard_state.down = *state.add(MOVE_DOWN.0 as usize);
        }
    }
}

impl BnanCamera {

    pub fn new() -> BnanCamera {
        BnanCamera {
            position: Vector3::zero(),
            rotation: Vector3::zero(),

            projection: Projection::default(),
            projection_matrix: Matrix4::identity(),
            inverse_projection_matrix: Matrix4::identity(),
            view_matrix: Matrix4::identity(),
            inverse_view_matrix: Matrix4::identity(),

            keyboard_state: InternalKeyboardState { up: false, down: false, forward: false, backward: false, left: false, right: false, },
            move_sense: 0.005,
            look_sense: 0.005,
        }
    }

    pub fn move_in_xz(&mut self, dt: f32) {
        let yaw = self.rotation.y;

        let forward = Vector3::new(yaw.sin(), 0.0, yaw.cos());
        let right = Vector3::new(forward.z, 0.0, -forward.x);
        let up = Vector3::new(0.0, -1.0, 0.0);

        let mut vec = Vector3::new(0.0, 0.0, 0.0);

        if self.keyboard_state.forward {
            vec += forward;
        }

        if self.keyboard_state.backward {
            vec -= forward;
        }

        if self.keyboard_state.left {
            vec -= right;
        }

        if self.keyboard_state.right {
            vec += right;
        }

        if self.keyboard_state.up {
            vec += up;
        }

        if self.keyboard_state.down {
            vec -= up;
        }

        if vec.magnitude() > f32::EPSILON {
            self.position += vec.normalize() * dt * self.move_sense;
            self.set_view(self.position, self.rotation);
        }
    }

    pub fn set_orthographic_projection(&mut self, aspect: f32, top: f32, bottom: f32, near: f32, far: f32) {

        let left = -aspect;
        let right = aspect;

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

        self.projection = Projection::Orthographic {
            aspect,
            top,
            bottom,
            near,
            far,
        };
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

        self.projection = Projection::Perspective {
            fovy,
            aspect,
            near,
            far,
        };
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
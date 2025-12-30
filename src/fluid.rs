use crate::DENSITY_DECAY;
use ocl::{ProQue, Buffer, MemFlags};
use std::fs;

pub fn ix(x: usize, y: usize, n: usize) -> usize {
    x + y * (n + 2)
}

pub struct Fluid {
    n: usize,
    dt: f32,
    diffusion_rate: f32,
    viscocity: f32,
    u: Buffer<f32>,
    v: Buffer<f32>,
    u_prev: Buffer<f32>,
    v_prev: Buffer<f32>,
    pub dens_host: Vec<f32>, 
    dens: Buffer<f32>,
    dens_prev: Buffer<f32>,
    pro_que: ProQue,
}

impl Fluid {
    pub fn new(n: usize, diffusion_rate: f32, viscocity: f32, dt: f32) -> Self {
        let size = (n + 2) * (n + 2);
        let kernel_src = fs::read_to_string("./kernels/lin_solve.cl").unwrap();
        let pro_que = ProQue::builder().src(kernel_src).dims(n * n).build().unwrap();

        let create_buf = || Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(size)
            .fill_val(0.0)
            .build().unwrap();

        Self {
            n, dt, diffusion_rate, viscocity,
            u: create_buf(), v: create_buf(),
            u_prev: create_buf(), v_prev: create_buf(),
            dens_host: vec![0.0; size],
            dens: create_buf(), dens_prev: create_buf(),
            pro_que,
        }
    }

    pub fn apply_mouse_force(&mut self, mx: usize, my: usize, brush_radius: f32, density: f32, fx: f32, fy: f32) {
        let kernel = self.pro_que.kernel_builder("inject_fluid_kernel")
            .arg(self.n as i32)
            .arg(mx as i32)
            .arg(my as i32)
            .arg(brush_radius)
            .arg(density)
            .arg(fx)
            .arg(fy)
            .arg(&self.dens)
            .arg(&self.u)
            .arg(&self.v)
            .build().unwrap();
        
        unsafe { kernel.enq().unwrap(); }
    }

    pub fn step(&mut self) {
        std::mem::swap(&mut self.u, &mut self.u_prev);
        std::mem::swap(&mut self.v, &mut self.v_prev);
        self.diffuse(1, &self.u, &self.u_prev, self.viscocity);
        self.diffuse(2, &self.v, &self.v_prev, self.viscocity);
        self.project();

        std::mem::swap(&mut self.u, &mut self.u_prev);
        std::mem::swap(&mut self.v, &mut self.v_prev);
        self.advect(1, &self.u, &self.u_prev, &self.u_prev, &self.v_prev);
        self.advect(2, &self.v, &self.v_prev, &self.u_prev, &self.v_prev);
        self.project();

        std::mem::swap(&mut self.dens, &mut self.dens_prev);
        self.diffuse(0, &self.dens, &self.dens_prev, self.diffusion_rate);
        std::mem::swap(&mut self.dens, &mut self.dens_prev);
        self.advect(0, &self.dens, &self.dens_prev, &self.u, &self.v);

        let size = (self.n + 2) * (self.n + 2);
        let decay_k = self.pro_que.kernel_builder("decay_kernel")
            .arg(size as i32).arg(&self.dens).arg(DENSITY_DECAY).global_work_size(size).build().unwrap();
        unsafe { decay_k.enq().unwrap(); }

        self.dens.read(&mut self.dens_host).enq().unwrap();
    }

    fn diffuse(&self, b: u32, x: &Buffer<f32>, x0: &Buffer<f32>, rate: f32) {
        let a = self.dt * rate * (self.n as f32).powi(2);
        self.lin_solve(b, x, x0, a, 1.0 + 4.0 * a);
    }

    fn lin_solve(&self, b: u32, x: &Buffer<f32>, x0: &Buffer<f32>, a: f32, c: f32) {
        let kernel = self.pro_que.kernel_builder("lin_solve_iter")
            .arg(self.n as i32).arg(x).arg(x0).arg(a).arg(c).build().unwrap();
        for _ in 0..20 {
            unsafe { kernel.enq().unwrap(); }
            self.set_bounds(b, x);
        }
    }

    fn advect(&self, b: u32, d: &Buffer<f32>, d0: &Buffer<f32>, u: &Buffer<f32>, v: &Buffer<f32>) {
        let kernel = self.pro_que.kernel_builder("advect_kernel")
            .arg(self.n as i32).arg(self.dt).arg(d).arg(d0).arg(u).arg(v).build().unwrap();
        unsafe { kernel.enq().unwrap(); }
        self.set_bounds(b, d);
    }

    fn project(&mut self) {
        let div = &self.u_prev;
        let p = &self.v_prev;
        
        let k1 = self.pro_que.kernel_builder("project_step1")
            .arg(self.n as i32).arg(&self.u).arg(&self.v).arg(p).arg(div).build().unwrap();
        unsafe { k1.enq().unwrap(); }
        self.set_bounds(0, div);
        self.set_bounds(0, p);

        self.lin_solve(0, p, div, 1.0, 4.0);

        let k2 = self.pro_que.kernel_builder("project_step2")
            .arg(self.n as i32).arg(&self.u).arg(&self.v).arg(p).build().unwrap();
        unsafe { k2.enq().unwrap(); }
        self.set_bounds(1, &self.u);
        self.set_bounds(2, &self.v);
    }

    fn set_bounds(&self, b: u32, x: &Buffer<f32>) {
        let k = self.pro_que.kernel_builder("set_bounds_kernel")
            .arg(self.n as i32).arg(b as i32).arg(x).global_work_size(self.n).build().unwrap();
        unsafe { k.enq().unwrap(); }
    }
}
use crate::DENSITY_DECAY;
use ocl::{ProQue};
use std::fs;

pub struct Fluid {
    n: usize,
    dt: f32,
    diffusion_rate: f32,
    viscocity: f32,
    u: Vec<f32>,
    v: Vec<f32>,
    u_prev: Vec<f32>,
    v_prev: Vec<f32>,
    pub dens: Vec<f32>,
    dens_prev: Vec<f32>,
    pro_que: ProQue, 
}

pub fn ix(x: usize, y: usize, n: usize) -> usize {
    x + y * (n + 2)
}

impl Fluid {
    pub fn new(n: usize, diffusion_rate: f32, viscocity: f32, dt: f32) -> Self {
        let size = (n + 2) * (n + 2);

        // OpenCL
        let kernel_src = fs::read_to_string("./kernels/lin_solve.cl")
            .expect("Failed to read OpenCL kernel file at ./kernels/lin_solve.cl");

        let dims = n * n;

        let pro_que = ProQue::builder()
            .src(kernel_src)
            .dims(dims)
            .build()
            .expect("Failed to build OpenCL ProQue");

        Self {
            n,
            dt,
            diffusion_rate,
            viscocity,
            u: vec![0.0; size],
            v: vec![0.0; size],
            u_prev: vec![0.0; size],
            v_prev: vec![0.0; size],
            dens: vec![0.0; size],
            dens_prev: vec![0.0; size],
            pro_que, 
        }
    }

    pub fn step(&mut self) {
        std::mem::swap(&mut self.u, &mut self.u_prev);
        std::mem::swap(&mut self.v, &mut self.v_prev);
        Fluid::diffuse(&self.pro_que, self.n, self.dt, 1, &mut self.u, &self.u_prev, self.viscocity);
        Fluid::diffuse(&self.pro_que, self.n, self.dt, 2, &mut self.v, &self.v_prev, self.viscocity);
        Fluid::project(
            &self.pro_que,
            self.n,
            &mut self.u,
            &mut self.v,
            &mut self.u_prev,
            &mut self.v_prev,
        );

        std::mem::swap(&mut self.u, &mut self.u_prev);
        std::mem::swap(&mut self.v, &mut self.v_prev);
        Fluid::advect(
            self.n,
            self.dt,
            1,
            &mut self.u,
            &self.u_prev,
            &self.u_prev,
            &self.v_prev,
        );
        Fluid::advect(
            self.n,
            self.dt,
            2,
            &mut self.v,
            &self.v_prev,
            &self.u_prev,
            &self.v_prev,
        );
        Fluid::project(
            &self.pro_que, 
            self.n,
            &mut self.u,
            &mut self.v,
            &mut self.u_prev,
            &mut self.v_prev,
        );

        std::mem::swap(&mut self.dens, &mut self.dens_prev);
        Fluid::diffuse(&self.pro_que, self.n, self.dt, 0, &mut self.dens, &self.dens_prev, self.diffusion_rate);
        std::mem::swap(&mut self.dens, &mut self.dens_prev);
        Fluid::advect(
            self.n,
            self.dt,
            0,
            &mut self.dens,
            &self.dens_prev,
            &self.u,
            &self.v,
        );

        for i in 0..self.dens.len() {
            self.dens[i] *= DENSITY_DECAY;
        }
    }

    pub fn add_density(&mut self, x: usize, y: usize, s: f32) {
        self.dens[ix(x, y, self.n)] += s;
    }

    pub fn add_velocity(&mut self, x: usize, y: usize, amount_x: f32, amount_y: f32) {
        let i = ix(x, y, self.n);
        self.u[i] += amount_x;
        self.v[i] += amount_y;
    }

    fn run_set_bounds_kernel(pro_que: &ProQue, n: usize, b: u32, buffer: &ocl::Buffer<f32>) {
        let n_i32 = n as i32;
        let size = (n + 2) * (n + 2);

        let kernel = pro_que.kernel_builder("set_bounds_kernel")
            .arg(n_i32)          
            .arg(b as i32)       
            .arg(buffer)         
            .global_work_size(size) 
            .build()
            .unwrap();
        
        unsafe {
            kernel.enq().unwrap();
        }
    }

    fn set_bounds(n: usize, b: u32, x: &mut [f32]) {
        for i in 1..=n {
            x[ix(0, i, n)] = if b == 1 { -x[ix(1, i, n)] } else { x[ix(1, i, n)] };
            x[ix(n + 1, i, n)] = if b == 1 { -x[ix(n, i, n)] } else { x[ix(n, i, n)] };
            x[ix(i, 0, n)] = if b == 2 { -x[ix(i, 1, n)] } else { x[ix(i, 1, n)] };
            x[ix(i, n + 1, n)] = if b == 2 { -x[ix(i, n, n)] } else { x[ix(i, n, n)] };
        }

        x[ix(0, 0, n)] = 0.5 * (x[ix(1, 0, n)] + x[ix(0, 1, n)]);
        x[ix(0, n + 1, n)] = 0.5 * (x[ix(1, n + 1, n)] + x[ix(0, n, n)]);
        x[ix(n + 1, 0, n)] = 0.5 * (x[ix(n, 0, n)] + x[ix(n + 1, 1, n)]);
        x[ix(n + 1, n + 1, n)] = 0.5 * (x[ix(n, n + 1, n)] + x[ix(n + 1, n, n)]);
    }

    fn lin_solve(pro_que: &ProQue, n: usize, b: u32, x: &mut [f32], x0: &[f32], a: f32, c: f32) {
        let n_i32 = n as i32;
        let size = (n + 2) * (n + 2); 
        let x_buffer = pro_que.buffer_builder::<f32>()
            .len(size)
            .copy_host_slice(x)
            .build()
            .unwrap();
        let x0_buffer = pro_que.buffer_builder::<f32>()
            .len(size)
            .copy_host_slice(x0)
            .build()
            .unwrap();

        for _ in 0..20 {
            let kernel = pro_que.kernel_builder("lin_solve_iter")
                .arg(n_i32)
                .arg(&x_buffer)
                .arg(&x0_buffer)
                .arg(a)
                .arg(c)
                .build()
                .unwrap();

            unsafe {
                kernel.enq().unwrap();
            }

            Fluid::run_set_bounds_kernel(pro_que, n, b, &x_buffer);
        }

        x_buffer.read(&mut *x).enq().unwrap();
        
        Fluid::set_bounds(n, b, x);
    }


    fn diffuse(pro_que: &ProQue, n: usize, dt: f32, b: u32, x: &mut [f32], x0: &[f32], diffusion_rate: f32) {
        let a = dt * diffusion_rate * n as f32 * n as f32;
        let c: f32 = 1.0 + 4.0 * a;
        Fluid::lin_solve(pro_que, n, b, x, x0, a, c);
    }

    fn advect(n: usize, dt: f32, b: u32, d: &mut [f32], d0: &[f32], u: &[f32], v: &[f32]) {
        let dt0 = dt * n as f32;
        for j in 1..=n {
            for i in 1..=n {
                let mut x = i as f32 - dt0 * u[ix(i, j, n)];
                let mut y = j as f32 - dt0 * v[ix(i, j, n)];

                if x < 0.5 {
                    x = 0.5;
                }
                if x > n as f32 + 0.5 {
                    x = n as f32 + 0.5;
                }
                let i0 = x.floor() as usize;
                let i1 = i0 + 1;

                if y < 0.5 {
                    y = 0.5;
                }
                if y > n as f32 + 0.5 {
                    y = n as f32 + 0.5;
                }
                let j0 = y.floor() as usize;
                let j1 = j0 + 1;

                let s1 = x - i0 as f32;
                let s0 = 1.0 - s1;
                let t1 = y - j0 as f32;
                let t0 = 1.0 - t1;

                d[ix(i, j, n)] = s0 * (t0 * d0[ix(i0, j0, n)] + t1 * d0[ix(i0, j1, n)])
                    + s1 * (t0 * d0[ix(i1, j0, n)] + t1 * d0[ix(i1, j1, n)]);
            }
        }
        Fluid::set_bounds(n, b, d);
    }

    fn project(pro_que: &ProQue, n: usize, u: &mut [f32], v: &mut [f32], p: &mut [f32], div: &mut [f32]) {
        let h = 1.0 / n as f32;

        for j in 1..=n {
            for i in 1..=n {
                div[ix(i, j, n)] = -0.5
                    * h
                    * (u[ix(i + 1, j, n)] - u[ix(i - 1, j, n)]
                        + v[ix(i, j + 1, n)] - v[ix(i, j - 1, n)]);
                p[ix(i, j, n)] = 0.0;
            }
        }

        Fluid::set_bounds(n, 0, div);
        Fluid::set_bounds(n, 0, p);
        
        // Use the GPU-accelerated solver
        Fluid::lin_solve(pro_que, n, 0, p, div, 1.0, 4.0);

        for j in 1..=n {
            for i in 1..=n {
                u[ix(i, j, n)] -= 0.5 * (p[ix(i + 1, j, n)] - p[ix(i - 1, j, n)]) / h;
                v[ix(i, j, n)] -= 0.5 * (p[ix(i, j + 1, n)] - p[ix(i, j - 1, n)]) / h;
            }
        }

        Fluid::set_bounds(n, 1, u);
        Fluid::set_bounds(n, 2, v);
    }
}
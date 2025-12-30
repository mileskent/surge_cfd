/*
    Surge CFD by Miles Kent

    Dynamics based on https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
    Used to showcase OpenCL + Rust
*/

use macroquad::prelude::*;
mod fluid; use fluid::*;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;
const GRID_WIDTH: usize = 200;
const GRID_HEIGHT: usize = GRID_WIDTH;
const GRID_SQUARE_WIDTH: f32 = WIDTH as f32 / GRID_WIDTH as f32;
const GRID_SQUARE_HEIGHT: f32 = HEIGHT as f32 / GRID_HEIGHT as f32;
const DT: f32 = 0.1;
const VISCOSITY: f32 = 0.00001;
const DIFFUSION: f32 = 0.00001;
pub const DENSITY_DECAY: f32 = 0.90;

fn window_conf() -> Conf {
    Conf {
        window_title: "Surge".to_owned(),
        window_width: WIDTH as i32,
        window_height: HEIGHT as i32,
        window_resizable: false,
        ..Default::default()
    }
}

fn density_to_color(density: f32) -> Color {
    let h: f32 = (2.0 / 3.0) * (1.0 - density.clamp(0.0, 1.0));
    let i: u32 = (h * 6.0) as u32;
    let f: f32 = (h * 6.0) - i as f32;
    let q: f32 = 1.0 - f;
    let (r, g, b) = match i % 6 {
        0 => (1.0, f, 0.0),
        1 => (q, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, q, 1.0),
        4 => (f, 0.0, 1.0),
        5 => (1.0, 0.0, q),
        _ => (1.0, f, 0.0),
    };
    Color::new(r, g, b, 1.0)
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut fluid = Fluid::new(GRID_WIDTH, DIFFUSION, VISCOSITY, DT);
    let mut last_mouse = mouse_position();

    loop {
        let curr_mouse = mouse_position();
        
        if is_mouse_button_down(MouseButton::Left) {
            let gx = (curr_mouse.0 / GRID_SQUARE_WIDTH) as usize;
            let gy = (curr_mouse.1 / GRID_SQUARE_HEIGHT) as usize;
            
            if gx > 0 && gx <= GRID_WIDTH && gy > 0 && gy <= GRID_HEIGHT {
                let dx = curr_mouse.0 - last_mouse.0;
                let dy = curr_mouse.1 - last_mouse.1;
                
                fluid.apply_mouse_force(
                    gx, gy, 
                    8.0,    
                    100.0,  
                    dx * 2.0,
                    dy * 2.0
                );
            }
        }
        last_mouse = curr_mouse;

        fluid.step();

        clear_background(BLUE);

        for j in 1..=GRID_HEIGHT {
            for i in 1..=GRID_WIDTH {
                let d = fluid.dens_host[ix(i, j, GRID_WIDTH)];
                if d > 0.1 {
                    draw_rectangle(
                        (i - 1) as f32 * GRID_SQUARE_WIDTH,
                        (j - 1) as f32 * GRID_SQUARE_HEIGHT,
                        GRID_SQUARE_WIDTH,
                        GRID_SQUARE_HEIGHT,
                        density_to_color(d),
                    );
                }
            }
        }

        draw_text(&format!("FPS: {}", get_fps()), 10.0, 20.0, 30.0, GREEN);
        next_frame().await
    }
}
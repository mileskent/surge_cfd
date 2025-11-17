/*
    Surge CFD by Miles Kent

    Based on https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
    We will just use this for something to have while transitioning to GPU
*/

use macroquad::prelude::*;
mod fluid; use fluid::*;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;
const GRID_WIDTH: usize = 200;
const GRID_HEIGHT: usize = GRID_WIDTH;
const GRID_SQUARE_WIDTH: u32 = WIDTH / GRID_WIDTH as u32;
const GRID_SQUARE_HEIGHT: u32 = HEIGHT / GRID_HEIGHT as u32;
const DT: f32 = 0.1;
const VISCOSITY: f32 = 0.0001;
const DIFFUSION: f32 = 0.0001;
const DENSITY_DECAY: f32 = 0.99;

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
    let mut v: f32 = density * 0.02;
    // clamp v to [0, 1]
    if v > 1.0 { v = 1.0; }
    if v < 0.0 { v = 0.0; }
    // map [0, 1] to [2/3, 0]
    let h: f32 = (2.0 / 3.0) * (1.0 - v);
    // calculate hues
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

fn draw_frame(fluid: &Fluid) {
    let mut image = Image::gen_image_color(WIDTH as u16, HEIGHT as u16, WHITE);

    for gy in 0..GRID_HEIGHT {
        for gx in 0..GRID_WIDTH {
            let dens_val = fluid.dens[ix(gx + 1, gy + 1, GRID_WIDTH)];
            let color = density_to_color(dens_val);
            set_grid_square(&mut image, gx as u32, gy as u32, color);
        }
    }

    let texture = Texture2D::from_image(&image);
    draw_texture(&texture, 0.0, 0.0, WHITE);
}

fn set_grid_square(image: &mut Image, gx: u32, gy: u32, color: Color) {
    for x in 0..GRID_SQUARE_WIDTH {
        for y in 0..GRID_SQUARE_HEIGHT {
            image.set_pixel(
                GRID_SQUARE_WIDTH * gx + x,
                GRID_SQUARE_HEIGHT * gy + y,
                color
            );
        }
    }
}

fn handle_mouse_button_down(
    curr_mouse_position: (f32, f32),
    last_mouse_position: (f32, f32),
    fluid: &mut Fluid
    ) {
    let brush_radius = 6;
    let force_multiplier = 500.0;
    if is_mouse_button_down(MouseButton::Left) {
        let (mouse_x, mouse_y) = curr_mouse_position;
        let dx = mouse_x - last_mouse_position.0;
        let dy = mouse_y - last_mouse_position.1;
        let grid_x = ((mouse_x / WIDTH as f32) * GRID_WIDTH as f32) as usize + 1;
        let grid_y = ((mouse_y / HEIGHT as f32) * GRID_HEIGHT as f32) as usize + 1;

        for j in -(brush_radius as i32)..=brush_radius {
            for i in -(brush_radius as i32)..=brush_radius {
                let x = (grid_x as i32 + i) as usize;
                let y = (grid_y as i32 + j) as usize;

                if x > 0 && x <= GRID_WIDTH && y > 0 && y <= GRID_HEIGHT {
                    let d2 = (i * i + j * j) as f32;
                    if d2 <= (brush_radius * brush_radius) as f32 {
                        let s = (1.0 - d2.sqrt() / (brush_radius as f32 + 0.0001)) * 50.0;
                        fluid.add_density(x, y, s);
                        fluid.add_velocity(x, y, dx * force_multiplier * 0.01, dy * force_multiplier * 0.01);
                    }
                }
            }
        }
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut fluid = Fluid::new(GRID_WIDTH, DIFFUSION, VISCOSITY, DT);
    let mut curr_mouse_position: (f32, f32) = (0.0, 0.0);
    let mut last_mouse_position: (f32, f32) = curr_mouse_position;

    loop {
        curr_mouse_position = mouse_position();
        handle_mouse_button_down(curr_mouse_position, last_mouse_position, &mut fluid);
        last_mouse_position = curr_mouse_position;

        fluid.step();
        draw_frame(&fluid);

        let fps = get_fps();
        let fps_text = format!("FPS: {}", fps);
        draw_text(
            &fps_text,
            10.0,            
            30.0,            
            30.0,            
            BLACK,           
        );

        next_frame().await;
    }
}
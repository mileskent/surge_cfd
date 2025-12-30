#define IX(i, j, N) ((i) + (j) * (N + 2))

/*
Handle mouse input
Inject fluid
*/
__kernel void inject_fluid_kernel(
const int N,
const int mouse_x,
const int mouse_y,
const float brush_radius,
const float density_amt,
const float force_x,
const float force_y,
__global float* dens,
__global float* u,
__global float* v)
{
int idx = get_global_id(0);
int i = (idx % N) + 1;
int j = (idx / N) + 1;

float dx = (float)i - (float)mouse_x;
float dy = (float)j - (float)mouse_y;
float dist_sq = dx * dx + dy * dy;
float r_sq = brush_radius * brush_radius;

if (dist_sq <= r_sq) {
    float strength = 1.0f - (sqrt(dist_sq) / (brush_radius + 0.0001f));
    dens[IX(i, j, N)] += strength * density_amt;
    u[IX(i, j, N)] += force_x * strength;
    v[IX(i, j, N)] += force_y * strength;
}


}

/*
Gauss-Seidel iterative solver.
Solves system of linear equations
*/
__kernel void lin_solve_iter(
const int N,
__global float* x,
__global const float* x0,
const float a,
const float c)
{
int idx = get_global_id(0);
int i = (idx % N) + 1;
int j = (idx / N) + 1;

x[IX(i, j, N)] = (x0[IX(i, j, N)] + a * (
    x[IX(i - 1, j, N)] + x[IX(i + 1, j, N)] +
    x[IX(i, j - 1, N)] + x[IX(i, j + 1, N)]
)) / c;


}

/*
Advection step.
Enforces that fluids at any location in
the grid maintain fluid properties 
like density or velocity across the flow field.
*/
__kernel void advect_kernel(
const int N,
const float dt,
__global float* d,
__global const float* d0,
__global const float* u,
__global const float* v)
{
int idx = get_global_id(0);
int i = (idx % N) + 1;
int j = (idx / N) + 1;

float dt0 = dt * (float)N;
float x = (float)i - dt0 * u[IX(i, j, N)];
float y = (float)j - dt0 * v[IX(i, j, N)];

if (x < 0.5f) x = 0.5f;
if (x > (float)N + 0.5f) x = (float)N + 0.5f;
int i0 = (int)x;
int i1 = i0 + 1;

if (y < 0.5f) y = 0.5f;
if (y > (float)N + 0.5f) y = (float)N + 0.5f;
int j0 = (int)y;
int j1 = j0 + 1;

float s1 = x - (float)i0;
float s0 = 1.0f - s1;
float t1 = y - (float)j0;
float t0 = 1.0f - t1;

d[IX(i, j, N)] = s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)])
               + s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);


}

/*
Projection. enforce grad u = 0.
Calculate divergence.
*/
__kernel void project_step1(
const int N,
__global const float* u,
__global const float* v,
__global float* p,
__global float* div)
{
int idx = get_global_id(0);
int i = (idx % N) + 1;
int j = (idx / N) + 1;

float h = 1.0f / (float)N;
div[IX(i, j, N)] = -0.5f * h * (
    u[IX(i + 1, j, N)] - u[IX(i - 1, j, N)] +
    v[IX(i, j + 1, N)] - v[IX(i, j - 1, N)]
);
p[IX(i, j, N)] = 0.0f;


}

/*
Projection. enforce grad u = 0.
Subtract pressure gradient from vels.
*/
__kernel void project_step2(
const int N,
__global float* u,
__global float* v,
__global const float* p)
{
int idx = get_global_id(0);
int i = (idx % N) + 1;
int j = (idx / N) + 1;

float h = 1.0f / (float)N;
u[IX(i, j, N)] -= 0.5f * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]) / h;
v[IX(i, j, N)] -= 0.5f * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]) / h;


}

/*
Make sure fluid stays inside the sim window
*/
__kernel void set_bounds_kernel(
const int N,
const int B,
__global float* x)
{
int idx = get_global_id(0);
int i = idx + 1;

if (i <= N) {
    x[IX(0, i, N)]     = (B == 1) ? -x[IX(1, i, N)] : x[IX(1, i, N)];
    x[IX(N + 1, i, N)] = (B == 1) ? -x[IX(N, i, N)] : x[IX(N, i, N)];
    x[IX(i, 0, N)]     = (B == 2) ? -x[IX(i, 1, N)] : x[IX(i, 1, N)];
    x[IX(i, N + 1, N)] = (B == 2) ? -x[IX(i, N, N)] : x[IX(i, N, N)];
}

if (idx == 0) {
    x[IX(0, 0, N)]         = 0.5f * (x[IX(1, 0, N)] + x[IX(0, 1, N)]);
    x[IX(0, N + 1, N)]     = 0.5f * (x[IX(1, N + 1, N)] + x[IX(0, N, N)]);
    x[IX(N + 1, 0, N)]     = 0.5f * (x[IX(N, 0, N)] + x[IX(N + 1, 1, N)]);
    x[IX(N + 1, N + 1, N)] = 0.5f * (x[IX(N, N + 1, N)] + x[IX(N + 1, N, N)]);
}


}

/*
Decay velocity
*/
__kernel void decay_kernel(const int size, __global float* dens, float decay) {
int idx = get_global_id(0);
if (idx < size) dens[idx] *= decay;
}
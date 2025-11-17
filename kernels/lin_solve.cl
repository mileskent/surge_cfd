#define IX(i, j, N) ((i) + (j) * (N + 2))

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
    
    x[IX(i, j, N)] = 
        (
            x0[IX(i, j, N)] +
            a * (
                x[IX(i - 1, j, N)] +
                x[IX(i + 1, j, N)] +
                x[IX(i, j - 1, N)] +
                x[IX(i, j + 1, N)]
            )
        )
        / c;
}

__kernel void set_bounds_kernel(
    const int N,
    const int B,
    __global float* x) 
{
    int idx = get_global_id(0);
    int size = (N + 2) * (N + 2);
    
    if (idx >= size) {
        return;
    }

    int i = idx % (N + 2);
    int j = idx / (N + 2);
    
    if (i == 0 && j > 0 && j <= N) { // Left Side
        x[IX(0, j, N)] = (B == 1) ? -x[IX(1, j, N)] : x[IX(1, j, N)];
    } else if (i == N + 1 && j > 0 && j <= N) { // Right Side
        x[IX(N + 1, j, N)] = (B == 1) ? -x[IX(N, j, N)] : x[IX(N, j, N)];
    } else if (j == 0 && i > 0 && i <= N) { // Bottom Side
        x[IX(i, 0, N)] = (B == 2) ? -x[IX(i, 1, N)] : x[IX(i, 1, N)];
    } else if (j == N + 1 && i > 0 && i <= N) { // Top Side
        x[IX(i, N + 1, N)] = (B == 2) ? -x[IX(i, N, N)] : x[IX(i, N, N)];
    }
    
    if (idx == 0) { // (0, 0)
        x[IX(0, 0, N)] = 0.5f * (x[IX(1, 0, N)] + x[IX(0, 1, N)]);
    } else if (idx == (N + 1)) { // (N+1, 0)
        x[IX(N + 1, 0, N)] = 0.5f * (x[IX(N, 0, N)] + x[IX(N + 1, 1, N)]);
    } else if (idx == size - (N + 2)) { // (0, N+1)
        x[IX(0, N + 1, N)] = 0.5f * (x[IX(1, N + 1, N)] + x[IX(0, N, N)]);
    } else if (idx == size - 1) { // (N+1, N+1)
        x[IX(N + 1, N + 1, N)] = 0.5f * (x[IX(N, N + 1, N)] + x[IX(N + 1, N, N)]);
    }
}
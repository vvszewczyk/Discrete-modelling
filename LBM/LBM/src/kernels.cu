#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Cell.h"

// Documentation: CUDA Programming Guide
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

__constant__ double tau = 1.0;

__global__ void collision(Cell* grid, int width, int height)
{
    // x and y indices for the current thread
    // blockIdx and threadIdx determine which thread in a given block deals with a given cell
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int idx = y * width + x;
    Cell& cell = grid[idx];

    if (cell.getWall())
    {
        return;
    }

    double C = 0.0;

    for (int i = 0; i < 5; ++i)
    {
        C += cell.getF_in(i);
    }

    cell.setC(C);

    // Weights
    double w0 = 1.0 / 3.0;
    double w1 = 1.0 / 6.0;

    cell.setF_eq(0, w0 * C); // 0
    cell.setF_eq(1, w1 * C); // N
    cell.setF_eq(2, w1 * C); // E
    cell.setF_eq(3, w1 * C); // S
    cell.setF_eq(4, w1 * C); // W

    // Collisions: f_out = f_in + (f_eq - f_in) / tau
    for (int i = 0; i < 5; ++i)
    {
        double f_in = cell.getF_in(i);
        double f_eq = cell.getF_eq(i);
        double f_out = f_in + (1.0 / tau) * (f_eq - f_in);
        cell.setF_out(i, f_out);
    }
}

__global__ void streaming(Cell* gridInput, Cell* gridOutput, int width, int height)
{
    // x and y indices for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    const Cell& inCell = gridInput[idx];

    Cell outCell;
    outCell.setWall(inCell.getWall());
    outCell.setC(inCell.getC());

    int cx[5] = { 0, 0, 1, 0, -1 };
    int cy[5] = { 0, 1, 0, -1, 0 };

    // For each f_in in outCell, we look for where f_out comes from:
    for (int i = 0; i < 5; ++i)
    {
        // Coordinates of neighbour cell
        int nx = x - cx[i];
        int ny = y - cy[i];

        double f_in_val = 0.0;

        // Check if neighbor cell is within bounds
        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
        {
            int neighborIdx = ny * width + nx;
            const Cell& neighborCell = gridInput[neighborIdx];

            if (neighborCell.getWall())
            {
                // bounce back
                // 0 -> 0
                // 1 (N) <-> 3 (S)
                // 2 (E) <-> 4 (W)
                int opposite[5] = { 0, 3, 4, 1, 2 };
                f_in_val = neighborCell.getF_out(opposite[i]);
            }
            else // Just streaming
            {
                f_in_val = neighborCell.getF_out(i);
            }
        }
        else
        {
            // Neighbor is out of bounds -> bounce back
            int opposite[5] = { 0, 3, 4, 1, 2 };
            f_in_val = inCell.getF_out(opposite[i]);
        }

        outCell.setF_in(i, f_in_val);
        outCell.setF_eq(i, inCell.getF_eq(i)); // From the previous collision step or copy
        outCell.setF_out(i, f_in_val); // From inCell or reset to f_in_val
    }

    gridOutput[idx] = outCell;
}

void collisionWrapper(Cell* grid, int width, int height)
{
    dim3 blockSize(32, 32); // maxThreadsPerBlock = 1024
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Number of blocks required to cover all grid

    collision <<<gridSize, blockSize >>> (grid, width, height); // Start kernel
    cudaDeviceSynchronize(); // CPU waits for GPU (synchronization, something like join)
}

void streamingWrapper(Cell* gridInput, Cell* gridOutput, int width, int height)
{
    dim3 blockSize(32, 32); // maxThreadsPerBlock = 1024
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Number of blocks required to cover all grid

    streaming <<<gridSize, blockSize >>> (gridInput, gridOutput, width, height); // Start kernel
    cudaDeviceSynchronize(); // CPU waits for GPU (synchronization, something like join)
}
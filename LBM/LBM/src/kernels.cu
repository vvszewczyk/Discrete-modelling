#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Cell.h"

// Documentation: CUDA Programming Guide
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

__global__ void collision(Cell* grid, int width, int height, double tau)
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
    // Calcualte concentration in cell (sum of distribution function - 9)
    for (int i = 0; i < 4; ++i)
    {
        C += cell.getF_in(i);
    }

    cell.setC(C);

    // Weights D2Q4
    double w = 1.0 / 4.0;

    // Calculate f_eq (equilibrium distribution function - 10)
    for (int i = 0; i < 4; ++i)
    {
        double f_eq = w * C;
        cell.setF_eq(i, f_eq);
    }

    // Collisions: f_out = f_in + (1.0 / tau) * (f_eq - f_in) - 11
    for (int i = 0; i < 4; ++i)
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

    int cx[4] = { 0, 1, 0, -1 };
    int cy[4] = { 1, 0, -1, 0 };

    int oppositeDir[4] = { 2, 3, 0, 1 };

    // For each f_in in outCell, we look for where f_out comes from:
    for (int i = 0; i < 4; ++i)
    {
        // Coordinates of neighbour cell
        int nx = x - cx[i];
        int ny = y - cy[i];

        double f_inVal = 0.0;

        // Check if neighbor cell is within bounds
        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
        {
            int neighborIdx = ny * width + nx;
            const Cell& neighborCell = gridInput[neighborIdx];

            if (neighborCell.getWall())
            {
                // bounce back
                // 0 <-> 2 (N <-> S)
                // 1 <-> 3 (E <-> W)
                f_inVal = neighborCell.getF_out(oppositeDir[i]);
            }
            else // Just streaming
            {
                f_inVal = neighborCell.getF_out(i);
            }
        }
        else
        {
            // Neighbor is out of bounds -> bounce back
            f_inVal = inCell.getF_out(oppositeDir[i]);
        }

        outCell.setF_in(i, f_inVal);
        outCell.setF_eq(i, inCell.getF_eq(i)); // From the previous collision
        outCell.setF_out(i, f_inVal); // Reset to f_in_val
    }

    gridOutput[idx] = outCell;
}

void collisionWrapper(Cell* grid, int width, int height, double tau)
{
    dim3 blockSize(16, 16); // maxThreadsPerBlock = 1024
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Number of blocks required to cover all grid

    collision <<<gridSize, blockSize >>> (grid, width, height, tau); // Start kernel
    cudaDeviceSynchronize(); // CPU waits for GPU (synchronization, something like join)
}

void streamingWrapper(Cell* gridInput, Cell* gridOutput, int width, int height)
{
    dim3 blockSize(16, 16); // maxThreadsPerBlock = 1024
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Number of blocks required to cover all grid

    streaming <<<gridSize, blockSize >>> (gridInput, gridOutput, width, height); // Start kernel
    cudaDeviceSynchronize(); // CPU waits for GPU (synchronization, something like join)
}
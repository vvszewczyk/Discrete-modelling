#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Cell.h"

// Documentation: CUDA Programming Guide
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

__global__ void collisionKernel(Cell* grid, int width, int height)
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

    bool north = cell.getDirection(0);
    bool east = cell.getDirection(1);
    bool south = cell.getDirection(2);
    bool west = cell.getDirection(3);

    // Collisions: (1, 0, 1, 0) ? (0, 1, 0, 1)
    if (north && south && !east && !west)
    {
        cell.setDirection(0, false);
        cell.setDirection(1, true);
        cell.setDirection(2, false);
        cell.setDirection(3, true);
    }
    else if (east && west && !north && !south)
    {
        cell.setDirection(0, true);
        cell.setDirection(1, false);
        cell.setDirection(2, true);
        cell.setDirection(3, false); 
    }
}

__global__ void streamingKernel(Cell* gridInput, Cell* gridOutput, int width, int height)
{
    // x and y indices for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    const Cell& cell = gridInput[idx];

    if (cell.getWall())
    {
        // Copy wall cells directly
        gridOutput[idx] = cell;
        return;
    }

    Cell newCell;
    newCell.setWall(false);
    newCell.resetDirections();

    // Check incoming particles and handle reflections
    for (int d = 0; d < 4; ++d)
    {
        // Coordinates of neighbour cell
        int nx = x;
        int ny = y;

        // Direction from which particles are coming into the current cell
        switch (d)
        {
        case 0: // North (particles coming from cell below)
            ny = y - 1;
            break;
        case 1: // East (particles coming from cell to the left)
            nx = x - 1;
            break;
        case 2: // South (particles coming from cell above)
            ny = y + 1;
            break;
        case 3: // West (particles coming from cell to the right)
            nx = x + 1;
            break;
        }

        // Opposite direction (for reflections)
        int oppositeDir = (d + 2) % 4;

        bool particleMoved = false;

        // Check if neighbor cell is within bounds
        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
        {
            int neighborIdx = ny * width + nx;
            const Cell& neighborCell = gridInput[neighborIdx];

            if (neighborCell.getWall())
            {
                // Neighbor is a wall; check if particle in current cell needs to reflect
                if (cell.getDirection(oppositeDir))
                {
                    newCell.setDirection(d, true); // Reflect particle
                }
            }
            else
            {
                // Neighbor is not a wall; check if particle is moving towards current cell
                if (neighborCell.getDirection(d))
                {
                    newCell.setDirection(d, true); // Particle moves into current cell
                    particleMoved = true;
                }
            }
        }
        else
        {
            // Neighbor is out of bounds; treat as wall
            if (cell.getDirection(oppositeDir))
            {
                newCell.setDirection(d, true); // Reflect particle
            }
        }

        // If no particle moved into current cell and the current cell has a particle moving in oppositeDir,
        // check if it needs to reflect due to wall or boundary
        if (!particleMoved && cell.getDirection(oppositeDir))
        {
            // Check if neighbor is wall or boundary
            if (nx < 0 || nx >= width || ny < 0 || ny >= height || (gridInput[ny * width + nx].getWall()))
            {
                newCell.setDirection(d, true); // Reflect particle
            }
        }
    }

    gridOutput[idx] = newCell;
}

void collisionKernelWrapper(Cell* grid, int width, int height)
{
    dim3 blockSize(32, 32); // maxThreadsPerBlock = 1024
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Number of blocks required to cover all grid

    collisionKernel <<<gridSize, blockSize >>> (grid, width, height); // Start kernel
    cudaDeviceSynchronize(); // CPU waits for GPU (synchronization, something like join)
}

void streamingKernelWrapper(Cell* gridInput, Cell* gridOutput, int width, int height)
{
    dim3 blockSize(32, 32); // maxThreadsPerBlock = 1024
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); // Number of blocks required to cover all grid

    streamingKernel <<<gridSize, blockSize >>> (gridInput, gridOutput, width, height); // Start kernel
    cudaDeviceSynchronize(); // CPU waits for GPU (synchronization, something like join)
}
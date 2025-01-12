#include <cuda_runtime.h>
#include <string>
#include "device_launch_parameters.h"
#include "Cell.h"
#include "kernels.h"

// Documentation: CUDA Programming Guide
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

__device__ __host__ int findOpposite(int i)
{
    switch (i)
    {
    case 0: return 0;
    case 1: return 2;
    case 2: return 1;
    case 3: return 4;
    case 4: return 3;
    case 5: return 7;
    case 6: return 8;
    case 7: return 5;
    case 8: return 6;
    }
    return 0; // default
}

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

    double rho = 0.0;
    // Calcualte rho in cell (sum of distribution function)
    for (int i = 0; i < Q9; ++i)
    {
        rho += cell.getF_in(i);
    }

    // ux * rho, uy * rho
    double uxRho = 0.0;
    double uyRho = 0.0;

    for (int i = 0; i < Q9; ++i)
    {
        uxRho += cell.getF_in(i) * cx[i];
        uyRho += cell.getF_in(i) * cy[i];
    }

    double ux = 0.0;
    double uy = 0.0;

    if (rho > 1e-14)
    {
        ux = uxRho / rho;
        uy = uyRho / rho;
    }

    cell.setRho(rho);
    cell.setUx(ux);
    cell.setUy(uy);

    // f_i^{eq} = w_i * rho * [1 + 3(c_i.u) + 4.5(c_i.u)^2 - 1.5 * (u^2)]
    for (int i = 0; i < Q9; ++i)
    {
        double c_iu = cx[i] * ux + cy[i] * uy; // (c_i · u)
        double uu = ux * ux + uy * uy; // u^2
        double term1 = 1.0 + (3.0 * c_iu);
        double term2 = 4.5 * (c_iu * c_iu);
        double term3 = -1.5 * uu;
        double feq = w[i] * rho * (term1 + term2 + term3);
        cell.setF_eq(i, feq);
    }


    // Collisions: fout = fin + (1.0 / tau) * (feq - fin)
    for (int i = 0; i < Q9; ++i)
    {
        double fin = cell.getF_in(i);
        double feq = cell.getF_eq(i);
        double fout = fin + (1.0 / tau) * (feq - fin);
        cell.setF_out(i, fout);
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
    outCell.setRho(inCell.getRho());
    outCell.setUx(inCell.getUx());
    outCell.setUy(inCell.getUy());

    // For each f_in in outCell, we look for where f_out comes from:
    for (int i = 0; i < Q9; ++i)
    {
        // Coordinates of neighbour cell
        int nx = x - cx[i];
        int ny = y - cy[i];

        double f_inVal = 0.0;

        // Check if neighbor cell is within bounds

        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
        {
            // np. outflow 
            f_inVal = 0.0;
        }
        else
        {
            int neighborIdx = ny * width + nx;
            const Cell& neighborCell = gridInput[neighborIdx];
            f_inVal = neighborCell.getF_out(i);
        }


        outCell.setF_in(i, f_inVal);
        outCell.setF_eq(i, inCell.getF_eq(i)); // From the previous collision
        outCell.setF_out(i, f_inVal); // Reset to f_in_val
    }

    gridOutput[idx] = outCell;
}

__global__ void boundaryConditions(Cell* grid, int width, int height)
{
    // x and y indices for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Not boundary = do nothing
    bool isTop = (y == height - 1);
    bool isBottom = (y == 0);
    bool isLeft = (x == 0);
    bool isRight = (x == width - 1);

    if (!isTop && !isBottom && !isLeft && !isRight)
        return; // Inside grid

    int idx = y * width + x;
    Cell& cell = grid[idx];

    double rhoBC = 1.0;
    double uxBC = 0.0;
    double uyBC = 0.0;

    // Top: ux=0.02
    if (isTop) 
    {
        uxBC = 0.02;
        uyBC = 0.0;
    }
    // Bottom: ux=0.0
    if (isBottom) 
    {
        uxBC = 0.0;
        uyBC = 0.0;
    }
    // Left and right: linear change from bottom to top <0.0, 0.02>
    //    "alpha" - position <0.0, 1.0>
    if (isLeft || isRight) 
    {
        double alpha = (double)y / (double)(height - 1);
        uxBC = 0.02 * alpha; // growing from 0.0 (y=0) to 0.02 (y=height-1)
        uyBC = 0.0;
    }

    cell.setRho(rhoBC);
    cell.setUx(uxBC);
    cell.setUy(uyBC);

    // f_i^eq = w_i * rhoBC * [1 + 3(c_i·uBC) + 4.5(c_i·uBC)^2 - 1.5(uBC^2)]
    // Overriting f_out[i] = feq[i]
    double uu = uxBC * uxBC;

    for (int i = 0; i < Q9; ++i)
    {
        double ciU = (double)cx[i] * uxBC + (double)cy[i] * uyBC; // c_i · u
        double feq = w[i] * rhoBC * (1.0 + 3.0 * ciU + 4.5 * (ciU * ciU) - 1.5 * uu);
        cell.setF_out(i, feq);
    }
}

__global__ void boundaryConditions2(Cell* grid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    bool isBottom = (y == 0);
    bool isTop = (y == height - 1);
    bool isLeft = (x == 0);
    bool isRight = (x == width - 1);

    if (!isBottom && !isTop && !isLeft && !isRight)
        return;

    int idx = y * width + x;
    Cell& cell = grid[idx];

    // 1) BOUNCE-BACK on bottom
    if (isBottom)
    {
        // veolcity (u=0)
        double rhoBC = 1.0;
        double uxBC = 0.0;
        double uyBC = 0.0;

        cell.setRho(rhoBC);
        cell.setUx(uxBC);
        cell.setUy(uyBC);

        // f_out[i] = f_eq(rhoBC, u=0)
        double uu = 0.0;
        for (int i = 0; i < Q9; ++i)
        {
            double ciU = 0.0; // u=0
            double feq = w[i] * rhoBC * (1.0 + 3.0 * ciU + 4.5 * (ciU * ciU) - 1.5 * uu);
            cell.setF_out(i, feq);
        }
        return;
    }

    // 2) SYMETRY on top
    if (isTop)
    {

        double rhoBC = 1.0;
        double uxBC = cell.getUx();
        double uyBC = 0.0;
        cell.setRho(rhoBC);
        cell.setUx(uxBC);
        cell.setUy(uyBC);

        // f_out[i] = feq(rhoBC, [uxBC,0]).
        double uu = uxBC * uxBC;
        for (int i = 0; i < Q9; i++)
        {
            double cidotU = cx[i] * uxBC;
            double feq = w[i] * rhoBC * (1 + 3 * cidotU + 4.5 * (cidotU * cidotU) - 1.5 * uu);
            cell.setF_out(i, feq);
        }
        return;
    }

    // 3) OPEN - left (0 -> 0.02)
    if (isLeft)
    {
        double alpha = (double)y / (double)(height - 1);
        double uWanted = 0.02 * alpha;
        double rhoBC = 1.0;
        double uxBC = uWanted;
        double uyBC = 0.0;

        cell.setRho(rhoBC);
        cell.setUx(uxBC);
        cell.setUy(uyBC);

        // f_out = feq(rhoBC, uxBC, 0)
        double uu = uxBC * uxBC;
        for (int i = 0; i < Q9; i++)
        {
            double ciU = cx[i] * uxBC;
            double feq = w[i] * rhoBC * (1 + 3 * ciU + 4.5 * (ciU * ciU) - 1.5 * uu);
            cell.setF_out(i, feq);
        }
        return;
    }

    // 4) OUTFLOW - right
    if (isRight)
    {
        double rhoBC = 1.0;
        double uxBC = 0.0;
        double uyBC = 0.0;

        cell.setRho(rhoBC);
        cell.setUx(uxBC);
        cell.setUy(uyBC);

        double uu = uxBC * uxBC + uyBC * uyBC;
        for (int i = 0; i < Q9; i++)
        {
            double ciU = cx[i] * uxBC + cy[i] * uyBC;
            double feq = w[i] * rhoBC * (1 + 3 * ciU + 4.5 * (ciU * ciU) - 1.5 * uu);
            cell.setF_out(i, feq);
        }
        return;
    }
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

void boundaryWrapper(Cell* grid, int width, int height, std::string variant)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    if(variant == "Variant 1")
    {
        boundaryConditions << <gridSize, blockSize >> > (grid, width, height); // Start kernel
    }
    else
    {
        boundaryConditions2 << <gridSize, blockSize >> > (grid, width, height); // Start kernel
    }
    cudaDeviceSynchronize(); // CPU waits for GPU (synchronization, something like join)
}
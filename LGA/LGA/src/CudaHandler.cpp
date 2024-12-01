#include "CudaHandler.h"
#include <cuda_runtime.h>
#include "kernels.h"
#include <iostream>

CudaHandler::CudaHandler(int width, int height) : gridWidth(width), gridHeight(height)
{
	allocateMemory();
}

CudaHandler::~CudaHandler()
{
	freeMemory();
}

void CudaHandler::allocateMemory()
{
	size_t size = gridWidth * gridHeight * sizeof(Cell);
	cudaMalloc((void**)&gridInput, size);
	cudaMalloc((void**)&gridOutput, size);
}

void CudaHandler::copyGridToGPU(const Cell* h_grid)
{
	size_t size = gridWidth * gridHeight * sizeof(Cell);
	cudaMemcpy(gridInput, h_grid, size, cudaMemcpyHostToDevice);
}

void CudaHandler::copyGridToCPU(Cell* h_grid)
{
	size_t size = gridWidth * gridHeight * sizeof(Cell);
	cudaMemcpy(h_grid, gridInput, size, cudaMemcpyDeviceToHost);
}

void CudaHandler::freeMemory()
{
	cudaFree(gridInput);
	cudaFree(gridOutput);
}

void CudaHandler::executeCollisionKernel()
{
	collisionKernelWrapper(gridInput, gridWidth, gridHeight);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Error in executeCollisionKernel: " << cudaGetErrorString(err) << std::endl;
	}
}

void CudaHandler::executeStreamingKernel()
{
	streamingKernelWrapper(gridInput, gridOutput, gridWidth, gridHeight);

	// Swap the grids
	Cell* temp = gridInput;
	gridInput = gridOutput;
	gridOutput = temp;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Error in executeStreamingKernel: " << cudaGetErrorString(err) << std::endl;
	}
}


void CudaHandler::initializeDeviceGrids(const Cell* grid)
{
	size_t size = gridWidth * gridHeight * sizeof(Cell);
	cudaMemcpy(gridInput, grid, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gridOutput, grid, size, cudaMemcpyHostToDevice);
}


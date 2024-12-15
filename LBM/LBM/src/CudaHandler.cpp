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
	size_t size = gridWidth * gridHeight * sizeof(Cell); // Compute size of memory required to allocate grid
	cudaMalloc((void**)&gridInput, size);
	cudaMalloc((void**)&gridOutput, size);
}

void CudaHandler::copyGridToGPU(const Cell* grid)
{
	size_t size = gridWidth * gridHeight * sizeof(Cell);
	cudaMemcpy(gridInput, grid, size, cudaMemcpyHostToDevice);
}

void CudaHandler::copyGridToCPU(Cell* grid)
{
	size_t size = gridWidth * gridHeight * sizeof(Cell);
	cudaMemcpy(grid, gridInput, size, cudaMemcpyDeviceToHost);
}

void CudaHandler::freeMemory()
{
	cudaFree(gridInput);
	cudaFree(gridOutput);
}

void CudaHandler::executeCollision(double tau)
{
	collisionWrapper(gridInput, gridWidth, gridHeight, tau);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Error in executeCollision: " << cudaGetErrorString(err) << std::endl;
	}
}

void CudaHandler::executeStreaming()
{
	streamingWrapper(gridInput, gridOutput, gridWidth, gridHeight);

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


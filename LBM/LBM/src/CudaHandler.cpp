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
	size_t size = this->gridWidth * this->gridHeight * sizeof(Cell); // Compute size of memory required to allocate grid
	cudaMalloc((void**)&this->gridInput, size);
	cudaMalloc((void**)&this->gridOutput, size);
}

void CudaHandler::copyGridToGPU(const Cell* grid)
{
	size_t size = this->gridWidth * this->gridHeight * sizeof(Cell);
	cudaMemcpy(this->gridInput, grid, size, cudaMemcpyHostToDevice);
}

void CudaHandler::copyGridToCPU(Cell* grid)
{
	size_t size = this->gridWidth * this->gridHeight * sizeof(Cell);
	cudaMemcpy(grid, this->gridInput, size, cudaMemcpyDeviceToHost);
}

void CudaHandler::freeMemory()
{
	cudaFree(this->gridInput);
	cudaFree(this->gridOutput);
}

void CudaHandler::executeCollision(double tau)
{
	collisionWrapper(this->gridInput, this->gridWidth, this->gridHeight, tau);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Error in executeCollision: " << cudaGetErrorString(err) << std::endl;
	}
}

void CudaHandler::executeStreaming()
{
	streamingWrapper(this->gridInput, this->gridOutput, this->gridWidth, this->gridHeight);

	Cell* temp = this->gridInput;
	this->gridInput = this->gridOutput;
	this->gridOutput = temp;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA Error in executeStreamingKernel: " << cudaGetErrorString(err) << std::endl;
	}
}


void CudaHandler::initializeDeviceGrids(const Cell* grid)
{
	size_t size = this->gridWidth * this->gridHeight * sizeof(Cell);
	cudaMemcpy(this->gridInput, grid, size, cudaMemcpyHostToDevice);
	cudaMemcpy(this->gridOutput, grid, size, cudaMemcpyHostToDevice);
}


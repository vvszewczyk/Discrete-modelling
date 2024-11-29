#pragma once

#include <vector>
#include "Cell.h"

class CudaHandler
{
private:
	Cell* gridInput; // input for GPU
	Cell* gridOutput; // output for GPU

	int gridWidth;
	int gridHeight;

public:
	CudaHandler(int width, int height) : gridWidth(width), gridHeight(height), gridInput(nullptr), gridOutput(nullptr) {};
	~CudaHandler()
	{
		//freeMemory();
	}

	// Initialization memory for GPU
	//void allocateMemory();

	// Data transfer GPU <-> CPU
	//void copyGridToGPU(const std::vector<std::vector<Cell>>& grid);
	//void copyGridToCPU(std::vector<std::vector<Cell>>& grid);

	// Trigger kernels
	//void executeCollisionKernel();
	//void executeStreamingKernel();

	// Free GPU memory
	//void freeMemory();
};
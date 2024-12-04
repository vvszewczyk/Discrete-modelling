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
	CudaHandler(int width, int height);
	~CudaHandler();

	void allocateMemory(); // Initialization memory for GPU

	// Data transfer GPU <-> CPU
	void copyGridToGPU(const Cell* grid);
	void copyGridToCPU(Cell* rid);

	// Trigger kernels
	void executeCollisionKernel();
	void executeStreamingKernel();

	void freeMemory(); // Free GPU memory

	void initializeDeviceGrids(const Cell* grid); // Copy both input and output grids on GPU
};
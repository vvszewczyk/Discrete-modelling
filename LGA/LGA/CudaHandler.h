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

	// Initialization memory for GPU
	void allocateMemory();

	// Data transfer GPU <-> CPU
	void copyGridToGPU(const Cell* h_grid);
	void copyGridToCPU(Cell* h_grid);

	// Trigger kernels
	void executeCollisionKernel();
	void executeStreamingKernel();

	// Free GPU memory
	void freeMemory();

	void initializeDeviceGrids(const Cell* h_grid);
};
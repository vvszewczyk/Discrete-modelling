#include "Grid.h"
#include "CudaHandler.h"
#include "SimulationController.h"
#include <iostream>

int main(int argc, char** argv)
{
	// Grid size
	const int gridWidth = 200;
	const int gridHeight = 200;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // 0 is the device ID
	std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max threads per dimension: " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << std::endl;
	std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;


	// Window size
	const int windowWidht = 800;
	const int windowHeight = 800;
	const int cellSize = windowWidht / gridWidth;

	// Grid initializing
	Grid grid(gridWidth, gridHeight);
	grid.initialize();

	// CUDA initializing
	CudaHandler cudaHandler(gridWidth, gridHeight);
	cudaHandler.initializeDeviceGrids(grid.getGridData());

	// SimulationController initializing
	SimulationController controller(&grid, &cudaHandler, windowWidht, windowHeight, cellSize);
	controller.initializeUI(argc, argv);

	glutMainLoop();
}
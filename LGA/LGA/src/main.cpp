#include "Grid.h"
#include "CudaHandler.h"
#include "SimulationController.h"

int main(int argc, char** argv)
{
	// Grid size
	const int gridWidth = 800;
	const int gridHeight = 800;

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
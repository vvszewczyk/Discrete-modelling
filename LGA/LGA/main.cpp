#include "Grid.h"
#include "CudaHandler.h"
#include "SimulationController.h"

int main(int argc, char** argv)
{
	// Grid size
	const int gridWidth = 100;
	const int gridHeight = 100;

	// Window size
	const int windowWidht = 800;
	const int windowHeight = 800;
	const int cellSize = windowWidht / gridWidth;

	// Grid initializing
	Grid grid(gridWidth, gridHeight);
	grid.initialize();
	grid.getCell(10, 15).setDirection(0, true);


	// CUDA initializing
	CudaHandler cudaHandler(gridWidth, gridHeight);
	//cudaHandler.allocateMemory();
	//cudaHandler.copyGridToGPU(grid);

	SimulationController controller(&grid, &cudaHandler, windowWidht, windowHeight, cellSize);
	controller.initializeOpenGL(argc, argv);

	glutMainLoop();
}
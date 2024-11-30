#pragma once

#include "Grid.h"
#include "CudaHandler.h"
#include <GL/freeglut.h>

class SimulationController
{
private:
	Grid* grid; // Pointer on grid
	CudaHandler* cudaHandler; // Pointer on CUDA handler
	bool isRunning; // Simulation flag

	// Size of window
	int windowWidth;
	int windowHeight;
	int cellSize; // Size of single cell in pixels

	std::string buttonLabel = "Start";
	bool placingWalls = false;

public:
	SimulationController(Grid* g, CudaHandler* ch, int width, int height, int cs);
	~SimulationController();

	// Manage simulation methods
	void startSimulation();
	void stopSimulation();
	void toggleSimulation();
	void resetSimulation();

	void initializeOpenGL(int argc, char** argv);
	static void display(); // Function for OpenGL for drawing
	static void reshape(int w, int h); // Change size of window
	static void keyboard(unsigned char key, int x, int y); // Keyboard operation
	
	void drawButton(float x, float y, float width, float height, const char* label);
	static void mouseHandler(int button, int state, int x, int y);
	void toggleWallPlacement();

	void updateSimulation(); // Update simulation state
};
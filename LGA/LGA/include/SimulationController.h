#pragma once

#include "Grid.h"
#include "CudaHandler.h"
#include <GL/freeglut.h>
#include <string>


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

	std::string buttonLabelSS; // Button label for stop/start
	bool placingWalls = false;
	bool isLeftMouseDown = false;
	std::string buttonLabelWE; // Button label for wall/empty
	int stepsPerFrame; // For managing simulation speed

	bool gridModified;
	static const int MIN_STEPS_PER_FRAME = 1;
	static const int MAX_STEPS_PER_FRAME = 20;

public:
	SimulationController(Grid* g, CudaHandler* ch, int width, int height, int cs);
	~SimulationController();

	// Manage simulation methods
	void startSimulation();
	void stopSimulation();
	void toggleSimulation();
	void toggleWallPlacement();
	void resetSimulation();

	void initializeUI(int argc, char** argv);
	static void display(); // Function for OpenGL for drawing
	static void reshape(int w, int h); // Change size of window
	static void keyboard(unsigned char key, int x, int y); // Keyboard operation
	
	void drawButton(float x, float y, float width, float height, const char* label);
	static void mouseHandler(int button, int state, int x, int y);
	void motionHandler(int x, int y);
	static void staticMotionHandler(int x, int y);

	void updateSimulation(); // Update simulation state
};
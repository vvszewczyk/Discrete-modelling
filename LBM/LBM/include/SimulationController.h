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
	int mainWindowID;
	int uxWindowID;
	int uyWindowID;

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
	static const int minStepsPerFrame = 1;
	static const int maxStepsPerFrame = 30;
	int totalIterations;

	double tau; // Relaxation time

public:
	SimulationController(Grid* g, CudaHandler* ch, int width, int height, int cs);
	~SimulationController();

	// Manage simulation methods
	void startSimulation();
	void stopSimulation();
	void toggleSimulation();
	void toggleWallPlacement();
	void resetSimulation();

	void initializeUI(int argc, char** argv); // Initialise UI, set OpenGL and GLUT
	static void displayMain(); // Draw main window
	static void displayUx(); // Draw X velocioty window
	static void displayUy(); // Draw Y velocioty window
	static void reshape(int w, int h); // Change size of window
	static void keyboard(unsigned char key, int x, int y); // Keyboard input operation

	void drawButton(float x, float y, float width, float height, const char* label);
	static void drawString(float x, float y, const char* text, void* font);
	static void mouseHandler(int button, int state, int x, int y); // Button managment
	void motionHandler(int x, int y); // Mouse managment in case when LMB pressed
	static void staticMotionHandler(int x, int y); // Motion handler but static

	void updateSimulation(); // Update simulation state
};
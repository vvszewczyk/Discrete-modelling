#include "SimulationController.h"
#include <iostream>
#include <string>


// Static pointer for current controller (GLUT REQUIRES IT)
static SimulationController* controller = nullptr;

SimulationController::SimulationController(Grid* g, CudaHandler* ch, int width, int height, int cs) : grid(g), cudaHandler(ch), windowHeight(height), windowWidth(width), cellSize(cs), isRunning(false), placingWalls(false), stepsPerFrame(1), gridModified(false)
{
	controller = this;
	buttonLabelSS = "Start";
	buttonLabelWE = "Wall";
};

SimulationController::~SimulationController() {}

void SimulationController::startSimulation()
{
	isRunning = true;
}

void SimulationController::stopSimulation()
{
	isRunning = false;
}

void SimulationController::toggleSimulation()
{
	isRunning = !isRunning;
	buttonLabelSS = isRunning ? "Stop" : "Start";
	
	if (isRunning)
	{
		gridModified = true;
	}
}

void SimulationController::toggleWallPlacement()
{
	placingWalls = !placingWalls;
	buttonLabelWE = placingWalls ? "Empty" : "Wall";
}

void SimulationController::resetSimulation()
{
	isRunning = false;
	grid->initialize();
	buttonLabelSS = "Start";

	cudaHandler->initializeDeviceGrids(grid->getGridData());

	glutPostRedisplay();
}

void SimulationController::initializeUI(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("LGA");

	// Callbacks
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseHandler);
	glutMotionFunc(staticMotionHandler);
	glutIdleFunc([]()
		{
			if (controller->isRunning)
			{
				controller->updateSimulation();
			}
		});
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, windowWidth, 0, windowHeight);
}

void SimulationController::display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	int buttonViewportHeight = controller->windowHeight / 20;

	glViewport(0, buttonViewportHeight, controller->windowWidth, controller->windowHeight - buttonViewportHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, controller->grid->getWidth() * controller->cellSize, 0, controller->grid->getHeight() * controller->cellSize);

	int width = controller->grid->getWidth();
	int height = controller->grid->getHeight();

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const Cell& cell = controller->grid->getCell(x, y);

			if (cell.getWall()) // Wall
			{
				glColor3f(1.0f, 1.0f, 1.0f);
			}
			else if (cell.getDirection(0)) // North
			{
				glColor3f(1.0f, 0.0f, 0.0f);
			}
			else if (cell.getDirection(1)) // East
			{
				glColor3f(0.0f, 1.0f, 0.0f);
			}
			else if (cell.getDirection(2)) // South
			{
				glColor3f(0.0f, 0.0f, 1.0f);
			}
			else if (cell.getDirection(3)) // West
			{
				glColor3f(1.0f, 1.0f, 0.0f);
			}
			else
			{
				glColor3f(0.0f, 0.0f, 0.0f); // Empty
			}

			glRecti(
				x * controller->cellSize, 
				y * controller->cellSize,
				(x + 1) * controller->cellSize, 
				(y + 1) * controller->cellSize
			);
		}
	}
	// Drawing grid
	glColor3f(0.0f, 0.0f, 0.0f);
	glLineWidth(1.0f);

	// User interface (buttons)
	glViewport(0, 0, controller->windowWidth, buttonViewportHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, controller->windowWidth, 0, buttonViewportHeight);

	controller->drawButton(5, buttonViewportHeight / 2 - 15, 100, 30, controller->buttonLabelSS.c_str());
	controller->drawButton(120, buttonViewportHeight / 2 - 15, 100, 30, "Reset");
	controller->drawButton(235, buttonViewportHeight / 2 - 15, 100, 30, controller->buttonLabelWE.c_str());
	controller->drawButton(360, buttonViewportHeight / 2 - 15, 30, 30, "<");
	controller->drawButton(400, buttonViewportHeight / 2 - 15, 60, 30, std::to_string(controller->stepsPerFrame).c_str());
	controller->drawButton(470, buttonViewportHeight / 2 - 15, 30, 30, ">");

	glutSwapBuffers();
}

void SimulationController::reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, w, 0, h);
	glMatrixMode(GL_MODELVIEW);
}

void SimulationController::keyboard(unsigned char key, int x, int y)
{
	if (key == ' ')
	{
		controller->toggleSimulation();
	}
	else if (key == 27) // ESC
	{
		exit(0);
	}
}

void SimulationController::drawButton(float x, float y, float width, float height, const char* label)
{
	glColor3f(0.8f, 0.8f, 0.8f);
	glRectf(x, y, x + width, y + height);

	// Center string
	int textWidth = 0;
	for (const char* c = label; *c != '\0'; c++)
	{
		textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_12, *c);
	}

	float textX = x + (width - textWidth) / 2;
	float textY = y + (height - 12) / 2;

	glColor3f(0.0f, 0.0f, 0.0f);
	glRasterPos2f(textX, textY);
	for (const char* c = label; *c != '\0'; c++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c);
	}
}

void SimulationController::mouseHandler(int button, int state, int x, int y)
{
	y = controller->windowHeight - y;

	int buttonWidth = 100;
	int buttonHeight = 30;
	int buttonViewportHeight = controller->windowHeight / 20;
	int buttonY = buttonViewportHeight / 2 - buttonHeight / 2;

	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			controller->isLeftMouseDown = true;

			// Start/stop button
			if (x >= 5 && x <= 5 + buttonWidth && y >= buttonY && y <= buttonY + buttonHeight)
			{
				controller->toggleSimulation();
			}

			// Reset button
			if (x >= 120 && x <= 120 + buttonWidth && y >= buttonY && y <= buttonY + buttonHeight)
			{
				controller->resetSimulation();
			}

			// Wall placing button
			if (x >= 240 && x <= 240 + buttonWidth && y >= buttonY && y <= buttonY + buttonHeight)
			{
				controller->toggleWallPlacement();
			}

			// Slow down
			if (x >= 360 && x <= 390 && y >= buttonY && y <= buttonY + buttonHeight)
			{
				controller->stepsPerFrame =std::max(MIN_STEPS_PER_FRAME, controller->stepsPerFrame - 1);
				std::cout << "Decreased simulation speed: " << controller->stepsPerFrame << " steps/frame" << std::endl;
			}

			// Speed up
			if (x >= 470 && x <= 500 && y >= buttonY && y <= buttonY + buttonHeight)
			{
				controller->stepsPerFrame = std::min(MAX_STEPS_PER_FRAME, controller->stepsPerFrame + 1);
				std::cout << "Increased simulation speed: " << controller->stepsPerFrame << " steps/frame" << std::endl;
			}

		}
		else if (state == GLUT_UP)
		{
			controller->isLeftMouseDown = false;
		}
	}
}

void SimulationController::motionHandler(int x, int y)
{
	y = controller->windowHeight - y;
	int buttonViewportHeight = controller->windowHeight / 20;

	if (controller->isLeftMouseDown)
	{
		int gridX = x / controller->cellSize;
		int gridY = (y - buttonViewportHeight) / controller->cellSize;

		if (gridX >= 0 && gridX < controller->grid->getWidth() && gridY >= 0 && gridY < controller->grid->getHeight())
		{
			if (controller->placingWalls)
			{
				controller->grid->getCell(gridX, gridY).setWall(true);
			}
			else
			{
				controller->grid->getCell(gridX, gridY).setWall(false);
				controller->grid->getCell(gridX, gridY).resetDirections();
			}

			controller->gridModified = true;
			glutPostRedisplay();
		}
	}
}

void SimulationController::staticMotionHandler(int x, int y)
{
	if (controller != nullptr)
	{
		controller->motionHandler(x, y);
	}
}


void SimulationController::updateSimulation()
{
	if (gridModified)
	{
		cudaHandler->copyGridToGPU(grid->getGridData());
		gridModified = false;
	}

	for (int i = 0; i < stepsPerFrame; ++i)
	{
		cudaHandler->executeCollisionKernel();
		cudaHandler->executeStreamingKernel();
	}

	cudaHandler->copyGridToCPU(grid->getGridData());

	glutPostRedisplay();
}
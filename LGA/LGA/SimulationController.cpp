#include "SimulationController.h"

// Static pointer for current controller (GLUT REQUIRES IT)
static SimulationController* controller = nullptr;

SimulationController::SimulationController(Grid* g, CudaHandler* ch, int width, int height, int cs) : grid(g), cudaHandler(ch), windowHeight(height), windowWidth(width), cellSize(cs) 
{
	controller = this;
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
}

void SimulationController::initializeOpenGL(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("LGA");

	// Callbacks
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
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

	int width = controller->grid->getWidth();
	int height = controller->grid->getHeight();

	for (std::size_t y = 0; y < height; ++y)
	{
		for (std::size_t x = 0; x < width; ++x)
		{
			const Cell& cell = controller->grid->getCell(x, y);

			if (cell.getWall()) // Wall
			{
				glColor3f(0.2f, 0.2f, 0.2f);
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
				glColor3f(0.8f, 0.8f, 0.8f); // Empty
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

	glBegin(GL_LINES);
	// Vertical lines
	for (std::size_t x = 0; x <= width; ++x) 
	{
		int xPos = x * controller->cellSize;
		glVertex2i(xPos, 0);
		glVertex2i(xPos, height * controller->cellSize);
	}

	// Horizontal lines
	for (std::size_t y = 0; y <= height; ++y)
	{
		int yPos = y * controller->cellSize;
		glVertex2i(0, yPos);
		glVertex2i(width * controller->cellSize, yPos);
	}
	glEnd();
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

void SimulationController::updateSimulation()
{
	grid->collision();
	grid->streaming();

	//cudaHandler->executeCollisionKernel();
	//cudaHandler->executeStreamingKernel();
	//cudaHandler->copyGridToCPU(*grid);

	glutPostRedisplay();
}



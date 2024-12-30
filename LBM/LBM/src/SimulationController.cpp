#include "SimulationController.h"
#include <iostream>
#include <string>


// Static pointer for current controller (GLUT REQUIRES IT)
static SimulationController* controller = nullptr;

SimulationController::SimulationController(Grid* g, CudaHandler* ch, int width, int height, int cs) : grid(g), cudaHandler(ch), windowHeight(height), windowWidth(width), cellSize(cs), isRunning(false), placingWalls(false), stepsPerFrame(1), gridModified(false), tau(1.0), mainWindowID(0), uxWindowID(0), uyWindowID(0), totalIterations(0)
{
	controller = this;
	this->buttonLabelSS = "Start";
	this->buttonLabelWE = "Wall";
};

SimulationController::~SimulationController() {}

void SimulationController::startSimulation()
{
	this->isRunning = true;
}

void SimulationController::stopSimulation()
{
	this->isRunning = false;
}

void SimulationController::toggleSimulation()
{
	this->isRunning = !isRunning;
	this->buttonLabelSS = isRunning ? "Stop" : "Start";

	if (isRunning)
	{
		this->gridModified = true;
	}
}

void SimulationController::toggleWallPlacement()
{
	this->placingWalls = !placingWalls;
	this->buttonLabelWE = placingWalls ? "Empty" : "Wall";
}

void SimulationController::resetSimulation()
{
	this->isRunning = false;
	this->grid->initialize();
	this->buttonLabelSS = "Start";
	this->totalIterations = 0;

	this->cudaHandler->initializeDeviceGrids(this->grid->getGridData());

	glutSetWindow(this->mainWindowID);
	glutPostRedisplay();

	glutSetWindow(this->uxWindowID);
	glutPostRedisplay();

	glutSetWindow(this->uyWindowID);
	glutPostRedisplay();
}

void SimulationController::initializeUI(int argc, char** argv)
{
	glutInit(&argc, argv); // Initialise GLUT library
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(this->windowWidth, this->windowHeight);
	glutInitWindowPosition(100, 100);
	this->mainWindowID = glutCreateWindow("LBM - density");

	// Callbacks
	glutDisplayFunc(displayMain);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseHandler);
	glutMotionFunc(staticMotionHandler);
	glutIdleFunc([]() // Registers a function that is called when the application is "idle".
		{
			if (controller->isRunning)
			{
				controller->updateSimulation();
			}
		});
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, this->windowWidth, 0, this->windowHeight);

	glutInitWindowSize(this->windowWidth / 2 - 20, this->windowHeight / 2 - 20);
	glutInitWindowPosition(900, 100);
	this->uxWindowID = glutCreateWindow("LBM - uX");
	glutDisplayFunc(displayUx);

	glutInitWindowSize(this->windowWidth / 2 - 20, this->windowHeight / 2 - 20);
	glutInitWindowPosition(900, 520);
	this->uyWindowID = glutCreateWindow("LBM - uY");
	glutDisplayFunc(displayUy);

	glutSetWindow(this->mainWindowID);
}

void SimulationController::displayMain()
{
	glClear(GL_COLOR_BUFFER_BIT); // Clear last frame

	int buttonViewportHeight = controller->windowHeight / 20; // Space for buttons

	glViewport(0, buttonViewportHeight, controller->windowWidth, controller->windowHeight - buttonViewportHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, controller->grid->getWidth() * controller->cellSize, 0, controller->grid->getHeight() * controller->cellSize);

	// Size of gird
	int width = controller->grid->getWidth();
	int height = controller->grid->getHeight();

	// Set color of cell based on the state
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const Cell& cell = controller->grid->getCell(x, y);

			if (cell.getWall()) // Wall
			{
				glColor3f(0.0f, 1.0f, 0.0f);
			}
			else
			{
				double rho = cell.getRho();
				double val = rho;
				if (val > 1.0) val = 1.0;
				if (val < 0.0) val = 0.0;
				glColor3f(static_cast<GLfloat>(val), static_cast<GLfloat>(val), static_cast<GLfloat>(val));
			}

			glRecti(
				x * controller->cellSize,
				y * controller->cellSize,
				(x + 1) * controller->cellSize,
				(y + 1) * controller->cellSize
			);
		}
	}

	// User interface (buttons)
	glViewport(0, 0, controller->windowWidth, buttonViewportHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, controller->windowWidth, 0, buttonViewportHeight);

	// Buttons placement
	controller->drawButton(5, buttonViewportHeight / 2.f - 15.f, 100, 30, controller->buttonLabelSS.c_str());
	controller->drawButton(120, buttonViewportHeight / 2.f - 15.f, 100, 30, "Reset");
	controller->drawButton(235, buttonViewportHeight / 2.f - 15.f, 100, 30, controller->buttonLabelWE.c_str());
	controller->drawButton(360, buttonViewportHeight / 2.f - 15.f, 30, 30, "-");
	controller->drawButton(400, buttonViewportHeight / 2.f - 15.f, 60, 30, std::to_string(controller->stepsPerFrame).c_str());
	controller->drawButton(470, buttonViewportHeight / 2.f - 15.f, 30, 30, "+");
	controller->drawButton(530, buttonViewportHeight / 2.f - 15.f, 30, 30, "-");
	controller->drawButton(570, buttonViewportHeight / 2.f - 15.f, 60, 30, std::to_string(controller->tau).c_str());
	controller->drawButton(640, buttonViewportHeight / 2.f - 15.f, 30, 30, "+");

	// Iteration string

	glColor3f(1.0f, 1.0f, 1.0f);
	std::string iterationText = "Iterations: " + std::to_string(controller->totalIterations);

	// Rysuj tekst
	controller->drawButton(700 - 10, buttonViewportHeight / 2.f - 15.f, 100, 30, " ");
	controller->drawString(700, buttonViewportHeight / 2.f - 5.f, iterationText.c_str(), GLUT_BITMAP_HELVETICA_12);
	glutSwapBuffers(); // Double buffer for smooth rendering
}

void SimulationController::displayUx()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluOrtho2D(0, controller->grid->getWidth(), 0, controller->grid->getHeight());

	int width = controller->grid->getWidth();
	int height = controller->grid->getHeight();

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const Cell& cell = controller->grid->getCell(x, y);

			if (cell.getWall())
			{
				glColor3f(0.0f, 1.0f, 0.0f);
			}
			else
			{
				double ux = cell.getUx();
				double val = fabs(ux);
				if (val > 1.0) val = 1.0;

				// Negative = purple, positive = green
				if (ux > 0)
				{
					glColor3f(0.0, static_cast<GLfloat>(val), 0.0);
				}
				else
				{
					glColor3f(static_cast<GLfloat>(val), 0.0, static_cast<GLfloat>(val));
				}
			}

			glRecti(x, y, x + 1, y + 1);
		}
	}

	glutSwapBuffers();
}

void SimulationController::displayUy()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluOrtho2D(0, controller->grid->getWidth(), 0, controller->grid->getHeight());

	int width = controller->grid->getWidth();
	int height = controller->grid->getHeight();

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const Cell& cell = controller->grid->getCell(x, y);

			if (cell.getWall())
			{
				glColor3f(0.0f, 1.0f, 0.0f);
			}
			else
			{
				double uy = cell.getUy();
				double val = fabs(uy);
				if (val > 1.0) val = 1.0;

				// Negative = purple, positive = green
				if (uy > 0)
				{
					glColor3f(0.0, static_cast<GLfloat>(val), 0.0);
				}
				else
				{
					glColor3f(static_cast<GLfloat>(val), 0.0, static_cast<GLfloat>(val));
				}
			}

			glRecti(x, y, x + 1, y + 1);
		}
	}

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
	glColor3f(0.8f, 0.8f, 0.8f); // Button color
	glRectf(x, y, x + width, y + height); // Draw rectangle


	int textWidth = 0;

	for (const char* c = label; *c != '\0'; c++)
	{
		textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_12, *c); // glutBitmapWidth - returns the width of a single char in the selected font
	}
	// Center string coordinates
	float textX = x + (width - textWidth) / 2;
	float textY = y + (height - 12) / 2;

	glColor3f(0.0f, 0.0f, 0.0f); // Font color
	glRasterPos2f(textX, textY); // Set start position of drawing text

	for (const char* c = label; *c != '\0'; c++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *c); // Draw a single char in the current position of drawing coursor (glRasterPos2f)
	}
}

void SimulationController::drawString(float x, float y, const char* text, void* font = GLUT_BITMAP_HELVETICA_12)
{
	glRasterPos2f(x, y);
	for (const char* c = text; *c != '\0'; ++c)
	{
		glutBitmapCharacter(font, *c);
	}
}

void SimulationController::mouseHandler(int button, int state, int x, int y)
{
	y = controller->windowHeight - y; // Converting position (correct with OpenGL)

	// Button size
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
				controller->stepsPerFrame = std::max(minStepsPerFrame, controller->stepsPerFrame - 1);
				std::cout << "Decreased simulation speed: " << controller->stepsPerFrame << " steps/frame" << std::endl;
			}

			// Speed up
			if (x >= 470 && x <= 500 && y >= buttonY && y <= buttonY + buttonHeight)
			{
				controller->stepsPerFrame = std::min(maxStepsPerFrame, controller->stepsPerFrame + 1);
				std::cout << "Increased simulation speed: " << controller->stepsPerFrame << " steps/frame" << std::endl;
			}

			// Decrease tau
			if (x >= 530 && x <= 530 + 30 && y >= buttonY && y <= buttonY + buttonHeight)
			{
				
				controller->tau = std::max(1.0, controller->tau - 0.1);
				std::cout << "Decreased tau: " << controller->tau << std::endl;
			}

			// Increase tau
			if (x >= 640 && x <= 640 + 30 && y >= buttonY && y <= buttonY + buttonHeight)
			{
				controller->tau += 0.1;
				std::cout << "Increased tau: " << controller->tau << std::endl;
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
	y = controller->windowHeight - y; // Converting position (correct with OpenGL)
	int buttonViewportHeight = controller->windowHeight / 20; // Part of window for buttons

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
			}

			controller->gridModified = true;

			glutSetWindow(mainWindowID);
			glutPostRedisplay();

			glutSetWindow(uxWindowID);
			glutPostRedisplay();

			glutSetWindow(uyWindowID);
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
	if (this->gridModified)
	{
		this->cudaHandler->copyGridToGPU(grid->getGridData());
		this->gridModified = false;
	}

	for (int i = 0; i < stepsPerFrame; ++i)
	{
		// No need to copy grid to GPU, kernels work directly on iy
		this->cudaHandler->executeCollision(this->tau);
		this->cudaHandler->executeStreaming();
	}

	totalIterations += stepsPerFrame;

	this->cudaHandler->copyGridToCPU(grid->getGridData());
	// Odœwie¿ okno g³ówne:
	glutSetWindow(this->mainWindowID);
	glutPostRedisplay();

	// Odœwie¿ okno u_x:
	glutSetWindow(this->uxWindowID);
	glutPostRedisplay();

	// Odœwie¿ okno u_y:
	glutSetWindow(this->uyWindowID);
	glutPostRedisplay(); // Marks the current window as needing to be redisplayed
}
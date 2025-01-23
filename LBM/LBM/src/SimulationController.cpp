#include "SimulationController.h"
#include <iostream>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Static pointer for current controller (GLUT REQUIRES IT)
static SimulationController *controller = nullptr;
static const int cx_cpu[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int cy_cpu[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const double w_cpu[9] = {
    4.0 / 9.0,                                     // i=0
    1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, // i=1..4
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 // i=5..8
};

SimulationController::SimulationController(Grid *g, CudaHandler *ch, int width, int height, int cs)
    : grid(g), cudaHandler(ch), windowHeight(height), windowWidth(width), cellSize(cs), isRunning(false),
      placingWalls(false), stepsPerFrame(1), gridModified(false), tau(1.0), mainWindowID(0), uxWindowID(0),
      uyWindowID(0), totalIterations(0)
{
    controller = this;
    this->buttonLabelSS = "Start";
    this->buttonLabelWE = "Wall";
    this->buttonLabelVariants = "Variant 1";

    initializeParticles(15);
};

SimulationController::~SimulationController()
{
}

void SimulationController::startSimulation()
{
    this->isRunning = true;
}

void SimulationController::stopSimulation()
{
    this->isRunning = false;
}

void SimulationController::resumeSimulation()
{
    this->cudaHandler->copyGridToGPU(grid->getGridData());
    this->isRunning = true;
    std::cout << "Simulation resumed at iteration: " << totalIterations << std::endl;
}

void SimulationController::toggleSimulation()
{
    if (!isRunning && stateLoaded)
    {
        resumeSimulation();
        this->buttonLabelSS = "Stop";
    }
    else
    {
        this->isRunning = !isRunning;
        this->buttonLabelSS = isRunning ? "Stop" : "Start";
        if (isRunning)
        {
            this->cudaHandler->copyGridToGPU(grid->getGridData());
            this->gridModified = true;
        }
    }
}

void SimulationController::toggleWallPlacement()
{
    this->placingWalls = !placingWalls;
    this->buttonLabelWE = placingWalls ? "Empty" : "Wall";
}

void SimulationController::toggleVariant()
{
    this->variant = !variant;
    this->buttonLabelVariants = variant ? "Variant 2" : "Variant 1";
}

void SimulationController::resetSimulation()
{
    this->isRunning = false;
    this->grid->initialize();
    this->buttonLabelSS = "Start";
    this->buttonLabelWE = "Wall";
    this->buttonLabelVariants = "Variant 1";
    this->totalIterations = 0;

    this->cudaHandler->initializeDeviceGrids(this->grid->getGridData());

    particles.clear();

    initializeParticles(15);

    glutSetWindow(this->mainWindowID);
    glutPostRedisplay();

    glutSetWindow(this->uxWindowID);
    glutPostRedisplay();

    glutSetWindow(this->uyWindowID);
    glutPostRedisplay();
}

void SimulationController::initializeUI(int argc, char **argv)
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
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, controller->grid->getWidth() * controller->cellSize, 0,
               controller->grid->getHeight() * controller->cellSize);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Size of gird
    int width = controller->grid->getWidth();
    int height = controller->grid->getHeight();

    // Set color of cell based on the state
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const Cell &cell = controller->grid->getCell(x, y);

            if (cell.getWall()) // Wall
            {
                glColor3f(0.0f, 1.0f, 0.0f);
            }
            else
            {
                // Deviation from 1
                double rho = cell.getRho();
                double base_rho = 1.0;
                double scale = 50.0;

                double val = (rho - base_rho) * scale + 0.5;
                // val ~ 0.5, if rho == base_rho
                // val ~ 1.0, if rho == 1.01
                // val ~ 0.0, if rho == 0.99

                if (val < 0.0)
                    val = 0.0;
                if (val > 1.0)
                    val = 1.0;
                glColor3f(static_cast<GLfloat>(val), static_cast<GLfloat>(val), static_cast<GLfloat>(val));
            }

            glRecti(x * controller->cellSize, y * controller->cellSize, (x + 1) * controller->cellSize,
                    (y + 1) * controller->cellSize);
        }
    }

    // Particles trajectory
    glLineWidth(2.f);
    for (auto &p : controller->particles)
    {
        if (p.trace.size() < 2)
            continue;
        glColor3f(p.r, p.g, p.b);

        glBegin(GL_LINE_STRIP);
        for (auto &pt : p.trace)
        {
            float px = pt.first * controller->cellSize;
            float py = pt.second * controller->cellSize;
            glVertex2f(px, py);
        }
        glEnd();
    }

    glPopMatrix();

    // GUI drawing
    glViewport(0, 0, controller->windowWidth, buttonViewportHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, controller->windowWidth, 0, buttonViewportHeight);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Buttons placement
    controller->drawButton(5, buttonViewportHeight / 2.f - 15.f, 30, 30, controller->buttonLabelSS.c_str());
    controller->drawButton(40, buttonViewportHeight / 2.f - 15.f, 40, 30, "Reset");
    controller->drawButton(85, buttonViewportHeight / 2.f - 15.f, 40, 30, controller->buttonLabelWE.c_str());
    controller->drawButton(160, buttonViewportHeight / 2.f - 15.f, 40, 30, "Save");
    controller->drawButton(205, buttonViewportHeight / 2.f - 15.f, 40, 30, "Load");
    controller->drawButton(272, buttonViewportHeight / 2.f - 15.f, 60, 30, controller->buttonLabelVariants.c_str());
    controller->drawButton(360, buttonViewportHeight / 2.f - 15.f, 30, 30, "-");
    controller->drawButton(400, buttonViewportHeight / 2.f - 15.f, 60, 30,
                           std::to_string(controller->stepsPerFrame).c_str());
    controller->drawButton(470, buttonViewportHeight / 2.f - 15.f, 30, 30, "+");
    controller->drawButton(530, buttonViewportHeight / 2.f - 15.f, 30, 30, "-");
    controller->drawButton(570, buttonViewportHeight / 2.f - 15.f, 60, 30, std::to_string(controller->tau).c_str());
    controller->drawButton(640, buttonViewportHeight / 2.f - 15.f, 30, 30, "+");

    // Iteration string
    glColor3f(1.0f, 1.0f, 1.0f);
    std::string iterationText = "Iterations: " + std::to_string(controller->totalIterations);

    controller->drawButton(700 - 10, buttonViewportHeight / 2.f - 15.f, 100, 30, " ");
    controller->drawString(700, buttonViewportHeight / 2.f - 5.f, iterationText.c_str(), GLUT_BITMAP_HELVETICA_12);
    glutSwapBuffers(); // Double buffer for smooth rendering
}

void SimulationController::displayUx()
{
    displayVelocity(true);
}

void SimulationController::displayUy()
{
    displayVelocity(true);
}

void SimulationController::displayVelocity(bool isUx)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluOrtho2D(0, controller->grid->getWidth(), 0, controller->grid->getHeight());
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    int width = controller->grid->getWidth();
    int height = controller->grid->getHeight();

    // Background dependent on |ux| or |uy|
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const Cell &cell = controller->grid->getCell(x, y);

            if (cell.getWall())
            {
                glColor3f(0.0f, 1.0f, 0.0f);
            }
            else
            {
                double velocity = isUx ? cell.getUx() : cell.getUy();
                double val = std::fabs(velocity) * 50.0; // Strengthening color x50
                if (val > 1.0)
                {
                    val = 1.0;
                }

                // Positive -> green, Negative -> purple
                if (velocity > 0.0)
                {
                    glColor3f(0.0f, static_cast<GLfloat>(val), 0.0f);
                }
                else
                {
                    glColor3f(static_cast<GLfloat>(val), 0.0f, static_cast<GLfloat>(val));
                }
            }

            // 1x1 rectangle
            glRecti(x, y, x + 1, y + 1);
        }
    }

    // Hedgehog drawing
    glLineWidth(1.0f);
    glColor3f(1.0f, 1.0f, 1.0f);
    double vectorScale = 1000.0; // Vector length

    for (int j = 20; j < height; j += 20)
    {
        for (int i = 20; i < width; i += 20)
        {
            double vx = controller->grid->getCell(i, j).getUx();
            double vy = controller->grid->getCell(i, j).getUy();
            glBegin(GL_LINES);
            glVertex2f(static_cast<GLfloat>(i), static_cast<GLfloat>(j));
            glVertex2f(static_cast<GLfloat>(i + vectorScale * vx), static_cast<GLfloat>(j + vectorScale * vy));
            glEnd();
        }
    }

    // Streamlines
    glLineWidth(2.0f);

    int linesCount = 15; // Starting amount of lines on the left side
    for (int seed = 0; seed < linesCount; ++seed)
    {
        float xCur = 0.0f;
        float yCur = (float)seed * (height / (float)linesCount);

        float factor = 1.0f - float(seed) / (linesCount - 1);
        glColor3f(1.0f, factor, 0.0f);

        glBegin(GL_LINE_STRIP);
        for (int steps = 0; steps < 5000; steps++)
        {
            glVertex2f(xCur, yCur);
            int i = static_cast<int>(floor(xCur));
            int j = static_cast<int>(floor(yCur));

            if (i < 0 || i >= width || j < 0 || j >= height)
                break;

            double vx = controller->grid->getCell(i, j).getUx();
            double vy = controller->grid->getCell(i, j).getUy();

            // Euler
            float dt = 500.f; // Step
            xCur += dt * static_cast<float>(vx);
            yCur += dt * static_cast<float>(vy);

            double speed = std::fabs(vx) + std::fabs(vy);
            if (speed < 1e-16)
            {
                break;
            }
        }
        glEnd();
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

void SimulationController::drawButton(float x, float y, float width, float height, const char *label)
{
    glColor3f(0.8f, 0.8f, 0.8f);          // Button color
    glRectf(x, y, x + width, y + height); // Draw rectangle

    int textWidth = 0;

    for (const char *c = label; *c != '\0'; c++)
    {
        textWidth += glutBitmapWidth(GLUT_BITMAP_HELVETICA_12,
                                     *c); // glutBitmapWidth - returns the width of a single char in the selected font
    }
    // Center string coordinates
    float textX = x + (width - textWidth) / 2;
    float textY = y + (height - 12) / 2;

    glColor3f(0.0f, 0.0f, 0.0f); // Font color
    glRasterPos2f(textX, textY); // Set start position of drawing text

    for (const char *c = label; *c != '\0'; c++)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12,
                            *c); // Draw a single char in the current position of drawing coursor (glRasterPos2f)
    }
}

void SimulationController::drawString(float x, float y, const char *text, void *font = GLUT_BITMAP_HELVETICA_12)
{
    glRasterPos2f(x, y);
    for (const char *c = text; *c != '\0'; ++c)
    {
        glutBitmapCharacter(font, *c);
    }
}

void SimulationController::mouseHandler(int button, int state, int x, int y)
{
    y = controller->windowHeight - y; // Converting position (correct with OpenGL)

    // Button size
    int buttonWidth = 30;
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
            if (x >= 40 && x <= 40 + buttonWidth * (1.32) && y >= buttonY && y <= buttonY + buttonHeight)
            {
                controller->resetSimulation();
            }

            // Wall/empty button
            if (x >= 85 && x <= 85 + buttonWidth * (1.32) && y >= buttonY && y <= buttonY + buttonHeight)
            {
                controller->toggleWallPlacement();
            }

            // Save button
            if (x >= 160 && x <= 160 + buttonWidth * (1.32) && y >= buttonY && y <= buttonY + buttonHeight)
            {
                controller->saveCurrentFrameBMP("output/savedFrame.bmp", controller->windowWidth,
                                                controller->windowHeight);
                std::cout << "Frame saved as savedFrame.bmp" << std::endl;
                controller->saveSimulationStateCSV("output/savedFrame.csv", controller->grid);
                std::cout << "Frame saved as savedFrame.csv" << std::endl;
            }

            // Load button
            if (x >= 205 && x <= 205 + buttonWidth * (1.32) && y >= buttonY && y <= buttonY + buttonHeight)
            {
                controller->loadSimulationStateCSV("output/savedFrame.csv", controller->grid);
                std::cout << "Frame loaded from savedFrame.csv" << std::endl;
            }

            // Variants button
            if (x >= 272 && x <= 272 + buttonWidth * 2 && y >= buttonY && y <= buttonY + buttonHeight)
            {
                controller->toggleVariant();
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
    y = controller->windowHeight - y;                         // Converting position (correct with OpenGL)
    int buttonViewportHeight = controller->windowHeight / 20; // Part of window for buttons

    if (controller->isLeftMouseDown)
    {
        int gridX = x / controller->cellSize;
        int gridY = (y - buttonViewportHeight) / controller->cellSize;

        // Wall/empty boldness (3x3)
        if (gridX >= 0 && gridX < controller->grid->getWidth() && gridY >= 0 && gridY < controller->grid->getHeight())
        {
            if (controller->placingWalls)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        int newX = gridX + dx;
                        int newY = gridY + dy;

                        if (newX >= 0 && newX < controller->grid->getWidth() && newY >= 0 &&
                            newY < controller->grid->getHeight())
                        {
                            controller->grid->getCell(newX, newY).setWall(true);
                        }
                    }
                }
            }
            else
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        int newX = gridX + dx;
                        int newY = gridY + dy;

                        if (newX >= 0 && newX < controller->grid->getWidth() && newY >= 0 &&
                            newY < controller->grid->getHeight())
                        {
                            controller->grid->getCell(newX, newY).setWall(false);
                        }
                    }
                }
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

void SimulationController::initializeParticles(int numParticles)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    particles.clear(); // Delete old particles

    for (int i = 0; i < numParticles; ++i)
    {
        Particle p;
        p.x = 0.0f;
        p.y = grid->getHeight() / 2.0f + static_cast<float>(std::rand()) / RAND_MAX * (grid->getHeight() / 2.0f);

        p.m = 0.1f + 0.5f * (std::rand() / (float)RAND_MAX);

        p.grav = 0.002f + 0.003f * (std::rand() / (float)RAND_MAX);

        // Particle color
        p.r = static_cast<float>(std::rand()) / RAND_MAX;
        p.g = static_cast<float>(std::rand()) / RAND_MAX;
        p.b = static_cast<float>(std::rand()) / RAND_MAX;

        p.vx = 0.0f;
        p.vy = 0.0f;
        p.active = true;

        // Add first tracking point
        p.addTrace();

        particles.push_back(p);
    }
}

void SimulationController::updateParticles()
{
    float dt = 100.0f; // Speed up particles

    for (auto &p : particles)
    {
        if (!p.active)
        {
            continue;
        }

        int i = static_cast<int>(floor(p.x));
        int j = static_cast<int>(floor(p.y));

        if (i < 0 || i >= grid->getWidth() || j < 0 || j >= grid->getHeight())
        {
            p.active = false;
            continue;
        }

        // Get velocity
        double uxFluid = grid->getCell(i, j).getUx();
        double uyFluid = grid->getCell(i, j).getUy();

        // Update velocity
        float vxNew = p.m * p.vx + (1.f - p.m) * static_cast<float>(uxFluid);
        float vyNew = p.m * p.vy + (1.f - p.m) * static_cast<float>(uyFluid) - p.grav;

        // New coordinates
        float xNew = p.x + (p.vx + vxNew) * 0.5f * dt;
        float yNew = p.y + (p.vy + vyNew) * 0.5f * dt;

        // Save new data
        p.vx = vxNew;
        p.vy = vyNew;
        p.x = xNew;
        p.y = yNew;

        if (p.active)
        {
            p.addTrace(); // Add to the history
        }
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

        boundaryWrapper(this->cudaHandler->getGridInputPtr(), this->grid->getWidth(), this->grid->getHeight(),
                        this->buttonLabelVariants);

        this->cudaHandler->executeStreaming();

        updateParticles();
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

void SimulationController::copyWindowPixelsToBigBuffer(unsigned char *srcPixels, int srcWidth, int srcHeight,
                                                       unsigned char *dstPixels, int dstTotalWidth, int dstTotalHeight,
                                                       int offsetX)
{
    // Copying row by row
    for (int row = 0; row < srcHeight; row++)
    {
        // row in src: 0 = bottom edge
        int dstY = row;
        int srcRowStart = row * srcWidth * 3;
        int dstRowStart = (dstY * dstTotalWidth + offsetX) * 3;

        // Copy srcWidth * 3 bytes
        memcpy(dstPixels + dstRowStart, srcPixels + srcRowStart, srcWidth * 3);
    }
}

void SimulationController::saveCurrentFrameBMP(const char *filename, int width, int height)
{
    // main
    glutSetWindow(this->mainWindowID);
    int wMain = this->windowWidth;  // 800
    int hMain = this->windowHeight; // 800
    unsigned char *mainPixels = new unsigned char[wMain * hMain * 3];
    glReadPixels(0, 0, wMain, hMain, GL_RGB, GL_UNSIGNED_BYTE, mainPixels);

    // uX
    glutSetWindow(this->uxWindowID);
    int wUx = (this->windowWidth / 2) - 20;  // ~380
    int hUx = (this->windowHeight / 2) - 20; // ~380
    unsigned char *uxPixels = new unsigned char[wUx * hUx * 3];
    glReadPixels(0, 0, wUx, hUx, GL_RGB, GL_UNSIGNED_BYTE, uxPixels);

    // uY
    glutSetWindow(this->uyWindowID);
    int wUy = (this->windowWidth / 2) - 20;  // ~380
    int hUy = (this->windowHeight / 2) - 20; // ~380
    unsigned char *uyPixels = new unsigned char[wUy * hUy * 3];
    glReadPixels(0, 0, wUy, hUy, GL_RGB, GL_UNSIGNED_BYTE, uyPixels);

    // Big bufor for final output
    int outWidth = wMain + wUx;                 // 800 + 380 = 1180
    int outHeight = std::max(hMain, hUx + hUy); // max(800, 760) = 800

    unsigned char *bigPixels = new unsigned char[outWidth * outHeight * 3];
    std::memset(bigPixels, 0, outWidth * outHeight * 3);

    // Auxiliary function for copying one window to a selected location
    auto copyWindowPixels = [&](unsigned char *src, int srcW, int srcH, unsigned char *dst, int dstW, int dstH,
                                int offsetX, int offsetY) {
        for (int row = 0; row < srcH; ++row)
        {
            int dstY = offsetY + row;
            if (dstY >= 0 && dstY < dstH)
            {
                int srcPos = row * srcW * 3;
                int dstPos = (dstY * dstW + offsetX) * 3;
                std::memcpy(dst + dstPos, src + srcPos, srcW * 3);
            }
        }
    };

    // Copying:
    // main to the left side, from (0,0) to (wMain-1, hMain-1)
    copyWindowPixels(mainPixels, wMain, hMain, bigPixels, outWidth, outHeight, 0, 0);

    // uX to the right-top part
    int offsetX_ux = wMain;           // 800
    int offsetY_ux = outHeight - hUx; // 800 - 380 = 420
    copyWindowPixels(uxPixels, wUx, hUx, bigPixels, outWidth, outHeight, offsetX_ux, offsetY_ux);

    // uY to the right-bottom part
    int offsetX_uy = wMain; // 800
    int offsetY_uy = 0;     // bottom
    copyWindowPixels(uyPixels, wUy, hUy, bigPixels, outWidth, outHeight, offsetX_uy, offsetY_uy);

    // Swap bigPixels vertically
    unsigned char *flipped = new unsigned char[outWidth * outHeight * 3];
    int rowSize = outWidth * 3;
    for (int row = 0; row < outHeight; ++row)
    {
        std::memcpy(flipped + row * rowSize, bigPixels + (outHeight - 1 - row) * rowSize, rowSize);
    }

    stbi_write_bmp(filename, outWidth, outHeight, 3, flipped);

    delete[] mainPixels;
    delete[] uxPixels;
    delete[] uyPixels;
    delete[] bigPixels;
    delete[] flipped;

    std::cout << "Saved 3 windows in one BMP: " << filename << std::endl;
}

void SimulationController::saveSimulationStateCSV(const char *filename, const Grid *grid)
{
    int width = grid->getWidth();
    int height = grid->getHeight();

    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    // Header
    outFile << "Iteration,x,y,isWall,"
            << "rho,ux,uy,"
            << "f0,f1,f2,f3,f4,f5,f6,f7,f8\n";

    // Save state of every cell
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            const Cell &cell = grid->getCell(i, j);
            outFile << controller->totalIterations << "," << i << "," << j << "," << (cell.getWall() ? 1 : 0) << ","
                    << cell.getRho() << "," << cell.getUx() << "," << cell.getUy() << ",";

            for (int dir = 0; dir < 9; dir++)
            {
                outFile << cell.getF_in(dir);
                if (dir < 8)
                    outFile << ",";
            }
            outFile << "\n";
        }
    }

    outFile.close();
    std::cout << "Simulation state saved to: " << filename << std::endl;
}

void SimulationController::loadSimulationStateCSV(const char *filename, Grid *grid)
{
    int width = grid->getWidth();
    int height = grid->getHeight();

    std::ifstream inFile(filename);
    if (!inFile.is_open())
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    std::string line;
    // Skip header
    std::getline(inFile, line);

    int maxIter = controller->totalIterations;

    while (std::getline(inFile, line))
    {
        std::istringstream ss(line);
        std::string token;
        int iter, x, y;
        double rho, ux, uy;
        bool isWall;

        std::getline(ss, token, ',');
        iter = std::stoi(token);
        if (iter > maxIter)
        {
            maxIter = iter;
        }

        std::getline(ss, token, ',');
        x = std::stoi(token);
        std::getline(ss, token, ',');
        y = std::stoi(token);

        std::getline(ss, token, ',');
        isWall = (std::stoi(token) == 1);

        std::getline(ss, token, ',');
        rho = std::stod(token);

        std::getline(ss, token, ',');
        ux = std::stod(token);

        std::getline(ss, token, ',');
        uy = std::stod(token);

        // f_in:
        double fi[9];
        for (int dir = 0; dir < 9; ++dir)
        {
            std::getline(ss, token, (dir < 8 ? ',' : '\n'));
            fi[dir] = std::stod(token);
        }

        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            Cell &cell = grid->getCell(x, y);

            cell.setWall(isWall);
            cell.setRho(rho);
            cell.setUx(ux);
            cell.setUy(uy);

            for (int dir = 0; dir < 9; ++dir)
            {
                cell.setF_in(dir, fi[dir]);
                cell.setF_out(dir, fi[dir]);
                cell.setF_eq(dir, fi[dir]);
            }
        }
    }

    inFile.close();

    this->totalIterations = maxIter;

    cudaHandler->copyGridToGPU(grid->getGridData());
    std::cout << "Simulation state loaded from: " << filename << std::endl;
    controller->stateLoaded = true;
}

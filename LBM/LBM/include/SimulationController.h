#pragma once

#include "CudaHandler.h"
#include "Grid.h"
#include "Particle.h"
#include <GL/freeglut.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class SimulationController
{
  private:
    Grid *grid;               // Pointer on grid
    CudaHandler *cudaHandler; // Pointer on CUDA handler
    bool isRunning;           // Simulation flag
    int mainWindowID;
    int uxWindowID;
    int uyWindowID;
    int windowWidth;
    int windowHeight;
    int cellSize;                    // Size of single cell in pixels
    std::string buttonLabelSS;       // Button label for stop/start
    std::string buttonLabelVariants; // Button label for variants
    bool placingWalls = false;
    bool variant = false;
    bool isLeftMouseDown = false;
    bool stateLoaded = false;
    std::string buttonLabelWE; // Button label for wall/empty
    int stepsPerFrame;         // For managing simulation speed
    bool gridModified;
    static const int minStepsPerFrame = 1;
    static const int maxStepsPerFrame = 30;
    int totalIterations;
    double tau;                      // Relaxation time
    std::vector<Particle> particles; // Placeholder for particles
    void updateParticles();          // Particles movement logic
    void initializeParticles(int numParticles);

  public:
    SimulationController(Grid *g, CudaHandler *ch, int width, int height, int cs);
    ~SimulationController();

    // Manage simulation methods
    void startSimulation();
    void stopSimulation();
    void resumeSimulation();
    void toggleSimulation();
    void toggleWallPlacement();
    void toggleVariant();
    void resetSimulation();

    void initializeUI(int argc, char **argv); // Initialise UI, set OpenGL and GLUT
    static void displayMain();                // Draw main window
    static void displayUx();                  // Draw X velocioty window
    static void displayUy();                  // Draw Y velocioty window
    static void displayVelocity(bool isUx);
    static void reshape(int w, int h);                     // Change size of window
    static void keyboard(unsigned char key, int x, int y); // Keyboard input operation

    void drawButton(float x, float y, float width, float height, const char *label);
    static void drawString(float x, float y, const char *text, void *font);
    static void mouseHandler(int button, int state, int x, int y); // Button managment
    void motionHandler(int x, int y);                              // Mouse managment in case when LMB pressed
    static void staticMotionHandler(int x, int y);                 // Motion handler but static

    void updateSimulation(); // Update simulation state
    void saveCurrentFrameBMP(const char *filename, int width, int height);
    void saveSimulationStateCSV(const char *filename, const Grid *grid);
    void loadSimulationStateCSV(const char *filename, Grid *grid);
    void copyWindowPixelsToBigBuffer(unsigned char *srcPixels, int srcWidth, int srcHeight, unsigned char *dstPixels,
                                     int dstTotalWidth, int dstTotalHeight,
                                     int offsetX); // For saving 3 windows in one BMP
};

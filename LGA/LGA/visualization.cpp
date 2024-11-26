#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>

// Deklaracja funkcji CUDA
void runCudaCalculations();

// Wymiary okna
const int windowWidth = 800;
const int windowHeight = 600;

// Wsp�rz�dne i wymiary przycisku
const int buttonX = 50;
const int buttonY = 50;
const int buttonWidth = 150;
const int buttonHeight = 50;

// Flaga wskazuj�ca, czy przycisk zosta� klikni�ty
bool buttonClicked = false;

// Funkcja konfiguruj�ca rzutowanie ortograficzne
void setup2DView(int width, int height) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, width, 0, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Funkcja do rysowania prostok�ta (np. przycisku)
void drawRectangle(int x, int y, int width, int height, float r, float g, float b) {
    glColor3f(r, g, b);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();
}

// Funkcja renderuj�ca scen�
void renderScene()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Ustawienia rzutowania 2D
    setup2DView(windowWidth, windowHeight);

    // Rysowanie przycisku
    if (buttonClicked) {
        // Przyci�ni�ty przycisk
        drawRectangle(buttonX, buttonY, buttonWidth, buttonHeight, 0.0f, 1.0f, 0.0f);
    }
    else {
        // Normalny przycisk
        drawRectangle(buttonX, buttonY, buttonWidth, buttonHeight, 1.0f, 0.0f, 0.0f);
    }

    glutSwapBuffers();
}

// Funkcja obs�uguj�ca klikni�cia myszy
void mouseClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        // Zamiana wsp�rz�dnych okna na wsp�rz�dne OpenGL (o� Y odwr�cona)
        int glY = windowHeight - y;

        // Sprawdzenie, czy klikni�to w obszar przycisku
        if (x >= buttonX && x <= buttonX + buttonWidth &&
            glY >= buttonY && glY <= buttonY + buttonHeight) {
            buttonClicked = !buttonClicked; // Prze��cz stan przycisku
            printf("Przycisk klikni�ty!\n");

            // Dla testu: Wywo�aj obliczenia CUDA
            runCudaCalculations();
        }
    }

    glutPostRedisplay(); // Od�wie� okno
}

int main(int argc, char** argv)
{
    // Inicjalizacja FreeGLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("CUDA + OpenGL + Button");

    // Inicjalizacja GLEW
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK)
    {
        fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(glew_status));
        return 1;
    }

    // Inicjalizacja CUDA
    cudaSetDevice(0);

    // Ustawienie funkcji renderuj�cej i obs�ugi myszy
    glutDisplayFunc(renderScene);
    glutMouseFunc(mouseClick);

    // Rozpocz�cie p�tli GLUT
    glutMainLoop();

    return 0;
}
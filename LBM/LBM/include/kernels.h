#pragma once

#include "Cell.h"

void collision(Cell* d_grid, int width, int height, double tau);
void streaming(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);

// Wrapper functions to call from host code
void collisionWrapper(Cell* d_grid, int width, int height, double tau);
void streamingWrapper(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);
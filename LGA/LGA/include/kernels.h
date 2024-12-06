#pragma once

#include "Cell.h"

// Wrapper functions to call from host code
void collisionWrapper(Cell* d_grid, int width, int height);
void streamingWrapper(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);
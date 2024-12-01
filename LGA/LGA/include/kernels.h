#pragma once

#include "Cell.h"

void collisionKernelWrapper(Cell* d_grid, int width, int height);
void streamingKernelWrapper(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);
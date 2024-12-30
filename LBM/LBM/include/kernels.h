#pragma once

#include "Cell.h"

void collision(Cell* d_grid, int width, int height, double tau);
void streaming(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);

// i:    0, 1, 2, 3, 4, 5, 6, 7, 8
// opp:  0, 2, 1, 4, 3, 7, 8, 5, 6
int findOpposite(int i); // Find opposite direction based on i 

// Wrapper functions to call from host code
void collisionWrapper(Cell* d_grid, int width, int height, double tau);
void streamingWrapper(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);
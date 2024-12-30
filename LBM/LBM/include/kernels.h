#pragma once

#include "Cell.h"

const int Q9 = 9; // Number of directions D2Q9

// Weights D2Q9
__device__ const double w[Q9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
__device__ const int cx[Q9] = { 0, 1, -1, 0, 0, 1, -1, -1, 1 }; // X directions
__device__ const int cy[Q9] = { 0, 0,  0, 1, -1, 1,  1, -1, -1 }; // Y directions

__global__ void collision(Cell* d_grid, int width, int height, double tau);
__global__ void streaming(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);

__device__ __host__ int findOpposite(int i); // Find opposite direction based on i		// i:    0, 1, 2, 3, 4, 5, 6, 7, 8
																						// opp:  0, 2, 1, 4, 3, 7, 8, 5, 6

// Wrapper functions to call from host code
void collisionWrapper(Cell* d_grid, int width, int height, double tau);
void streamingWrapper(Cell* d_gridInput, Cell* d_gridOutput, int width, int height);
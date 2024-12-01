#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Cell
{
private:
    bool isWall;
    bool direction[4]; // Directions: [N, E, S, W]

public:
    __host__ __device__ Cell(bool w = false) : isWall(w)
    {
        for (int i = 0; i < 4; ++i)
            direction[i] = false;
    }

    __host__ __device__ inline void setDirection(int dir, bool value)
    {
        if (dir >= 0 && dir < 4)
        {
            direction[dir] = value;
        }
    }

    __host__ __device__ inline bool getDirection(int dir) const
    {
        if (dir >= 0 && dir < 4)
        {
            return direction[dir];
        }
        return false;
    }

    __host__ __device__ inline void setWall(bool value)
    {
        isWall = value;
    }

    __host__ __device__ inline bool getWall() const
    {
        return isWall;
    }

    __host__ __device__ inline void resetDirections()
    {
        for (int i = 0; i < 4; ++i)
            direction[i] = false;
    }
};

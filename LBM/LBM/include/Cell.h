#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Cell
{
private:
    bool isWall;
    double f_in[5];
    double f_out[5];
    double f_eq[5];
    double C; // Concentration

public:
    __host__ __device__ Cell(bool w = false) : isWall(w), C(0.0)
    {
        for (int i = 0; i < 5; ++i)
        {
            f_in[i] = 0.0;
            f_out[i] = 0.0;
            f_eq[i] = 0.0;
        }
    }

    __host__ __device__ inline void setWall(bool value)
    {
        isWall = value;
    }

    __host__ __device__ inline bool getWall() const
    {
        return isWall;
    }

    __host__ __device__ inline void setF_in(int i, double val)
    {
        f_in[i] = val;
    }

    __host__ __device__ inline double getF_in(int i) const
    {
        return f_in[i];
    }

    __host__ __device__ inline void setF_out(int i, double val)
    {
        f_out[i] = val;
    }

    __host__ __device__ inline double getF_out(int i) const
    {
        return f_out[i];
    }

    __host__ __device__ inline void setF_eq(int i, double val)
    {
        f_eq[i] = val;
    }

    __host__ __device__ inline double getF_eq(int i) const
    {
        return f_eq[i];
    }

    __host__ __device__ inline void setC(double val)
    {
        C = val;
    }

    __host__ __device__ inline double getC() const
    {
        return C;
    }

    __host__ __device__ inline void resetCell()
    {
        isWall = false;
        C = 0.0;
        for (int i = 0; i < 5; ++i)
        {
            f_in[i] = 0.0;
            f_out[i] = 0.0;
            f_eq[i] = 0.0;
        }
    }
};

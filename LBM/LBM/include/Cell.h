#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Cell
{
private:
    bool isWall;
    double f_in[4]; // Input values ??of the distribution function for 4 directions
    double f_out[4]; // Output values ??of the distribution function for 4 directions
    double f_eq[4]; // Values ??of the equilibrium distribution function for 4 directions (tr: equilibrium distribution - rozklad rownowagowy)
    double C; // Concentration

public:
    __host__ __device__ Cell(bool w = false) : isWall(w), C(0.0)
    {
        for (int i = 0; i < 4; ++i)
        {
            f_in[i] = 0.0;
            f_out[i] = 0.0;
            f_eq[i] = 0.0;
        }
    }

    __host__ __device__ void setWall(bool value)
    {
        isWall = value;
    }

    __host__ __device__ bool getWall() const
    {
        return isWall;
    }

    __host__ __device__ void setF_in(int i, double val) 
    { 
        f_in[i] = val; 
    }

    __host__ __device__ double getF_in(int i) const 
    { 
        return f_in[i]; 
    }

    __host__ __device__ void setF_out(int i, double val) 
    { 
        f_out[i] = val; 
    }

    __host__ __device__ double getF_out(int i) const 
    { 
        return f_out[i]; 
    }

    __host__ __device__ void setF_eq(int i, double val) 
    {
        f_eq[i] = val; 
    }

    __host__ __device__ double getF_eq(int i) const 
    { 
        return f_eq[i]; 
    }

    __host__ __device__ void setC(double val) 
    { 
        C = val; 
    }

    __host__ __device__ double getC() const 
    { 
        return C; 
    }

    __host__ __device__ void resetCell()
    {
        isWall = false;
        C = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            f_in[i] = 0.0;
            f_out[i] = 0.0;
            f_eq[i] = 0.0;
        }
    }
};

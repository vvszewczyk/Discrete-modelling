#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class Cell
{
private:
    bool isWall;
    double f_in[9]; // Input values ​​of the distribution function for 4 directions
    double f_out[9]; // Output values ​​of the distribution function for 4 directions
    double f_eq[9]; // Values ​​of the equilibrium distribution function for 4 directions (tr: equilibrium distribution - rozklad rownowagowy)
    double rho; // Density, for difusion it was C - concentration
    double ux; // Velocity x
    double uy; // Velocity y
public:
    __host__ __device__ Cell(bool w = false) : isWall(w), rho(0.0), ux(0.0), uy(0.0)
    {
        for (int i = 0; i < 9; ++i)
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

    __host__ __device__ void setRho(double val) 
    { 
        rho = val; 
    }

    __host__ __device__ double getRho() const 
    { 
        return rho; 
    }

    __host__ __device__ void setUx(double val)
    {
        ux = val;
    }

    __host__ __device__ double getUx() const 
    {
        return ux; 
    }

    __host__ __device__ void setUy(double val) 
    { 
        uy = val; 
    }

    __host__ __device__ double getUy() const 
    { 
        return uy; 
    }

    __host__ __device__ void resetCell()
    {
        isWall = false;
        rho = 0.0;
        ux = 0.0;
        uy = 0.0;

        for (int i = 0; i < 9; ++i)
        {
            f_in[i] = 0.0;
            f_out[i] = 0.0;
            f_eq[i] = 0.0;
        }
    }
};

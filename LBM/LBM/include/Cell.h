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
            this->f_in[i] = 0.0;
            this->f_out[i] = 0.0;
            this->f_eq[i] = 0.0;
        }
    }

    __host__ __device__ void setWall(bool value)
    {
        this->isWall = value;
    }

    __host__ __device__ bool getWall() const
    {
        return this->isWall;
    }

    __host__ __device__ void setF_in(int i, double val) 
    { 
        this->f_in[i] = val;
    }

    __host__ __device__ double getF_in(int i) const 
    { 
        return this->f_in[i];
    }

    __host__ __device__ void setF_out(int i, double val) 
    { 
        this->f_out[i] = val;
    }

    __host__ __device__ double getF_out(int i) const 
    { 
        return this->f_out[i];
    }

    __host__ __device__ void setF_eq(int i, double val) 
    {
        this->f_eq[i] = val;
    }

    __host__ __device__ double getF_eq(int i) const 
    { 
        return this->f_eq[i];
    }

    __host__ __device__ void setRho(double val) 
    { 
        this->rho = val;
    }

    __host__ __device__ double getRho() const 
    { 
        return this->rho;
    }

    __host__ __device__ void setUx(double val)
    {
        this->ux = val;
    }

    __host__ __device__ double getUx() const 
    {
        return this->ux;
    }

    __host__ __device__ void setUy(double val) 
    { 
        this->uy = val;
    }

    __host__ __device__ double getUy() const 
    { 
        return this->uy;
    }

    __host__ __device__ void resetCell()
    {
        this->isWall = false;
        this->rho = 0.0;
        this->ux = 0.0;
        this->uy = 0.0;

        for (int i = 0; i < 9; ++i)
        {
            this->f_in[i] = 0.0;
            this->f_out[i] = 0.0;
            this->f_eq[i] = 0.0;
        }
    }
};

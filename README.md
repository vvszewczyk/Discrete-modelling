# Discrete Modeling - AGH University

## About the Course  
This repository contains projects developed as part of the **Discrete Modeling** course at **AGH University of Science and Technology**. The course introduces students to computational modeling techniques used for simulating various physical and engineering phenomena.  

The main focus is on **cellular automata**, **lattice gas models**, and **lattice Boltzmann methods (LBM)**. These techniques are widely used in **material science, fluid dynamics, and complex system simulations**.  

## Projects Overview  

Each project is implemented using **Python** or **C++ with CUDA** to leverage GPU acceleration for high-performance simulations.

### 🔹 Image Processing
- **Binarization of an Image** *(Python - `binarization/`)*  
  Converts an image to a binary format using thresholding techniques.  
  ![Binarization Example](media/binarization.gif)  

### 🔹 Cellular Automata (CA)
- **1D Cellular Automata** *(Python - `contextual and morphological transformations/`)*  
  Generates patterns based on simple one-dimensional automata rules.  
  ![1D CA Example](media/1d_ca.png)  
- **2D Cellular Automata - Conway's Game of Life** *(Python - `CA - Conway's Game of Life/`)*  
  A simulation of John Conway’s famous Game of Life.  
  ![Game of Life](media/RANDOM_1.gif)
  ![Game of Life 2](media/RANDOM_2.gif)
  ![Game of Life 3](media/GLIDER_GUN_REFLECTING_5000.gif) 
- **2D Cellular Automata - Forest Fire Simulation** *(Python - `forest fire/`)*  
  Models the spread of fire in a forest using probabilistic rules.  
  ![Forest Fire Simulation](media/forest.gif)  

### 🔹 Lattice Gas and Lattice Boltzmann Methods
- **Lattice Gas Automata (LGA) Simulation** *(C++ with CUDA - `LGA/`)*  
  Simulates gas flow using cellular automata principles.  
  ![LGA Simulation](media/LGA.gif)  
- **Diffusion Simulation with LBM** *(C++ with CUDA - `LBM/`)*  
  Models diffusion using the Lattice Boltzmann Method.  
  ![LBM Diffusion](media/LBM_diffusion.gif)  
- **Fluid Flow Simulation with LBM** *(C++ with CUDA - `LBM/`)*  
  Simulates fluid flow, capturing complex behaviors like turbulence.  
  ![LBM Flow](media/lbm_fluid_flow.gif)  
- **Boundary Conditions for LBM Simulations** *(C++ with CUDA - `LBM/`)*  
  Implements various boundary conditions for better accuracy.  
  ![LBM Boundary Conditions](media/lbm_bc.gif)  
- **Advanced LBM Flow Simulation with Visualization** *(C++ with CUDA - `LBM/`)*  
  Enhances LBM simulations with additional particles, gravity effects, and data export options (CSV, BMP).  
  ![Advanced LBM](media/particles.gif)  

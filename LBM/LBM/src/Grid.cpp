#include "Grid.h"
#include <iostream>
#include <string>

Grid::Grid(int w, int h) : width(w), height(h)
{
    this->grid = new Cell[this->width * this->height];
}

Grid::~Grid()
{
    delete[] this->grid;
}

Cell *Grid::getGridData()
{
    return this->grid;
}

const Cell *Grid::getGridData() const
{
    return this->grid;
}

int Grid::getWidth() const
{
    return this->width;
}

int Grid::getHeight() const
{
    return this->height;
}

Cell &Grid::getCell(int x, int y)
{
    return this->grid[y * this->width + x];
}

const Cell &Grid::getCell(int x, int y) const
{
    return this->grid[y * this->width + x];
}

void Grid::setCell(int x, int y, const Cell &cell)
{
    this->grid[y * this->width + x] = cell;
}

void Grid::initialize(bool defaultWall)
{
    resetGrid(); // Clear new walls

    for (int y = 0; y < this->height; ++y)
    {
        for (int x = 0; x < this->width; ++x)
        {
            Cell &cell = getCell(x, y);
            double rhoInit = 1.0;
            cell.setRho(rhoInit);
            cell.setUx(0.0);
            cell.setUy(0.0);

            for (int i = 0; i < Q9; ++i)
            {
                double feq = w[i] * rhoInit; // u=0 -> (c_i � u)=0, => f_i^{eq} = w_i * rho
                cell.setF_in(i, feq);
                cell.setF_eq(i, feq);
                cell.setF_out(i, feq);
            }

            // Set the wall on the middle
            int wallColumn = this->width / 2;
            int gapStart = this->height / 2;
            int gapEnd = 5 * this->height / 3;

            if (x == wallColumn && (y < gapStart || y > gapEnd))
            {
                cell.setWall(true);
                cell.setUx(0.0);
                cell.setUy(0.0);
            }
        }
    }
}

void Grid::resetGrid()
{
    for (int i = 0; i < this->width * this->height; ++i)
    {
        this->grid[i].resetCell();
    }
}

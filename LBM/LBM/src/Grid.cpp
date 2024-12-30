#include "Grid.h"
#include <string>
#include <iostream>

Grid::Grid(int w, int h) : width(w), height(h)
{
    grid = new Cell[width * height];
}

Grid::~Grid()
{
    delete[] grid;
}

Cell* Grid::getGridData()
{
    return grid;
}

const Cell* Grid::getGridData() const
{
    return grid;
}

int Grid::getWidth() const
{
    return this->width;
}

int Grid::getHeight() const
{
    return this->height;
}

Cell& Grid::getCell(int x, int y)
{
    return grid[y * width + x];
}

const Cell& Grid::getCell(int x, int y) const
{
    return grid[y * width + x];
}

void Grid::setCell(int x, int y, const Cell& cell)
{
    grid[y * width + x] = cell;
}

void Grid::initialize(bool defaultWall)
{
    resetGrid(); // Clear new walls

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            Cell& cell = getCell(x, y);
            double rhoInit = (x < width / 2) ? 1.0 : 0.5;
            cell.setRho(rhoInit);
            cell.setUx(0.0);
            cell.setUy(0.0);

            for (int i = 0; i < Q9; ++i)
            {
                double feq = w[i] * rhoInit; // u=0 -> (c_i · u)=0, => f_i^{eq} = w_i * rho
                cell.setF_in(i, feq);
                cell.setF_eq(i, feq);
                cell.setF_out(i, feq);
            }

            if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
            {
                cell.setWall(true);
            }

            int wallColumn = width / 2;
            int gapStart = height / 3;
            int gapEnd = 2 * height / 3;

            if (x == wallColumn && (y < gapStart || y > gapEnd))
            {
                cell.setWall(true);
            }
        }
    }
}

void Grid::resetGrid()
{
    for (int i = 0; i < width * height; ++i)
    {
        grid[i].resetCell();
    }
}
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
    //resetGrid();

    double w0 = 2.0 / 6.0;
    double w1 = 1.0 / 6.0; // for N,E,S,W

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            Cell& cell = getCell(x, y);
            double CInit = (x < width / 6) ? 1.0 : 0.0;
            cell.setC(CInit);

            cell.setF_in(0, w0 * CInit); // 0
            cell.setF_in(1, w1 * CInit); // N
            cell.setF_in(2, w1 * CInit); // E
            cell.setF_in(3, w1 * CInit); // S
            cell.setF_in(4, w1 * CInit); // W

            if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
            {
                cell.setWall(true);
            }

            int wallColumn = width / 6;
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
#include "Grid.h"
#include <string>
#include <iostream>

Grid::Grid(int w, int h) : width(w), height(h), grid(h, std::vector<Cell>(w)) {}

int Grid::getWidth() const
{
    return this->width;
}

int Grid::getHeight() const
{
    return this->height;
}

void Grid::initialize(bool defaultWall)
{
	for (std::size_t y = 0; y < width; ++y)
	{
		for (std::size_t x = 0; x < height; ++x)
		{
			grid[y][x] = Cell(defaultWall);

			if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
			{
				grid[y][x].setWall(true);
			}

            int wallColumn = width / 6;
            int gapStart = height / 3;
            int gapEnd = 2 * height / 3;

            if (x == wallColumn && (y < gapStart || y > gapEnd))
            {
                grid[y][x].setWall(true);
            }

			if (!grid[y][x].getWall() && x < width / 6)
			{
				for (int dir = 0; dir < 4; ++dir) 
				{
					grid[y][x].setDirection(dir, rand() % 2);
				}
			}
		}
	}
}

Cell& Grid::getCell(int x, int y)
{
	return grid[y][x];
}

const Cell& Grid::getCell(int x, int y) const
{
	return grid[y][x];
}

void Grid::setCell(int x, int y, Cell& cell)
{
	grid[y][x] = cell;
}

void Grid::collision()
{
    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            Cell& cell = grid[y][x];

            if (cell.getWall())
            {
                continue;
            }

            bool north = cell.getDirection(0);
            bool east = cell.getDirection(1);
            bool south = cell.getDirection(2);
            bool west = cell.getDirection(3);

            // Collision 1: (1, 0, 1, 0) ? (0, 1, 0, 1)
            if (north && south && !east && !west)
            {
                cell.setDirection(0, false);
                cell.setDirection(2, false);
                cell.setDirection(1, true);
                cell.setDirection(3, true);
            }
            // Collision 2: (0, 1, 0, 1) ? (1, 0, 1, 0)
            else if (east && west && !north && !south)
            {
                cell.setDirection(1, false);
                cell.setDirection(3, false);
                cell.setDirection(0, true); 
                cell.setDirection(2, true); 
            }
        }
    }
}

void Grid::streaming()
{
    std::vector<std::vector<Cell>> newGrid = grid;

    // Reset directions on the new grid
    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            if (!newGrid[y][x].getWall())
            {
                newGrid[y][x].resetDirections();
            }
        }
    }

    // Manage streaming and reflecting
    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            const Cell& cell = grid[y][x];

            if (cell.getWall())
            {
                continue;
            }

            // North
            if (cell.getDirection(0))
            {
                if (y < height - 1 && !grid[y + 1][x].getWall())
                {
                    newGrid[y + 1][x].setDirection(0, true); // Stream
                }
                else
                {
                    newGrid[y][x].setDirection(2, true); // Reflect
                }
            }

            // East
            if (cell.getDirection(1))
            {
                if (x < width - 1 && !grid[y][x + 1].getWall())
                {
                    newGrid[y][x + 1].setDirection(1, true); // Stream
                }
                else
                {
                    newGrid[y][x].setDirection(3, true); // Reflect
                }
            }

            // South
            if (cell.getDirection(2))
            {
                if (y > 0 && !grid[y - 1][x].getWall())
                {
                    newGrid[y - 1][x].setDirection(2, true); // Stream
                }
                else
                {
                    newGrid[y][x].setDirection(0, true); // Reflect
                }
            }

            // West
            if (cell.getDirection(3))
            {
                if (x > 0 && !grid[y][x - 1].getWall())
                {
                    newGrid[y][x - 1].setDirection(3, true); // Stream
                }
                else
                {
                    newGrid[y][x].setDirection(1, true); // Reflect
                }
            }
        }
    }

    grid = newGrid;
}

void Grid::resetGrid()
{
    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            grid[y][x].resetDirections();
        }
    }
}
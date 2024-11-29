#include "Grid.h"
#include <string>
#include <iostream>


void Grid::initialize(const std::string& defaultState, bool defaultWall)
{
	for (std::size_t y = 0; y < width; ++y)
	{
		for (std::size_t x = 0; x < height; ++x)
		{
			grid[y][x] = Cell(defaultState, defaultWall);

			if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
			{
				grid[y][x].setWall(true);
			}

			if (x == width / 6)
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


void Grid::printGrid() const
{
	for (std::size_t y = 0; y < height; ++y)
	{
		for (std::size_t x = 0; x < width; ++x)
		{
			std::cout << "Cell(" << x << ", " << y << "): " << "State=" << grid[y][x].getState() << ", isWall=" << grid[y][x].getWall() << "\n";
		}
	}
}

void Grid::collision()
{
	for (std::size_t y = 0; y < height; ++y)
	{
		for (std::size_t x = 0; x < width; ++x)
		{
			Cell& cell = grid[y][x];

			if (cell.getDirection(0) && cell.getDirection(2)) // Collision between North and South cells
			{
				cell.setDirection(0, false);
				cell.setDirection(2, false);

				cell.setDirection(1, true); // East
				cell.setDirection(3, true); // West
			}

			if (cell.getDirection(1) && cell.getDirection(3)) // Collision between East and West cells
			{
				cell.setDirection(1, false);
				cell.setDirection(3, false);

				cell.setDirection(0, true); // North
				cell.setDirection(2, true); // South
			}
		}
	}
}

void Grid::streaming()
{
	std::vector<std::vector<Cell>> newGrid(height, std::vector<Cell>(width, Cell("empty", false)));

	for (std::size_t y = 0; y < height; ++y)
	{
		for (std::size_t x = 0; x < width; ++x)
		{
			if(!newGrid[y][x].getWall())
			{
				newGrid[y][x].resetDirections();
			}
		}
	}

	for (std::size_t y = 0; y < height; ++y)
	{
		for (std::size_t x = 0; x < width; ++x)
		{
			const Cell& cell = grid[y][x];

			if (cell.getWall())
			{
				continue;
			}
			
			if (cell.getDirection(0) && y < height - 1)
			{
				newGrid[y + 1][x].setDirection(0, true);
			}

			if (cell.getDirection(1) && x < width - 1)
			{
				newGrid[y][x + 1].setDirection(1, true);
			}

			if (cell.getDirection(2) && y > 0)
			{
				newGrid[y - 1][x].setDirection(2, true);
			}

			if (cell.getDirection(3) && x > 0)
			{
				newGrid[y][x - 1].setDirection(3, true);
			}
		}
	}

	grid = newGrid;
}
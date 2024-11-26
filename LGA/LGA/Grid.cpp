#include "Grid.h"
#include <string>


void Grid::initialize(const std::string& defaultState = "empty", bool defaultWall = false)
{
	for (std::size_t y = 0; y < width; ++y)
	{
		for (std::size_t x = 0; x < height; ++x)
		{
			grid[y][x] = Cell(defaultState, defaultWall);
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

//TODO: collision streaming resetGrid printGrid
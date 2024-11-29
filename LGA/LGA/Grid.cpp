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

//TODO: collision streaming resetGrid 

void Grid::printGrid() const
{
	for (int y = 0; y < height; ++y) 
	{
		for (int x = 0; x < width; ++x) 
		{
			std::cout << "Cell(" << x << ", " << y << "): " << "State=" << grid[y][x].getState() << ", isWall=" << grid[y][x].getWall() << "\n";
		}
	}
}
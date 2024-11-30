#pragma once

#include <vector>
#include "Cell.h"

class Grid
{
private:
	std::vector<std::vector<Cell>> grid;
	int width;
	int height;

public:
	Grid(int w, int h);

	int getWidth() const;
	int getHeight() const;

	void initialize(bool defaultWall = false);

	Cell& getCell(int x, int y);
	const Cell& getCell(int x, int y) const;
	void setCell(int x, int y, Cell& cell);

	void collision();
	void streaming();
	void resetGrid();
};
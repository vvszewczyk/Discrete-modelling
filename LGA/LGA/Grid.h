#pragma once

#include "Cell.h"

class Grid
{
private:
	Cell* grid;
	int width;
	int height;

public:
	Grid(int w, int h);
	~Grid();

	int getWidth() const;
	int getHeight() const;

	void initialize(bool defaultWall = false);

	Cell& getCell(int x, int y);
	const Cell& getCell(int x, int y) const;
	void setCell(int x, int y, const Cell& cell);

	//void collision();
	//void streaming();
	void resetGrid();

	Cell* getGridData() { return grid; }
	const Cell* getGridData() const { return grid; }
};
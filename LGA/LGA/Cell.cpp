#include "Cell.h"
#include <algorithm>

Cell::Cell(bool w) : isWall(w), direction(4, false) {};

void Cell::setDirection(int dir, bool value)
{
	if (dir >= 0 && dir < direction.size())
	{
		direction[dir] = value;
	}
}

bool Cell::getDirection(int dir) const
{
	if (dir >= 0 && dir < direction.size())
	{
		return direction[dir];
	}
	return false;
}

void Cell::resetDirections()
{
	std::fill(direction.begin(), direction.end(), false);
}

void Cell::setWall(bool value)
{
	this->isWall = value;
}

bool Cell::getWall() const
{
	return this->isWall;
}

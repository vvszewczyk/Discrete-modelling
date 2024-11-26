#include "Cell.h"
#include <algorithm>

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
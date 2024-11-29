#pragma once

#include <string>
#include <vector>

class Cell
{
private:
	bool isWall;
	std::vector<bool> direction; // Directions: [N, E, S, W]

public:
	Cell(bool w = false) : isWall(w), direction(4, false) {};

	void setDirection(int dir, bool value);
	bool getDirection(int dir) const;

	void setWall(bool value)
	{
		this->isWall = value;
	}

	bool getWall() const
	{
		return this->isWall;
	}

	
	void resetDirections();
};
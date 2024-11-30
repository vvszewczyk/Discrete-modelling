#pragma once

#include <string>
#include <vector>

class Cell
{
private:
	bool isWall;
	std::vector<bool> direction; // Directions: [N, E, S, W]

public:
	Cell(bool w = "false");

	void setDirection(int dir, bool value);
	bool getDirection(int dir) const;

	void setWall(bool value);
	bool getWall() const;
	
	void resetDirections();
};
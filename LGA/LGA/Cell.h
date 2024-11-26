#pragma once

#include <string>
#include <vector>

class Cell
{
private:
	std::string state; // (empty, particle)
	bool isWall;
	std::vector<bool> direction; // Directions: [N, E, S, W]

public:
	Cell(std::string s = "empty", bool w = false) : state(s), isWall(w), direction(4, false) {};

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

	void setState(std::string& s)
	{
		this->state = s;
	}

	std::string getState() const
	{
		return this->state;
	}
	
	void resetDirections();
};
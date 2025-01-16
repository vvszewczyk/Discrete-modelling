#pragma once

#include "Cell.h"
#include "kernels.h"

class Grid
{
  private:
    Cell *grid;
    int width;
    int height;

  public:
    Grid(int w, int h);
    ~Grid();

    Cell *getGridData();
    const Cell *getGridData() const;

    int getWidth() const;
    int getHeight() const;
    Cell &getCell(int x, int y);
    const Cell &getCell(int x, int y) const;
    void setCell(int x, int y, const Cell &cell);

    void initialize(bool defaultWall = false); // Initiallize grid with cells (walls, particles etc.)
    void resetGrid();                          // Reset all directions on the whole grid

    // Sequential versions:
    // void collision();
    // void streaming();
};

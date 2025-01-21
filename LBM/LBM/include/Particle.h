#pragma once

class Particle
{
  public:
    // Coordinates
    float x;
    float y;

    // Velocity
    float vx;
    float vy;

    float m;    // mass
    float grav; // gravity

    // Color
    float r;
    float g;
    float b;

    bool active; // Activate/disactivate drawing
    std::vector<std::pair<float, float>> trace;

    // Path history
    void addTrace()
    {
        trace.emplace_back(x, y);
        if (trace.size() > 500000)
            trace.erase(trace.begin());
    }

    Particle() : x(0.f), y(0.f), vx(0.f), vy(0.f), m(0.f), grav(0.f), r(1.f), g(1.f), b(1.f), active(true)
    {
    }

    Particle(float px, float py, float pmass, float pg, float pr, float pgcol, float pb)
        : x(px), y(py), vx(0.f), vy(0.f), m(pmass), grav(pg), r(pr), g(pgcol), b(pb), active(true)
    {
    }
};

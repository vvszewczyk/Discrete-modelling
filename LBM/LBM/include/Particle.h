#pragma once

class Particle
{
  public:
    // Po³o¿enie w uk³adzie siatki (np. float, bo i tak bêdziemy interpolowaæ/zaokr¹glaæ)
    float x;
    float y;

    // Prêdkoœæ w³asna cz¹stki
    float vx;
    float vy;

    // Parametry fizyczne
    float mass; // "m" z instrukcji (zakres 0..1)
    float g;    // grawitacja (np. 0.005f)

    // Kolor do rysowania
    float r;
    float gcol; // "gcol" zamiast "g", by unikn¹æ kolizji nazwy
    float b;

    // Flaga aktywnoœci (by np. wy³¹czyæ rysowanie, gdy wypadnie poza siatkê)
    bool active;

    // Konstruktor domyœlny
    Particle() : x(0.f), y(0.f), vx(0.f), vy(0.f), mass(0.f), g(0.f), r(1.f), gcol(1.f), b(1.f), active(true)
    {
    }

    // Konstruktor z parametrami (opcjonalnie)
    Particle(float px, float py, float pmass, float pg, float pr, float pgcol, float pb)
        : x(px), y(py), vx(0.f), vy(0.f), mass(pmass), g(pg), r(pr), gcol(pgcol), b(pb), active(true)
    {
    }
};

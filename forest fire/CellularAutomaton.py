import numpy as np
from PIL import Image
from Cell import Cell

class CellularAutomaton:
    def __init__(self, width, height, image_path=None):
        self.width = width
        self.height = height
        self.image_path = image_path
        self.grid = self.initialize_grid()
        self.wind_direction = "N"

        if self.image_path:
            self.load_map_from_image(self.image_path)

    def initialize_grid(self):
        grid = np.empty((self.height, self.width), dtype=object)
        for x in range(self.height):
            for y in range(self.width):
                grid[x][y] = Cell("green")
        return grid

    def load_map_from_image(self, image_path):
        # Wczytaj obrazek
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.width, self.height))

        for x in range(self.height):
            for y in range(self.width):
                r, g, b = image.getpixel((y, x))  # Pobierz kolor piksela

                if b > 150 and b > g and b > r:  # Niebieski - woda
                    self.grid[x][y] = Cell("water")
                elif g > 100 and g > r and b < 80:  # Zielony - drzewa
                    self.grid[x][y] = Cell("green")
                elif r > 180 and g > 180 and b > 180:  # Szary - kamienie
                    self.grid[x][y] = Cell("rock")
                elif r > 130 and g > 60 and b < 30:  # Brązowy - scięte drzewa
                    self.grid[x][y] = Cell("cut")
                else:
                    self.grid[x][y] = Cell("green")  # Domyślny stan

    def step(self):
        new_grid = self.grid.copy()
        for x in range(self.height):
            for y in range(self.width):
                cell = self.grid[x][y]
                if cell.state == "burning":
                    new_grid[x][y].state = "ash"
                    self.spread_fire(new_grid, x, y)
                elif cell.state == "dynamite" and self.check_neighbor_burning(x, y):
                    self.explode(new_grid, x, y)
        self.grid = new_grid

    def check_neighbor_burning(self, x, y):
        neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neighbors_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                if self.grid[nx][ny].state == "burning":
                    return True
        return False

    def explode(self, grid, x, y):
        explosion_radius = 4
        for dx in range(-explosion_radius, explosion_radius + 1):
            for dy in range(-explosion_radius, explosion_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    grid[nx][ny].state = "burning"

    def spread_fire(self, grid, x, y):
        neighbors_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Sąsiedztwo Von Neumanna
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Sąsiedztwo Moore'a
        ]

        # Sprawdzenie sąsiadów
        for dx, dy in neighbors_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                neighbor = self.grid[nx][ny]

                if neighbor.state == "water":
                    grid[x][y].state = "green"
                    return

                if neighbor.state == "dynamite":
                    self.explode(grid, nx, ny)

                if neighbor.state == "ash":
                    continue

                if neighbor.state == "green" and self.should_catch_fire(neighbor, dx, dy):
                    neighbor.state = "burning"

    def should_catch_fire(self, cell, dx, dy):
        prob = (1 - cell.humidity) * (1 + cell.wind * 0.1)

        # Mapa kierunków wiatru
        wind_map = {
            "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
            "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
        }
        wind_dx, wind_dy = wind_map[self.wind_direction]

        # Bonus za zgodność z kierunkiem wiatru
        if (dx, dy) == (wind_dx, wind_dy):  # Dokładny kierunek wiatru
            prob += 0.1
        elif (dx, dy) in [
            (wind_dx + 1, wind_dy), (wind_dx - 1, wind_dy),  # Sąsiednie kierunki
            (wind_dx, wind_dy + 1), (wind_dx, wind_dy - 1)
        ]:
            prob += 0.01

        return np.random.rand() < prob
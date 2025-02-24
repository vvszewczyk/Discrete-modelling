import pygame

class Visualization:
    def __init__(self, automaton, cell_size=15):
        self.automaton = automaton
        self.cell_size = cell_size
        self.width = automaton.width * cell_size
        self.height = automaton.height * cell_size

    def draw_grid(self, screen):
        for x in range(self.automaton.height):
            for y in range(self.automaton.width):
                cell = self.automaton.grid[x][y]
                color = self._get_color(cell.state)
                pygame.draw.rect(
                    screen,
                    color,
                    (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                )

    def _get_color(self, state):
        colors = {
            "green": (34, 139, 34),
            "burning": (255, 69, 0),
            "ash": (169, 169, 169),
            "water": (30, 144, 255),
            "dynamite": (255, 255, 0),
            "cut": (139, 69, 19),
            "rock": (50, 50, 50)
        }
        return colors.get(state, (255, 255, 255))

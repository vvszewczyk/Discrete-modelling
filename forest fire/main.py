import pygame

from CellularAutomaton import CellularAutomaton
from Visualization import Visualization
from SimulationController import SimulationController


def main():
    pygame.init()
    cell_size = 15
    width = 1500 // cell_size
    height = (690 - 50) // cell_size

    image_path = "forest.png"
    automaton = CellularAutomaton(width, height, image_path=image_path)
    visualization = Visualization(automaton, cell_size)
    controller = SimulationController(automaton, visualization)

    screen = pygame.display.set_mode((1500, 700))
    pygame.display.set_caption("Чаваш Вармане")

    clock = pygame.time.Clock()
    last_step_time = 0

    while controller.running:
        controller.handle_events()

        current_time = pygame.time.get_ticks()
        if current_time - last_step_time > 1000 // controller.simulation_speed:
            if not controller.simulation_paused:
                automaton.step()
            last_step_time = current_time

        screen.fill((255, 255, 255))
        visualization.draw_grid(screen)
        controller.draw_buttons(screen)
        controller.draw_simulation_speed(screen)
        controller.draw_wind_direction(screen)
        pygame.display.flip()

        clock.tick(360)


if __name__ == "__main__":
    main()

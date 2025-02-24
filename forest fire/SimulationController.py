import pygame

class SimulationController:
    def __init__(self, automaton, visualization):
        self.automaton = automaton
        self.visualization = visualization
        self.running = True
        self.simulation_paused = True
        self.fire_active = False
        self.dynamite_active = False
        self.water_drop_active = False
        self.axe_active = False
        self.simulation_speed = 1

        # Przyciski
        self.buttons = [
            # Start i Reset
            {"label": "Start", "rect": pygame.Rect(10, 635, 100, 30), "action": self.toggle_pause},
            {"label": "Reset", "rect": pygame.Rect(10, 668, 100, 30), "action": self.reset_simulation},

            # Set Fire, Water Drop
            {"label": "Set Fire", "rect": pygame.Rect(140, 635, 120, 30), "action": self.fire_mode},
            {"label": "Water Drop", "rect": pygame.Rect(140, 668, 120, 30), "action": self.water_drop_mode},

            # Set Dynamite, Axe
            {"label": "Set Dynamite", "rect": pygame.Rect(263, 635, 140, 30), "action": self.dynamite_mode},
            {"label": "Axe", "rect": pygame.Rect(263, 668, 140, 30), "action": self.axe_mode},

            # Change Wind
            {"label": "Wind", "rect": pygame.Rect(1370, 640, 60, 50), "action": self.change_wind},

            # Speed Up, Slow Down
            {"label": " ▲", "rect": pygame.Rect(1307, 635, 20, 21), "action": self.speed_up},
            {"label": " ▼", "rect": pygame.Rect(1307, 675, 20, 20), "action": self.slow_down},
        ]

    def toggle_pause(self):
        self.simulation_paused = not self.simulation_paused
        for button in self.buttons:
            if button["label"] in ["Start", "Stop"]:
                button["label"] = "Stop" if not self.simulation_paused else "Start"

    def reset_simulation(self):
        self.automaton.load_map_from_image(self.automaton.image_path)
        self.simulation_paused = True
        for button in self.buttons:
            if button["label"] in ["Start", "Stop"]:
                button["label"] = "Start"

    def water_drop_mode(self):
        self.water_drop_active = not self.water_drop_active
        if self.water_drop_active:
            self.fire_active = False
            self.dynamite_active = False
            self.axe_active = False

    def fire_mode(self):
        self.fire_active = not self.fire_active
        if self.fire_active:
            self.water_drop_active = False
            self.dynamite_active = False
            self.axe_active = False

    def dynamite_mode(self):
        self.dynamite_active = not self.dynamite_active
        if self.dynamite_active:
            self.water_drop_active = False
            self.fire_active = False
            self.axe_active = False

    def axe_mode(self):
        self.axe_active = not self.axe_active
        self.water_drop_active = False
        self.fire_active = False
        self.dynamite_active = False

    def change_wind(self):
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        current_index = directions.index(self.automaton.wind_direction)
        self.automaton.wind_direction = directions[(current_index + 1) % len(directions)]

    def speed_up(self):
        if self.simulation_speed < 60:
            self.simulation_speed += 1

    def slow_down(self):
        if self.simulation_speed > 1:
            self.simulation_speed -= 1

    def handle_events(self):
        mouse_held = pygame.mouse.get_pressed()[0]  # Sprawdza, czy lewy przycisk myszy jest wciśnięty
        mouse_pos = pygame.mouse.get_pos()

        if mouse_held:
            x, y = mouse_pos
            cell_x = y // self.visualization.cell_size
            cell_y = x // self.visualization.cell_size
            if 0 <= cell_x < self.automaton.height and 0 <= cell_y < self.automaton.width:
                if self.water_drop_active:
                    # Stawianie wody w obszarze 2x2
                    for dx in range(-1, 1):
                        for dy in range(-1, 1):
                            nx, ny = cell_x + dx, cell_y + dy
                            if 0 <= nx < self.automaton.height and 0 <= ny < self.automaton.width:
                                self.automaton.grid[nx][ny].state = "water"
                elif self.fire_active:
                    # Ogień można stawiać tylko na "drzewach" lub "dynamicie"
                    if self.automaton.grid[cell_x][cell_y].state in ["green", "dynamite"]:
                        self.automaton.grid[cell_x][cell_y].state = "burning"
                elif self.dynamite_active:
                    # Dynamit można ustawić w dowolnym miejscu
                    self.automaton.grid[cell_x][cell_y].state = "dynamite"
                elif self.axe_active:
                    # Wycinanie drzew - tylko na zielonych
                    if self.automaton.grid[cell_x][cell_y].state == "green":
                        self.automaton.grid[cell_x][cell_y].state = "cut"

        # Obsługa zdarzeń systemowych
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.handle_button_click(event.pos):
                    return

    def handle_button_click(self, pos):
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                button["action"]()
                return True
        return False

    def draw_buttons(self, screen):
        for button in self.buttons:
            if (button["label"] == "Water Drop" and self.water_drop_active) or \
               (button["label"] == "Set Fire" and self.fire_active) or \
               (button["label"] == "Set Dynamite" and self.dynamite_active) or \
                    button["label"] == "Axe" and self.axe_active:
                color = (150, 200, 255)  # Aktywny tryb
            else:
                color = (200, 200, 200)  # Nieaktywny
            pygame.draw.rect(screen, color, button["rect"])
            pygame.draw.rect(screen, (0, 0, 0), button["rect"], 2)

            font = pygame.font.SysFont("Arial", 22)
            text = font.render(button["label"], True, (0, 0, 0))
            text_rect = text.get_rect(center=button["rect"].center)  # Wyśrodkowanie tekstu
            screen.blit(text, text_rect)  # Rysowanie tekstu

    def draw_simulation_speed(self, screen):
        font = pygame.font.Font(None, 20)
        speed_text = f"{self.simulation_speed} FPS"
        text = font.render(speed_text, True, (0, 0, 0))

        text_rect = text.get_rect()
        text_rect.topleft = (1300, 659)  # Lewy górny róg
        screen.blit(text, text_rect)

    def draw_wind_direction(self, screen):
        # Pozycja strzałki na ekranie
        center_x, center_y = 1470, 670  # Środek strzałki

        # Mapa przesunięć dla kierunków wiatru
        arrow_map = {
            "N": (0, -15),  # Strzałka w górę
            "NE": (12, -12),  # Strzałka w prawo-górę
            "E": (15, 0),  # Strzałka w prawo
            "SE": (12, 12),  # Strzałka w prawo-dół
            "S": (0, 15),  # Strzałka w dół
            "SW": (-12, 12),  # Strzałka w lewo-dół
            "W": (-15, 0),  # Strzałka w lewo
            "NW": (-12, -12),  # Strzałka w lewo-górę
        }

        # Punkt startowy strzałki (środek)
        start_x, start_y = center_x, center_y - 9

        # Koniec strzałki na podstawie kierunku wiatru
        dx, dy = arrow_map[self.automaton.wind_direction]
        end_x, end_y = start_x + dx, start_y + dy

        pygame.draw.line(screen, (0, 0, 0), (start_x, start_y), (end_x, end_y), 3)  # Strzałka

        pygame.draw.circle(screen, (0, 0, 0), (start_x, start_y), 5)  # Kropka w środku strzałki

        # Dodanie tekstu z kierunkiem wiatru
        font = pygame.font.Font(None, 24)
        wind_text = f"{self.automaton.wind_direction}"
        text = font.render(wind_text, True, (0, 0, 0))
        screen.blit(text, (center_x - 6, center_y + 10))


import numpy as np
from PIL import Image, ImageDraw


#  Automat komórkowy 2D - Gra w Życie
def cellular_automaton_2D(grid, steps, boundary):
    size_x, size_y = grid.shape
    output = np.zeros((steps, size_x, size_y), dtype=int)
    output[0] = grid

    # Warianty położeń względem komórki (Sąsiedztwo Moore'a)
    neighbors_offsets = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1),           (0, 1),
                          (1, -1),  (1, 0),  (1, 1)]

    for t in range(1, steps):
        new_grid = output[t - 1].copy()

        for x in range(size_x):
            for y in range(size_y):
                neighbors = 0
                for dx, dy in neighbors_offsets:
                    nx, ny = x + dx, y + dy # współrzędne sąsiada

                    # p(periodyczny), r(odbijający)
                    if boundary == "p":
                        if nx < 0:
                            nx += size_x
                        elif nx >= size_x:
                            nx -= size_x

                        if ny < 0:
                            ny += size_y
                        elif ny >= size_y:
                            ny -= size_y

                    elif boundary == "r":
                        if nx < 0:
                            nx = 0
                        elif nx >= size_x:
                            nx = size_x - 1

                        if ny < 0:
                            ny = 0
                        elif ny >= size_y:
                            ny = size_y - 1

                    neighbors += output[t - 1][nx][ny]

                # Reguły
                if output[t - 1][x][y] == 0:
                    # 1) Jeśli komórka martwa i 3 żywych sąsiadów -> staje się żywa
                    if neighbors == 3:
                        new_grid[x][y] = 1
                else:
                    # 2) Jeśli komórka żywa i posiada 2 lub 3 sąsiadów żywych -> pozostaje żywa
                    if neighbors == 2 or neighbors == 3:
                        new_grid[x][y] = 1
                    # 3) Jeśli komórka żywa posiada > 3 sąsiadów żywych -> umiera
                    elif neighbors > 3:
                        new_grid[x][y] = 0
                    # 4) Jeśli komórka żywa posiada < 2 sąsiadów żywych -> umiera
                    elif neighbors < 2:
                        new_grid[x][y] = 0

        output[t] = new_grid

    return output


def glider(grid, x=0, y=0):
    glider_array = np.array([
                       [0, 1, 1],
                       [1, 1, 0],
                       [0, 1, 0]])

    if x + glider_array.shape[0] <= grid.shape[0] and y + glider_array.shape[1] <= grid.shape[1]:
        grid[x:x + glider_array.shape[0], y:y + glider_array.shape[1]] = glider_array
    else:
        print("Glider position exceeds grid boundaries.")


def oscillator(grid, x=0, y=0):
    oscillator_array = np.array([
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0]])
    if x + oscillator_array.shape[0] <= grid.shape[0] and y + oscillator_array.shape[1] <= grid.shape[1]:
        grid[x:x + oscillator_array.shape[0], y:y + oscillator_array.shape[1]] = oscillator_array
    else:
        print("Oscillator position exceeds grid boundaries.")



def random(grid, x=0, y=0, side = 3):
    random_array = np.random.choice([0, 1], size=(side,side))
    if x + random_array.shape[0] <= grid.shape[0] and y + random_array.shape[1] <= grid.shape[1]:
        grid[x:x + random_array.shape[0], y:y + random_array.shape[1]] = random_array
    else:
        print("Random position exceeds grid boundaries.")



def stable(grid, x=0, y=0):
    stable_array = np.array([
                       [0, 1, 1, 0],
                       [1, 0, 0, 1],
                       [0, 1, 1, 0]])
    if x + stable_array.shape[0] <= grid.shape[0] and y + stable_array.shape[1] <= grid.shape[1]:
        grid[x:x + stable_array.shape[0], y:y + stable_array.shape[1]] = stable_array
    else:
        print("Stable pattern position exceeds grid boundaries.")

def oscillator2(grid, x=0, y=0):
    hehe_array = np.array([
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
])
    if x + hehe_array.shape[0] <= grid.shape[0] and y + hehe_array.shape[1] <= grid.shape[1]:
        grid[x:x + hehe_array.shape[0], y:y + hehe_array.shape[1]] = hehe_array
    else:
        print("Oscillator2 pattern position exceeds grid boundaries.")

def period_246_glider_gun(grid, x=0, y=0):
    gun = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    if x + gun.shape[0] <= grid.shape[0] and y + gun.shape[1] <= grid.shape[1]:
        grid[x:x + gun.shape[0], y:y + gun.shape[1]] = gun
    else:
        print("PERIOD 246 GLIDER GUN pattern position exceeds grid boundaries.")


# Funkcja do tworzenia GIF-a z symulacji
def create_gif(history, filename, cell_size=10):
    frames = []
    for step in history:
        img = Image.new("RGB", (step.shape[1] * cell_size, step.shape[0] * cell_size), "yellow")
        draw = ImageDraw.Draw(img)
        for x in range(step.shape[0]):
            for y in range(step.shape[1]):
                if step[x, y] == 1:  # Rysowanie żywych komórek jako czarne kwadraty
                    draw.rectangle(
                        [y * cell_size, x * cell_size, (y + 1) * cell_size, (x + 1) * cell_size],
                        fill="black"
                    )
        frames.append(img)


    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # Czas trwania każdej klatki w milisekundach
        loop=0
    )


def main():
    size_x, size_y = 25, 25
    #size_x, size_y = 100, 100
    steps = 15
    grid = np.zeros((size_x, size_y), dtype=int)

    pattern_choice = input("Select a starting pattern (g(glider), o(oscillator), r(random), s(stable), gg(GLIDER GUN), o2(oscillator2)): ").strip().lower()
    try:
        x_choice = int(input("Select x coordinate: "))
        y_choice = int(input("Select y coordinate: "))
    except ValueError:
        print("Invalid coordinates. Using default (0, 0).")
        x_choice, y_choice = 0, 0

    if not (0 <= x_choice < size_x and 0 <= y_choice < size_y):
        print(f"Coordinates ({x_choice}, {y_choice}) are out of grid bounds. Using default (0, 0).")
        x_choice, y_choice = 0, 0

    if pattern_choice == "g":
        glider(grid, x_choice, y_choice)
    elif pattern_choice == "o":
        oscillator(grid, x_choice, y_choice)
    elif pattern_choice == "r":
        random(grid, x_choice, y_choice)
    elif pattern_choice == "s":
        stable(grid, x_choice, y_choice)
    elif pattern_choice == "o2":
        oscillator2(grid, x_choice, y_choice)
    elif pattern_choice == "gg":
        period_246_glider_gun(grid, x_choice, y_choice)
    else:
        print("Invalid pattern. Default r(random) pattern used.")
        random(grid)

    boundary_choice = input("Select boundary condition (p(periodic), r(reflecting)): ").strip().lower()
    if boundary_choice not in ["p", "r"]:
        print("Invalid boundary condition. Default p(periodic) used.")
        boundary_choice = "p"

    simulation = cellular_automaton_2D(grid, steps, boundary=boundary_choice)

    # Generowanie GIF-a
    gif_name = input("Enter GIF name:")
    if len(gif_name) > 0:
        create_gif(simulation, filename=gif_name + ".gif")
    else:
        gif_name = "unknown"
        create_gif(simulation, filename=gif_name + ".gif")

    print("GIF has been saved as " + gif_name + ".gif")


if __name__ == "__main__":
    main()
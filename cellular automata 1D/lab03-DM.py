import numpy as np
import matplotlib.pyplot as plt
import csv

def cellular_automaton(initial_state, steps, rule_number, boundary):
    size = len(initial_state)
    output = np.zeros((steps, size), dtype=int)

    for i in range(size):
        output[0][i] = initial_state[i]

    binaryRule = f"{rule_number:08b}" # 08 - ilość cyfr, b - konwersja na liczbę binarną
    ruleOutputs = [0] * 8
    for i in range(8):
        ruleOutputs[i] = int(binaryRule[i])

    for t in range(1, steps):
        for i in range(size):
            # Stan komórki po lewej stronie komórki i
            if i > 0:
                left = output[t - 1][i - 1]
            else: # i == 0
                left = output[t - 1][size - 1] if boundary == "periodic" else 0

            # Stan aktualnej komórki i w poprzednim kroku czasowym
            center = output[t - 1][i]

            # Stan komórki po prawej stronie komórki i
            if i < size - 1:
                right = output[t - 1][i + 1]
            else: # i == size - 1
                right = output[t - 1][0] if boundary == "periodic" else 0

            # Znalezienie indeksu pierwszej kombinacji w binary_vectors
            neighborhood = [left, center, right]
            index = binary_vectors.index(neighborhood)

            # Ustawienie wyniku w bieżącym wierszu
            output[t][i] = ruleOutputs[index]

    return output

# Warianty sąsiedztw
binary_vectors = [
    #L  C  R
    [1, 1, 1], # ruleOutputs[0]
    [1, 1, 0], # ruleOutputs[1]
    [1, 0, 1], # ruleOutputs[2]
    [1, 0, 0], # ruleOutputs[3]
    [0, 1, 1], # ruleOutputs[4]
    [0, 1, 0], # ruleOutputs[5]
    [0, 0, 1], # ruleOutputs[6]
    [0, 0, 0]  # ruleOutputs[7]
]

def visualize(history, title="Wizualizacja automatu komórkowego"):
    plt.figure(figsize=(10, 6))
    plt.imshow(history, cmap="binary", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Pozycja komórki")
    plt.ylabel("Krok czasowy")
    plt.show()

def getInput():
    while True:
        try:
            size = int(input("Podaj liczbę komórek (długość wiersza): "))
            steps = int(input("Podaj liczbę iteracji (ilość kroków czasowych): "))
            if size > 0 and steps > 0:
                return size, steps
            else:
                print("Błąd: Liczba komórek i liczba iteracji muszą być większe od zera.")
        except ValueError:
            print("Błąd: Wprowadź prawidłowe liczby całkowite.")

def saveCSV(history, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in history:
            writer.writerow(row)

def main():
    size, steps = getInput()
    initial_state = np.zeros(size, dtype=int)
    pattern = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    initial_state[:min(size, len(pattern))] = pattern[:min(size, len(pattern))]

    rules = [41, 67, 30, 190]

    for rule in rules:
        # Periodyczny warunek brzegowy
        periodic = cellular_automaton(initial_state, steps, rule, boundary="periodic")
        visualize(periodic, title=f"Automat komórkowy 1D - Reguła {rule} (periodyczny)")
        saveCSV(periodic, f"output/output_rule_{rule}_periodic.csv")


        # Absorpcyjny warunek brzegowy
        absorbing = cellular_automaton(initial_state, steps, rule, boundary="absorbing")
        visualize(absorbing, title=f"Automat komórkowy 1D - Reguła {rule} (absorpcyjny)")
        saveCSV(absorbing, f"output/output_rule_{rule}_absorbing.csv")

if __name__ == "__main__":
    main()

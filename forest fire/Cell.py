class Cell:
    def __init__(self, state, humidity=0.5, wind=0):
        self.state = state  # "green", "burning", "burned", "water", "dynamite", "cut", "rock"
        self.humidity = humidity
        self.wind = wind

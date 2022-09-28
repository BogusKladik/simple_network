import random, math

class Neuron():
    def __init__(self, count_weigths: int) -> object:
        self.weigths = [random.uniform(-1, 1) for _ in range(count_weigths)]

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def predict(self, data: list) -> float:
        return self._sigmoid(sum([self.weigths[i] * data[i] for i in range(len(self.weigths))]))

    def _learn(self, data: list, result: float, goal: float, nu: float = 0.01) -> None:
        error = goal - result
        for i in range(len(self.weigths)):
            self.weigths[i] = self.weigths[i] + error * nu * data[i]

    def train(self, data_set: list, goal_set: list, iter: int = 100) -> None:
        for _ in range(iter):
            for i in range(len(data_set)):
                prediction = self.predict(data_set[i])
                self._learn(data_set[i], prediction, goal_set[i])

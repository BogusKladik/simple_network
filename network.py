from neuron import Neuron

class Network():
    def __init__(self, count_weigths: int, count_neurons: int = 1) -> object:
        self.neurons = [Neuron(count_weigths) for _ in range(count_neurons)]

    def predict(self, data: list) -> list:
        return [self.neurons[i].predict(data) for i in range(len(self.neurons))]

    def _learn(self, data: list, result: list, goal: list, nu: float = 0.01) -> None:
        for i in range(len(goal)):
            self.neurons[i]._learn(data, result[i], goal[i], nu)

    def train(self, data_set: list, goal_set: list, iter: int = 100) -> None:
        for _ in range(iter):
            for i in range(len(data_set)):
                prediction = self.predict(data_set[i])
                self._learn(data_set[i], prediction, goal_set[i])

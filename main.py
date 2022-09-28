from network import Network

def get_data(data: list,number: int) -> None:
    with open("data/" + str(number) + ".data") as f:
        for nums in f:
            data.append(list(map(float, nums.split(' '))))

def main() -> None:
    data_set = []
    for i in range(10):
        get_data(data_set, i)
    
    goal_set = [[1 if j == i else 0 for j in range(10)] for i in range(10)]

    network = Network(len(data_set[0]), len(data_set))
    print("\n\nВеса до обучения:\n\n")
    for i in range(len(data_set)):
        print(f"Вес {i + 1} нейрона отвечающего за {i}:")
        print(network.neurons[i].weigths)

    network.train(data_set, goal_set, 1200)
    print("\n\nВеса после обучения:\n\n")
    for i in range(len(data_set)):
        print(f"Вес {i + 1} нейрона отвечающего за {i}:")
        print(network.neurons[i].weigths)

    print("\n\nПроверка нейросети на правильность(цифры должный идти по порядку)\n\n")
    for i in range(10):
        prediction = network.predict(data_set[i])
        print(prediction)
        result_number = prediction.index(max(prediction))
        print(result_number)


if __name__ == "__main__":
    main()

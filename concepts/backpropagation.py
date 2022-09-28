import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, X, Y, hidden_layer_size=4):
        self.hidden_layer_size = hidden_layer_size

        self.X = np.array(X)
        self.Y = np.array(Y)

        self.Wh = np.random.rand(self.X.shape[1], self.hidden_layer_size)
        self.Wo = np.random.rand(self.hidden_layer_size, self.Y.shape[1])

    def feedforward(self):

        self.Zh = np.array(np.dot(self.X, self.Wh))
        self.Yh = np.array(sigmoid(self.Zh))

        self.Zo = np.array(np.dot(self.Yh, self.Wo))
        self.Yo = np.array(sigmoid(self.Zo))

        self.Loss = np.mean(np.square(self.Y - self.Yo))

    def backpropagation(self):
        # L = (Y - Yo)^2
        dL_dYo = np.array(2 * (self.Y - self.Yo))
        # Yo = sigmoid(Zo)
        dYo_dZo = sigmoid_derivative(self.Yo)
        # Zo = Yh * Wo
        dZo_dWo = self.Yh.T

        # dL_dWo = dZo_dWo * dYo_dZo * dL_dYo
        dL_dWo = np.dot(dZo_dWo, dL_dYo * dYo_dZo)

        self.Wo += dL_dWo

        # Zo = Yh * Wo
        dZo_dYh = self.Wo.T
        # Yh = sigmoid(Zh)
        dYh_Zh = sigmoid_derivative(self.Yh)
        # Zh = X * Wh
        Zh_dWh = self.X.T

        # dL_dWh = Zh_dWh * dYh_Zh * dZo_dYh * dYo_dZo * dL_dYo
        dL_dWh = np.dot(Zh_dWh, np.dot(dL_dYo * dYo_dZo, dZo_dYh) * dYh_Zh)

        self.Wh += dL_dWh

    def train(self, iterations, sample_rate=100):
        losses = []
        for i in range(1, iterations + 1):
            self.feedforward()
            self.backpropagation()
            if i % sample_rate == 0:
                losses.append([i, nn.Loss])

        lossesT = np.array(losses).T
        print(
            pd.DataFrame(
                {
                    "iteration": lossesT[0],
                    "loss": lossesT[1],
                }
            ).head(len(losses))
        )


X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]))
Y = np.array(([0], [1], [1], [0]))
nn = NeuralNetwork(X, Y)

iterations = 1000

nn.train(iterations)

print(
    pd.DataFrame(
        {
            "input 1": X.T[0],
            "input 2": X.T[1],
            "input 3": X.T[2],
            "target": Y.T[0],
            "output": nn.Yo.T[0],
        }
    ).head(len(X))
)

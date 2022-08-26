import numpy as np


class Perceptron:
    def __init__(self, inputs, weights, name=None):
        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
        self.name = name or "Default perceptron"
        self.size = len(self.inputs)

    def decide(self, threshold):
        return (self.inputs @ self.weights) >= threshold


inputs, weights = [], []

considerations = ["does it have warranty? ", "does it have promo? "]

for consideration in considerations:
    i = int(input(consideration))
    w = int(input("weight: "))

    inputs.append(i)
    weights.append(w)

threshold = int(input("threshold: "))


perceptron = Perceptron(inputs, weights)

print("Perceptron size:", perceptron.size)

print("Buy it?", perceptron.decide(threshold))

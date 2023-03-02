from numpy import *
from random import *

class NeuralNetwork:
    def __init__(self, layout):

        self.layout = layout
        self.layersNum = len(self.layout)
        self.layersMaxSize = 0
        self.inputSize = self.layout[0]
        self.outputSize = self.layout[self.layersNum - 1]

        for i in range(self.layersNum):
            if (self.layersMaxSize < self.layout[i]):
                self.layersMaxSize = self.layout[i]


        # self.values[layer][node]                  filled with -1
        self.values = []
        for i in range(self.layersNum):
            self.values.append([])
            for j in range(self.layersMaxSize):
                self.values[i].append(-1)

        # self.weights[layer][node][previous_node]  filled with 1
        self.weights = []
        for i in range(self.layersNum):
            self.weights.append([])
            for j in range(self.layersMaxSize):
                self.weights[i].append([])
                for k in range(self.layersMaxSize):
                    w = uniform(0, 2)
                    self.weights[i][j].append(w)

        self.biases = []
        # self.bias[layer][node]                    filled with 0
        for i in range(self.layersNum):
            self.biases.append([])
            for j in range(self.layersMaxSize):
                b = uniform(0, 1)
                self.biases[i].append(b)

        self.sigmoid_derivative = []
        # self.sigmoid_derivative[layer][node]      filled with 0
        for i in range(self.layersNum):
            self.sigmoid_derivative.append([])
            for j in range(self.layersMaxSize):
                self.sigmoid_derivative[i].append(0)

        self.delta = []
        # self.cost[layer][node]                    filled with 0
        for i in range(self.layersNum):
            self.delta.append([])
            for j in range(self.layersMaxSize):
                self.delta[i].append(0)

        self.inputLayer = []
        self.outputLayer = []

    def sigmoid(self, x):
        # try:
        #     ans = 1/(1 + exp(-x))
        # except OverflowError:
        #     ans = float(1/(1 + exp(-x)))

        return 1/(1 + exp(-x))
    

    def forward(self, X):

        self.inputLayer = X

        if (len(self.inputLayer) != self.inputSize):
            print("Invalid input size")
            return

        for layer in range(self.layersNum):
            if (layer == 0):
                for node in range(self.inputSize):
                    self.values[0][node] = self.inputLayer[node]
            else:
                for node in range(self.layout[layer]):
                    current_value = 0
                    for previous_node in range(self.layout[layer - 1]):
                        current_value += self.weights[layer][node][previous_node]*(self.values[layer - 1][previous_node])

                    current_value = self.sigmoid(current_value + self.biases[layer][node])
                    self.values[layer][node] = current_value
                    self.sigmoid_derivative[layer][node] = current_value*(1 - current_value)

        self.outputLayer = [x for x in self.values[self.layersNum - 1] if x != -1]

    def backward(self, Y):
        for layer in reversed(range(self.layersNum)):
            if (layer == self.layersNum - 1):
                for node in (range(self.outputSize)):
                    self.delta[layer][node] = ((self.values[layer][node] - Y[node]) * self.sigmoid_derivative[layer][node])
            else:
                for node in (range(self.layout[layer])):
                    sum = 0.0
                    for next_node in (range(self.layout[layer + 1])):
                        sum += (self.delta[layer + 1][next_node] * self.weights[layer + 1][next_node][node])
                    self.delta[layer][node] = sum * self.sigmoid_derivative[layer][node]

    def calculateGradientVector(self, dataSize, gradient_weights, gradient_biases):
        for layer in range(self.layersNum):
            if (layer != 0):
                for node in range(self.layout[layer]):      
                    gradient_biases[layer][node] += self.delta[layer][node] / dataSize
                    for previous_node in range(self.layout[layer - 1]):
                        gradient_weights[layer][node][previous_node] += (self.delta[layer][node] * self.values[layer - 1][previous_node]) / dataSize                    

    def updateWeights(self, rate, gradient_weights, gradient_biases):
        for layer in range(self.layersNum):
            if (layer != 0):
                for node in range(self.layout[layer]): 
                    self.biases[layer][node] -= rate * gradient_biases[layer][node]        
                    for previous_node in range(self.layout[layer - 1]):
                        self.weights[layer][node][previous_node] -= rate*gradient_weights[layer][node][previous_node]
                           

    def interval(self, dataSet, rate):
        if (len(dataSet[0][0]) != self.inputSize or len(dataSet[0][1]) != self.outputSize):
            print("Invalid input or output size")
            return
        
        gradient_weights = []
        # self.gradient_weights[layer][node][previous weight]       filled with 0
        for i in range(self.layersNum):
            gradient_weights.append([])
            for j in range(self.layersMaxSize):
                gradient_weights[i].append([])
                for k in range(self.layersMaxSize):
                    gradient_weights[i][j].append(0)

        gradient_biases = []
        # self.gradient_biases[layer][node]                         filled with 0
        for i in range(self.layersNum):
            gradient_biases.append([])
            for j in range(self.layersMaxSize):
                gradient_biases[i].append(0)

        for i in range(len(dataSet)):
            X = dataSet[i][0]
            Y = dataSet[i][1]
            self.forward(X)
            self.backward(Y)
            self.calculateGradientVector(len(dataSet), gradient_weights, gradient_biases)
        self.updateWeights(rate, gradient_weights, gradient_biases)

    def train(self, dataSet, rate, interval):
        for time in range(interval):
            self.interval(dataSet, rate)            
            
            

    def feedForward(self, X):
        self.forward(X)
        print(self.outputLayer)
    


if __name__ == "__main__":

    layout = [3,4,1]
    NN = NeuralNetwork(layout)

    X1 = [0.1, 0.1, 0.1]
    Y1 = [0.1]
    X2 = [0.2, 0.2, 0.2]
    Y2 = [0.2]
    X3 = [0.3, 0.3, 0.3]
    Y3 = [0.3]

    dataSet = [[X1, Y1],[X2, Y2],[X3,Y3]]

    NN.feedForward(X1)
    NN.feedForward(X2)
    NN.feedForward(X3)

    print()

    NN.train(dataSet, rate=0.1, interval=100000)

    NN.feedForward(X1)
    NN.feedForward(X2)
    NN.feedForward(X3)
    NN.feedForward([0.4,0.4,0.4])
    
    
    print()

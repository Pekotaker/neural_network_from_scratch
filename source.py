from numpy import *
from random import *

class NeuralNetwork:

    # Constructor
    def __init__(self, layout):

        self.layout = layout
        self.layersNum = len(self.layout)
        self.layersMaxSize = 0
        self.inputSize = self.layout[0]
        self.outputSize = self.layout[self.layersNum - 1]

        for i in range(self.layersNum):
            if (self.layersMaxSize < self.layout[i]):
                self.layersMaxSize = self.layout[i]


        # self.activation_values[layer][node]      filled with -1
        self.activation_value = []
        for i in range(self.layersNum):
            self.activation_value.append([])
            for j in range(self.layersMaxSize):
                self.activation_value[i].append(-1)

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
        # self.delta[layer][node]                    filled with 0
        for i in range(self.layersNum):
            self.delta.append([])
            for j in range(self.layersMaxSize):
                self.delta[i].append(0)

        self.inputLayer = []
        self.outputLayer = []

    # Sigmoid function
    def sigmoid(self, x):
        return 1/(1 + exp(-x))
    
    # The derivative of Sigmoid Function
    def sigmoidDerivative(self,x):
        return (self.sigmoid(x)*(1 - self.sigmoid(x)))
    

    # Forward propagation
    def forward(self, X):
        self.inputLayer = X

        for layer in range(self.layersNum):

            # The first layer (input layer)
            if (layer == 0):

                for node in range(self.inputSize):
                    self.activation_value[0][node] = self.inputLayer[node]

            # The other layers
            else:

                for node in range(self.layout[layer]):
                    
                    # Value before activation : z[j][l] = sum{k}(weight[j,k][l] * a[k][l-1]) + bias[j][l]
                    z_value = 0
                    for previous_node in range(self.layout[layer - 1]):

                        z_value += self.weights[layer][node][previous_node] * (self.activation_value[layer - 1][previous_node])

                    # Activation value : a[j][l] = sigmoid(z[j][l])
                    self.activation_value[layer][node] = self.sigmoid(z_value + self.biases[layer][node])

                    # The derivative of the sigmoid function
                    self.sigmoid_derivative[layer][node] = self.sigmoidDerivative(self.activation_value[layer][node])

        self.outputLayer = [x for x in self.activation_value[self.layersNum - 1] if x != -1]


    # Backward propagation : calculating delta for each neuron
    def backward(self, Y):

        # Y is the desired activation value
        for layer in reversed(range(self.layersNum)):

            # Start with the last layer
            if (layer == self.layersNum - 1):
                for node in (range(self.outputSize)):

                    # The constant "2" is optional, it will be multiplied along with the rates anyway
                    self.delta[layer][node] = (2*(self.activation_value[layer][node] - Y[node]) * self.sigmoid_derivative[layer][node])
            else:

            # Propagate backward to previous layers
                for node in (range(self.layout[layer])):
                    sum = 0.0

                    for next_node in (range(self.layout[layer + 1])):
                        sum += (self.delta[layer + 1][next_node] * self.weights[layer + 1][next_node][node])

                    self.delta[layer][node] = sum * self.sigmoid_derivative[layer][node]


    # The gradient vector is calculated for each sample in the data set, then take the average value out of all the samples in said data set
    def calculateGradientVector(self, dataSize, gradient_weights, gradient_biases):
        for layer in range(self.layersNum):
            if (layer != 0):
                for node in range(self.layout[layer]):    

                    # The gradient vector for biases  
                    gradient_biases[layer][node] += self.delta[layer][node] / dataSize
                    for previous_node in range(self.layout[layer - 1]):

                        # The gradient vector for weights
                        gradient_weights[layer][node][previous_node] += (self.delta[layer][node] * self.activation_value[layer - 1][previous_node]) / dataSize                    


    # Used the previous calculated gradient vector to adjust the weights and biases of the network
    def updateWeightsAndBiases(self, rate, gradient_weights, gradient_biases):
        for layer in range(self.layersNum):
            if (layer != 0):
                for node in range(self.layout[layer]): 

                    # Update biases
                    self.biases[layer][node] -= rate * gradient_biases[layer][node]        
                    for previous_node in range(self.layout[layer - 1]):

                        # Update weights
                        self.weights[layer][node][previous_node] -= rate*gradient_weights[layer][node][previous_node]
                           

    # Learning one data set
    def interval(self, dataSet, rate):

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

        # Learning
        for i in range(len(dataSet)):
            X = dataSet[i][0]
            Y = dataSet[i][1]
            self.forward(X)
            self.backward(Y)
            self.calculateGradientVector(len(dataSet), gradient_weights, gradient_biases)
        self.updateWeightsAndBiases(rate, gradient_weights, gradient_biases)


    # Checking data sets' layout and start training by looping through intervals
    def train(self, dataSet, rate, interval):

        # Verify datasets' format
        for i in range(len(dataSet)):
            if (len(dataSet[i][0]) != self.inputSize):
                print("Invalid input size")
                return     
            if (len(dataSet[i][1]) != self.outputSize):
                print("Invalid output size")
                return
        
        # Training
        for time in range(interval):
            self.interval(dataSet, rate)            


    # Check the output
    def feedForward(self, X):

        self.inputLayer = X
        if (len(self.inputLayer) != self.inputSize):
            print("Invalid input size")
            return
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

import value_class
import random

class Neuron:
    def __init__(self, nin):
        self.weights = [value_class.Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = value_class.Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = act.tanh()
        return out

class Layer:
    def __init__(self, nin, nouts):
        self.neurons = [Neuron(nin) for _ in range(nouts)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
from numpy import exp

def sigmoid(x):
  return 1 / (1 + exp(-x))

def sigmoidPrime(x):
    sig = sigmoid(x)

    return sig * (1 - sig)

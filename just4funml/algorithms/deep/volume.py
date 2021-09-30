import numpy as np

class Vol:
    def __init__(self, tensor):
        self.tensor = tensor
        self.grads = np.zeros((tensor.shape))
    
    def __repr__(self):
        return f"{id(self)}: tensor{self.tensor.shape} | grads{self.grads.shape}"

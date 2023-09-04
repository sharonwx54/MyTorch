import numpy as np


class Linear:

    def __init__(self, in_features, out_features, weight_init_fn=None, bias_init_fn=None, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        if weight_init_fn is None:
            self.W = np.zeros((out_features, in_features))  
        else:
            self.W = weight_init_fn(out_features, in_features)
        if bias_init_fn is None:
            self.b = np.zeros((out_features, 1)) 
        else:
            self.b = bias_init_fn(out_features)

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = self.A.shape[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = self.A@self.W.T+self.Ones@self.b.T # TODO

        return Z

    def backward(self, dLdZ):

        dZdA = self.W.T  # TODO
        dZdW = self.A  # TODO
        dZdb = self.Ones  # TODO

        dLdA = dLdZ @ dZdA.T  # TODO
        dLdW = dLdZ.T @ dZdW  # TODO
        dLdb = dLdZ.T @ dZdb  # TODO
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA

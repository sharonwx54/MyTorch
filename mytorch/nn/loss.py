import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = np.multiply(self.A - self.Y, self.A - self.Y) # TODO
        sse = np.ones((self.N, 1)).T.dot(se).dot(np.ones((self.C, 1)))  # TODO
        mse = sse/(2*self.N*self.C) # TODO

        return mse

    def backward(self):

        dLdA = (self.A - self.Y)/(self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = self.A.shape[0]  # TODO
        C = self.A.shape[1]  # TODO

        Ones_C = np.ones((C, 1))  # TODO
        Ones_N = np.ones((N, 1))  # TODO

        self.softmax = np.exp(self.A)/np.sum(np.exp(self.A), axis=1).reshape(-1, 1)  # TODO
        crossentropy = (-self.Y*np.log(self.softmax)).dot(Ones_C)  # TODO
        sum_crossentropy = Ones_N.T.dot(crossentropy)  # TODO

        L = sum_crossentropy / N

        return L

    def backward(self):

        dLdA = self.softmax-self.Y  # TODO

        return dLdA

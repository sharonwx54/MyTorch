import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r_xh = np.dot(self.x, self.Wrx.T)+self.brx+np.dot(h_prev_t, self.Wrh.T)+self.brh
        self.r = self.r_act.forward(self.r_xh)
        self.z_xh = np.dot(self.x, self.Wzx.T)+self.bzx + np.dot(h_prev_t, self.Wzh.T)+self.bzh
        self.z = self.z_act.forward(self.z_xh)
        self.n_xh = np.dot(self.x, self.Wnx.T)+self.bnx + self.r*(np.dot(h_prev_t, self.Wnh.T)+self.bnh)
        self.n = self.h_act.forward(self.n_xh)
        h_t = (1-self.z)*self.n + self.z*h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
    

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        self.x = self.x.reshape(self.d, 1)
        self.hidden = self.hidden.reshape(self.h, 1)
        delta = delta.reshape(self.h, 1)
        self.r = self.r.reshape(self.h, 1)
        self.n = self.n.reshape(self.h, 1)
        self.z = self.z.reshape(self.h, 1)
        bnh = self.bnh.reshape(self.h, 1)
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        dLdn = delta*(1-self.z)
        dLdz = delta*(-self.n + self.hidden)
        dLdr = dLdn*self.h_act.backward(self.n)*(self.Wnh@self.hidden + bnh)
        
        self.dLdr = dLdr*np.expand_dims(self.r_act.backward(), 1)
        self.dWrx = self.dLdr @ self.x.T
        self.dWrh = self.dLdr @ self.hidden.T
        self.dbrx = self.dLdr.reshape(self.h, )
        self.dbrh = self.dLdr.reshape(self.h, )

        self.dLdz = dLdz*np.expand_dims(self.z_act.backward(), 1)
        self.dWzx = self.dLdz@self.x.T
        self.dWzh = self.dLdz@self.hidden.T
        self.dbzx = self.dLdz.reshape(self.h, )
        self.dbzh = self.dLdz.reshape(self.h, )

        self.dLdn = dLdn*self.h_act.backward(self.n)
        self.dWnx = self.dLdn@self.x.T
        self.dWnh = (self.dLdn*self.r)@self.hidden.T
        self.dbnx = self.dLdn.reshape(self.h, )
        self.dbnh = (self.dLdn*self.r).reshape(self.h, )
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.
        dx = self.dLdn.T @ self.Wnx + self.dLdz.T @ self.Wzx + self.dLdr.T @ self.Wrx
        dh_prev_t = (delta*self.z).T + (self.dLdn*self.r).T @ self.Wnh + self.dLdz.T @ self.Wzh + self.dLdr.T @ self.Wrh
        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t


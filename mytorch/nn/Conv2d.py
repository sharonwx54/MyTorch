import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch, in_chan, in_h, in_w = self.A.shape
        out_w = (in_w - self.kernel_size)+1
        out_h = (in_h - self.kernel_size)+1
        Z = np.zeros([batch, self.out_channels, out_h, out_w])

        for b in range(batch):
            for i in range(self.out_channels):
                for j in range(out_h):
                    for k in range(out_w):
                        Z[b, i, j, k] = np.sum(self.A[b, :, j:j+self.kernel_size, k:k+self.kernel_size]*self.W[i])

                Z[b, i] += self.b[i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        #batch, out_chan, out_h, out_w = dLdZ.shape
        batch, in_chan, in_h, in_w = self.A.shape
        out_h = in_h - self.kernel_size + 1
        out_w = in_w - self.kernel_size + 1

        dLdA = np.zeros([batch, self.in_channels, in_h, in_w])

        for b in range(batch):
            for i in range(in_chan):
                for j in range(out_h):
                    for k in range(out_w):
                        for h in range(j, j+self.kernel_size):
                            for w in range(k, k+self.kernel_size):
                                dLdA[b, i, h, w] += sum(self.W[o, i, h-j, w-k]*dLdZ[b, o, j, k] for  o in range(self.out_channels))
        
        self.dLdb = np.sum(np.sum(dLdZ, axis=(0,2)), axis=1)

        self.dLdW = np.zeros(self.W.shape)
        for o in range(self.out_channels):
            for k1 in range(self.kernel_size):
                for k2 in range(self.kernel_size):
                    for i in range(in_chan):
                        for b in range(batch):
                            for h in range(out_h):
                                for w in range(out_w):
                                    self.dLdW[o, i, k1, k2] += self.A[b, i, k1+h, k2+w]*dLdZ[b, o, h, w]

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Call Conv2d_stride1
        Z_stride1 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        ds_dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(ds_dLdZ)

        return dLdA

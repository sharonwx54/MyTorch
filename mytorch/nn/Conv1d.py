# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch, _, in_size = self.A.shape
        out_size = (in_size - self.kernel_size) + 1
        Z = np.zeros([batch, self.out_channels, out_size])

        for b in range(batch):
            for i in range(self.out_channels):
                for j in range(out_size):
                    slide = self.A[b, :, j:(j+self.kernel_size)]
                    Z[b, i, j] = np.sum(slide*self.W[i, :, :])
                Z[b, i] += self.b[i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch, out_chan, out_size = dLdZ.shape
        _, in_chan, in_size = self.A.shape

        self.dLdW = np.zeros(self.W.shape)
        for b in range(batch):
            for k in range(self.kernel_size):
                for i in range(in_chan):
                     for o in range(out_chan):
                         self.dLdW[o, i, k] += sum([self.A[b, i, j+k]*dLdZ[b, o, j] for j in range(out_size)])

        dLdA = np.zeros([batch, in_chan, in_size])
        for b in range(batch):
            for i in range(in_chan):
                for j in range(out_size):
                    for k in range(j, j+self.kernel_size):
                        dLdA[b, i, k] += sum(self.W[o, i, k-j]*dLdZ[b, o, j] for o in range(out_chan))

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z_stride1 = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        ds_dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(ds_dLdZ)

        return dLdA

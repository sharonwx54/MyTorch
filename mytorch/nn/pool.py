import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch, self.in_chan, in_w, in_h = self.A.shape
        out_w = in_w - self.kernel + 1
        out_h = in_h - self.kernel + 1
        Z = np.zeros([batch, self.in_chan, out_w, out_h])
        self.poolidx = np.zeros([batch, self.in_chan, out_w, out_h], dtype=int)
        for w in range(out_w):
            for h in range(out_h):
                slice = self.A[:, :, w:w+self.kernel, h:h+self.kernel]
                Z[:, :, w, h] = np.amax(slice, axis=(2, 3))


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch, out_chan, out_w, out_h = dLdZ.shape
        dLdA = np.zeros(self.A.shape)
        for b in range(batch):
            for i in range(out_chan):
                for w in range(out_w):
                    for h in range(out_h):
                        slice = self.A[b, i, w:w+self.kernel, h:h+self.kernel]
                        w_idx, h_idx = np.where(slice==slice.max())
                        dLdA[b, i, w+w_idx, h+h_idx] += dLdZ[b, i, w, h]

        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch, self.in_chan, in_w, in_h = self.A.shape
        out_w = in_w - self.kernel + 1
        out_h = in_h - self.kernel + 1
        Z = np.zeros([batch, self.in_chan, out_w, out_h])
        self.poolidx = np.zeros([batch, self.in_chan, out_w, out_h], dtype=int)
        for b in range(batch):
            for i in range(self.in_chan):
                for w in range(out_w):
                    for h in range(out_h):
                        idx = np.mean(self.A[b, i, w:w+self.kernel, h:h+self.kernel])
                        Z[b, i, w, h] = idx
                        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch, _, out_w, out_h = dLdZ.shape
        dLdA = np.zeros(self.A.shape)
        for b in range(batch):
            for i in range(self.in_chan):
                for w in range(out_w):
                    for h in range(out_h):
                        dLdA[b, i, w:w+self.kernel, h:h+self.kernel] += dLdZ[b, i, w, h]/(self.kernel**2)

        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        A_stride1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(A_stride1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        ds_dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(ds_dLdZ)
        return dLdA




class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_stride1)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        ds_dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(ds_dLdZ)
        return dLdA

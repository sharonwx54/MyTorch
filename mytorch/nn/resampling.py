import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch, in_chan, in_w = A.shape
        out_w = in_w + ((self.upsampling_factor-1) * (in_w - 1))
        Z = np.zeros((batch, in_chan, out_w))
        Z[:, :, ::self.upsampling_factor] = A
        self.orig_idx = [i*(self.upsampling_factor) for i in range(in_w)]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = dLdZ[:, :, self.orig_idx]  # TODO

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.batch, self.in_chan, self.in_w = A.shape
        out_w = self.in_w//self.downsampling_factor + 1
        Z = np.zeros(out_w)
        Z = A[:, :, ::self.downsampling_factor]
    
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros((self.batch, self.in_chan, self.in_w))
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch, in_chan, h, w = A.shape
        Z = np.zeros((batch, in_chan, self.upsampling_factor*(h-1)+1,self.upsampling_factor*(w-1)+1), dtype=A.dtype)
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]  # TODO

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.batch, self.in_chan, self.h, self.w = A.shape
        out_h = self.h//self.downsampling_factor + 1
        out_w = self.w//self.downsampling_factor + 1
        Z = np.zeros((self.batch, self.in_chan, out_h, out_w))
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor] 

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros((self.batch, self.in_chan, self.h, self.w))
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA

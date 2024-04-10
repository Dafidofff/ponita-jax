from typing import Optional

import jax.numpy as jnp
from flax import linen as nn


class SeparableFiberBundleConv(nn.Module):
    """
    """
    in_channels: int
    out_channels: int
    kernel_dim: int
    apply_bias: bool = True
    groups: int = 1

    def setup(self):
        # Check arguments
        if self.groups == 1:
            self.depthwise_separable = False
        elif self.groups == self.in_channels and self.groups == self.out_channels:
            self.depthwise_separable = True
        else:
            raise ValueError('Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)')

        # Construct kernels
        self.kernel = nn.Dense(self.kernel_dim, use_bias=False)
        self.fiber_kernel = nn.Dense(int(self.in_channels * self.out_channels / self.groups), use_bias=False)
        
        # Construct bias
        if self.apply_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.out_channels,))
        else:
            self.bias = None
        
        # Automatic re-initialization
        self.callibrated = False

    def __call__(self, x, kernel_basis, fiber_kernel_basis, edge_index, training=True):
        """
        """
        # 1. Do the spatial convolution
        message = x[edge_index[0]] * self.kernel(kernel_basis) # [num_edges, num_ori, in_channels]
        x_1 = jnp.zeros_like(x).at[edge_index[1]].add(message)

        # 2. Fiber (spherical) convolution
        fiber_kernel = self.fiber_kernel(fiber_kernel_basis)
        if self.depthwise_separable:
            x_2 = jnp.einsum('boc,poc->bpc', x_1, fiber_kernel) / fiber_kernel.shape[-2]
        else:
            x_2 = jnp.einsum('boc,podc->bpd', x_1, fiber_kernel.reshape(-1, self.out_channels, self.in_channels)) / fiber_kernel.shape[-2]

        # 3. Re-callibrate the initializaiton
        # if training and not(self.callibrated):
        #     self.callibrate(x.std(), x_1.std(), x_2.std())

        # 4. Add bias
        x2 = x_2 if self.apply_bias else x_2 + self.bias
        return x2 

    def callibrate(self, std_in, std_1, std_2):
        print('Callibrating...')
        self.kernel.param = self.kernel.param * std_in/std_1
        self.fiber_kernel.param = self.fiber_kernel.param * std_1/std_2
        self.callibrated = ~self.callibrated



class FullyConnectedSeparableFiberBundleConv(SeparableFiberBundleConv):
    in_channels: int
    out_channels: int
    kernel_dim: int
    groups: int = 1
    apply_bias: Optional[bool] = True

    # def setup(self):
    #     # Check arguments
    #     if self.groups == 1:
    #         self.depthwise_separable = False
    #     elif self.groups == self.in_channels and self.groups == self.out_channels:
    #         self.depthwise_separable = True
    #     else:
    #         raise ValueError('Invalid option for groups, should be groups=1 or groups=in_channels=out_channels (depth-wise separable)')

    #     # Construct kernels
    #     self.kernel = nn.Dense(self.kernel_dim, use_bias=False)
    #     self.fiber_kernel = nn.Dense(int(self.in_channels * self.out_channels / self.groups), use_bias=False)
        
    #     # Construct bias
    #     if self.apply_bias:
    #         self.bias = self.param('bias', nn.initializers.zeros, (self.out_channels,))
    #     else:
    #         self.bias = None
        
    #     # Automatic re-initialization
    #     self.callibrated = False

    # def callibrate(self, std_in, std_1, std_2):
    #     print('Callibrating...')
    #     self.kernel.param = self.kernel.param * std_in/std_1
    #     self.fiber_kernel.param = self.fiber_kernel.param * std_1/std_2
    #     self.callibrated = ~self.callibrated

    def __call__(self, x, kernel_basis, fiber_kernel_basis, mask, training=True):
        
        # 1. Do spatial massage passing
        kernel = self.kernel(kernel_basis)
        if mask is None:
            x1 = jnp.einsum("bnoc, bmnoc -> bmoc", x, kernel)
        else:
            x1 = jnp.einsum("bnoc, bmnoc, bn -> bmoc", x, kernel, mask)
        
        # 2. Do fiber convolution
        fiber_kernel = self.fiber_kernel(fiber_kernel_basis)
        if self.depthwise_separable:
            x2 = jnp.einsum("bmoc, poc -> bmpc", x1, fiber_kernel)
            x2 = x2 / self.out_channels
        else:
            x2 = jnp.einsum("bmoc, podc -> bmpd", x1, fiber_kernel.unflatten(-1, (self.out_channels, self.in_channels)))
            x2 = x2 / (self.in_channels * self.out_channels)

        # 3. Re-callibrate the initializaiton
        # if self.callibrate:
        #     _mask = ... if mask is None else mask
        #     self._callibrate(*map(lambda x: x[_mask].std(), [x, x1, x2]))
        
        # 4. Add bias
        x2 = x2 if self.apply_bias else x2 + self.bias
        return x2 

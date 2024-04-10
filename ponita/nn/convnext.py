from flax import linen as nn

from ponita.nn.conv import SeparableFiberBundleConv, FullyConnectedSeparableFiberBundleConv


class SeparableFiberBundleConvNext(nn.Module):
    """
    """
    channels: int
    kernel_dim: int
    activation: nn.activation = nn.relu
    layer_scale_val: float = 1e-6
    widening_factor: int = 4
    fully_connected: bool = False

    def setup(self):
        if self.fully_connected:
            self.conv = FullyConnectedSeparableFiberBundleConv(
                self.channels, self.channels, self.kernel_dim, groups=self.channels
            )
        else:
            self.conv = SeparableFiberBundleConv(self.channels, self.channels, self.kernel_dim, groups=self.channels)

        self.linear_1 = nn.Dense(self.widening_factor * self.channels)
        self.linear_2 = nn.Dense(self.channels)
        if self.layer_scale_val is not None:
            self.layer_scale = self.param('layer_scale', nn.initializers.ones, (self.channels,)) * self.layer_scale_val
        self.norm = nn.normalization.LayerNorm()

    def __call__(self, x, kernel_basis, fiber_kernel_basis, edge_index):
        """
        """
        input = x
        x = self.conv(x, kernel_basis, fiber_kernel_basis, edge_index)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        x = x + input
        return x

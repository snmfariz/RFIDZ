import torch
from torch import nn


class CDAEComponent(nn.Module):
    """A single convolutional denoising block stack."""
    def __init__(
        self,
        depth: int,
        hidden_channels: int,
        kernel: tuple,
        padding: tuple,
        norm: bool,
        img_size: tuple,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.blocks = nn.ModuleList()
        for d in range(depth):
            in_size = hidden_channels if d > 0 else in_channels
            out_size = hidden_channels if d < depth - 1 else out_channels
            use_norm = (d < depth - 1) and norm
            use_tanh = (d == depth - 1)
            self.blocks.append(self._block(in_size, out_size, kernel, padding, use_norm, use_tanh))

    def _block(self, in_size: int, out_size: int, kernel: tuple, padding: tuple, use_norm: bool, use_tanh: bool) -> nn.Sequential:
        layers = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=kernel, stride=1, padding=padding))
        if use_norm:
            layers.append(nn.LayerNorm([out_size, self.img_size[0], self.img_size[1]]))
        layers.append(nn.Tanh() if use_tanh else nn.LeakyReLU(0.2))
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Model(nn.Module):
    """Stacked CDAE components with optional residual concatenation."""
    def __init__(
        self,
        depth: int,
        hidden_channels: int,
        kernel: tuple,
        norm: bool,
        img_size: tuple,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.kernel = kernel
        self.padding = (kernel[0] // 2, kernel[1] // 2)
        self.norm = norm
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.components = nn.ModuleList()

    def add_cdae(self, residual: bool = True, kernel: tuple | None = None, norm: bool | None = None):
        """Append a CDAE; freeze existing components."""
        kernel = self.kernel if kernel is None else kernel
        padding = self.padding if kernel is self.kernel else (kernel[0] // 2, kernel[1] // 2)
        norm = self.norm if norm is None else norm

        for comp in self.components:
            for p in comp.parameters():
                p.requires_grad = False

        in_ch = self.in_channels + self.out_channels if residual else self.in_channels
        self.components.append(
            CDAEComponent(
                depth=self.depth,
                hidden_channels=self.hidden_channels,
                kernel=kernel,
                padding=padding,
                norm=norm,
                img_size=self.img_size,
                in_channels=in_ch,
                out_channels=self.out_channels,
            )
        )

    def forward(self, x: torch.Tensor, n_components: int | None = None, return_intermediate: bool = False):
        """Forward through n components; optionally return intermediate tensors."""
        if n_components is None:
            n_components = len(self.components)
        assert n_components > 0 and len(self.components) >= n_components, "Add at least one CDAE before calling forward."

        x_inter = x
        z_inter = self.components[0](x)
        for comp in self.components[1:n_components]:
            x_inter = x_inter - z_inter
            z_inter = comp(torch.cat([x, x_inter], dim=1))

        if return_intermediate:
            return x_inter, z_inter
        return (x_inter - z_inter).detach()

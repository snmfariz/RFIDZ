from types import SimpleNamespace
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from simulate import ground_truth, noise
from model import Model

args = SimpleNamespace(
    # Output
    path_to_output=".",
    model_name="model",
    seed=0,

    # Simulation
    theta_bg_intensity=(0.1, 0.1),
    theta_gaussian_intensity=(0.1, 0.5),
    theta_n_channels=5,
    theta_channel_height=2,
    theta_channel_intensity=(0.5, 1.0),
    theta_frb_intensity=(0.1, 0.5),
    img_size=(256, 256),
    dataset_intensity_scale=(0.0, 1.0),  # (offset, scale)

    # Model
    n_cdae=9,
    depth=8,
    n_hidden=8,
    kernel=(13, 5),
    n_norm=3,

    # Training
    n_train=10000,
    n_valid=1000,
    batch_size=32,
    n_epochs_per_cdae=10,
    learning_rate=1e-4,
)

class SyntheticAKRDataset(Dataset):
    """Generate (noisy, clean) pairs on demand."""
    def __init__(self, n_samples, sim_args, seed=None):
        self.n = int(n_samples)
        self.a = sim_args
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.seed is not None:
            np.random.seed(self.seed + idx)
        clean, _, _, bg_noise = ground_truth(self.a)
        noisy, _ = noise(clean, self.a, bg_noise)
        x = torch.from_numpy(noisy[None, ...]).float()
        y = torch.from_numpy(clean[None, ...]).float()
        return x, y


def get_loss(criterion, model, x, y):
    """Compute loss between predicted and true noise."""
    x_inter, z_inter = model(x, return_intermediate=True)
    noise_gt = x_inter - y
    return criterion(z_inter, noise_gt)


def train_one_epoch(loader, model, criterion, opt, device):
    """Train for one epoch and return mean loss."""
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = get_loss(criterion, model, x, y)
        loss.backward()
        opt.step()
        total += loss.item()
        count += 1
    return total / max(1, count)


@torch.no_grad()
def validate_one_epoch(loader, model, criterion, device):
    """Validate for one epoch and return mean loss."""
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = get_loss(criterion, model, x, y)
        total += loss.item()
        count += 1
    return total / max(1, count)


def init_random(seed):
    """Seed NumPy and PyTorch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(a):
    """Create Model and move to device."""
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Model(
        depth=a.depth,
        hidden_channels=a.n_hidden,
        kernel=a.kernel,
        norm=True,
        img_size=a.img_size,
        in_channels=1,
        out_channels=1,
    ).to(device)
    return m, device



class Wrapped(nn.Module):
    """Add resize + per-image standardize/scale + inverse + conditional resize."""
    def __init__(self, core: nn.Module, offset: float, scale: float, model_hw: tuple[int, int]):
        super().__init__()
        self.core = core
        self.register_buffer("offset", torch.tensor(float(offset)))
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.model_h, self.model_w = int(model_hw[0]), int(model_hw[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2], x.shape[-1]
        if (in_h, in_w) != (self.model_h, self.model_w):
            x = F.interpolate(x, size=(self.model_h, self.model_w), mode="nearest")

        N = x.shape[0]
        flat = x.view(N, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        xz = (flat - mean) / std
        xz = (xz * self.scale) + self.offset
        xz = xz.clamp_(0.0, 1.0).view_as(x)

        y = self.core(xz)

        yf = y.view(N, -1)
        yf = (yf - self.offset) / self.scale
        y = (yf * std + mean).view_as(y)

        if (in_h, in_w) != (self.model_h, self.model_w):
            y = F.interpolate(y, size=(in_h, in_w), mode="nearest")
        return y


def export_onnx(core_model: nn.Module, a: SimpleNamespace, onnx_path: str):
    """Export ONNX with dynamic batch/H/W and built-in pre/post."""
    core_model.eval().to("cpu")
    offset, scale = getattr(a, "dataset_intensity_scale", (0.0, 1.0))
    wrapped = Wrapped(core_model, offset=offset, scale=scale, model_hw=a.img_size).eval()
    dummy = torch.randn(1, 1, a.img_size[0], a.img_size[1])
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch", 2: "in_h", 3: "in_w"},
                      "y": {0: "batch", 2: "in_h", 3: "in_w"}},
        opset_version=17,
    )
    print(f"Exported ONNX to {onnx_path}")


def start_training(a: SimpleNamespace):
    """Train components and export ONNX instead of saving .pt."""
    init_random(a.seed)

    sim_args = SimpleNamespace(
        theta_frb_intensity=a.theta_frb_intensity,
        theta_bg_intensity=a.theta_bg_intensity,
        theta_gaussian_intensity=a.theta_gaussian_intensity,
        theta_n_channels=a.theta_n_channels,
        theta_channel_height=a.theta_channel_height,
        theta_channel_intensity=a.theta_channel_intensity,
        img_size=a.img_size,
    )

    ds_train = SyntheticAKRDataset(a.n_train, sim_args, seed=a.seed)
    ds_valid = SyntheticAKRDataset(a.n_valid, sim_args, seed=None)

    use_cuda = torch.cuda.is_available()
    loader_train = DataLoader(
        ds_train, batch_size=a.batch_size, shuffle=True,
        num_workers=(2 if use_cuda else 0), pin_memory=use_cuda
    )
    loader_valid = DataLoader(
        ds_valid, batch_size=a.batch_size, shuffle=False,
        num_workers=(2 if use_cuda else 0), pin_memory=use_cuda
    )

    model, device = init_model(a)
    criterion = nn.MSELoss()

    for c in range(a.n_cdae):
        model.add_cdae(residual=(c > 0), norm=(c < a.n_norm))
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=a.learning_rate)
        for epoch in range(1, a.n_epochs_per_cdae + 1):
            tr = train_one_epoch(loader_train, model, criterion, opt, device)
            va = validate_one_epoch(loader_valid, model, criterion, device)
            print(f"[CDAE {c+1}/{a.n_cdae}] Epoch {epoch}/{a.n_epochs_per_cdae} | train {tr:.6f} | valid {va:.6f}")

    os.makedirs(a.path_to_output, exist_ok=True)
    onnx_path = os.path.join(a.path_to_output, f"{a.model_name}_prepost_conditionalHW.onnx")
    export_onnx(model, a, onnx_path)


if __name__ == "__main__":
    start_training(args)

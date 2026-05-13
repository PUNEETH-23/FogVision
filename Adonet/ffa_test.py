import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for FFA-Net inference. Install it with:\n"
        "  pip install torch torchvision\n"
        "Then run this script again."
    ) from exc


MODEL_PRESETS = {
    "its": "its_train_ffa_3_19.pk",
    "ots": "ots_train_ffa_3_19.pk",
}


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class PALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.pa(x)


class CALayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.ca(self.avg_pool(x))


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        return res + x


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super().__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        return self.gp(x) + x


class FFANet(nn.Module):
    def __init__(self, gps=3, blocks=19, conv=default_conv):
        super().__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size),
        ]

        self.pre = nn.Sequential(*pre_process)
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * gps, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.palayer = PALayer(self.dim)
        self.post = nn.Sequential(*post_process)

    def forward(self, x):
        x_in = x
        x = self.pre(x)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)

        weights = self.ca(torch.cat([res1, res2, res3], dim=1))
        weights = weights.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = (
            weights[:, 0, ...] * res1
            + weights[:, 1, ...] * res2
            + weights[:, 2, ...] * res3
        )
        out = self.palayer(out)
        out = self.post(out)
        return out + x_in


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a PyTorch state_dict.")

    cleaned = {}
    for key, value in checkpoint.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_model(model_path, device):
    model = FFANet(gps=3, blocks=19).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as exc:
        if exc.__class__.__name__ != "UnpicklingError":
            raise

        print(
            "PyTorch could not load this checkpoint in weights-only mode. "
            "Retrying with weights_only=False because this script is intended "
            "for a local FFA-Net checkpoint you trust."
        )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    state_dict = unwrap_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    if missing or unexpected:
        raise RuntimeError("Checkpoint did not match the FFA-Net architecture.")

    model.eval()
    return model


def read_image(image_path, max_size):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    if max_size and max(image_bgr.shape[:2]) > max_size:
        height, width = image_bgr.shape[:2]
        scale = max_size / max(height, width)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Resized input from {width}x{height} to {new_width}x{new_height}.")

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = original_rgb
    image = image_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    original = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, original


def print_tensor_stats(name, tensor):
    tensor = tensor.detach()
    print(
        f"{name}: min={tensor.min().item():.4f}, "
        f"max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}"
    )


def save_image(tensor, output_path):
    print_tensor_stats("Final output", tensor)
    image = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0).numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255.0).round().astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])


def match_color_statistics(output, reference):
    output_mean = output.mean(dim=(2, 3), keepdim=True)
    output_std = output.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    reference_mean = reference.mean(dim=(2, 3), keepdim=True)
    reference_std = reference.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    matched = (output - output_mean) / output_std * reference_std + reference_mean
    return matched.clamp(0.0, 1.0)


def safe_dehaze_output(output, original, strength):
    output = output.clamp(0.0, 1.0)
    output = match_color_statistics(output, original)
    blended = original * (1.0 - strength) + output * strength
    return blended.clamp(0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run FFA-Net dehazing on one image.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a custom FFA-Net PyTorch checkpoint. Overrides --preset.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(MODEL_PRESETS),
        default="its",
        help="Checkpoint preset to use when --model is not provided.",
    )
    parser.add_argument(
        "--input",
        default="test_frames/00026.JPG",
        help="Path to the hazy input image.",
    )
    parser.add_argument(
        "--output",
        default="ffa_output.jpg",
        help="Path where the dehazed image will be saved.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. Use auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help=(
            "Resize the longest image side before inference. "
            "Use 0 to keep the original size."
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Save the raw FFA-Net output without color correction or blending.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.35,
        help="Safe-mode blend strength. Lower values preserve more of the input image.",
    )
    return parser.parse_args()


def select_device(args):
    if args.cpu:
        print("Using device: CPU (--cpu was set).")
        return torch.device("cpu")

    if args.device == "cpu":
        print("Using device: CPU.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA ({device_name}).")
        return torch.device("cuda")

    if args.device == "cuda":
        raise SystemExit(
            "CUDA was requested, but this PyTorch install cannot see CUDA.\n"
            f"Installed torch: {torch.__version__}\n"
            f"torch.version.cuda: {torch.version.cuda}\n"
            "Install a CUDA-enabled PyTorch build, for example:\n"
            "  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )

    print(
        "Using device: CPU. CUDA is not available in this PyTorch install.\n"
        f"Installed torch: {torch.__version__}\n"
        "For GPU inference, install a CUDA-enabled PyTorch build, for example:\n"
        "  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )
    return torch.device("cpu")


def main():
    args = parse_args()
    device = select_device(args)

    model_path = Path(args.model or MODEL_PRESETS[args.preset])
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Using FFA-Net checkpoint: {model_path}")
    model = load_model(model_path, device)
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be between 0 and 1.")

    image_tensor, original_tensor = read_image(input_path, args.max_size)
    image_tensor = image_tensor.to(device)
    original_tensor = original_tensor.to(device)

    with torch.inference_mode():
        output = model(image_tensor)

    print_tensor_stats("Input tensor", image_tensor)
    if args.raw:
        final_output = output
        print("Saving raw FFA-Net output.")
    else:
        final_output = safe_dehaze_output(output, original_tensor, args.strength)
        print(f"Saving safe output with color matching and strength={args.strength:.2f}.")

    save_image(final_output, output_path)
    print(f"FFA-Net dehazed image saved to: {output_path}")


if __name__ == "__main__":
    main()

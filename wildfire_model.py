"""Архитектура и служебные функции CNN для прогноза пожара."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALL_FEATURES = (
    "elevation", "th", "vs", "tmmn", "tmmx", "sph",
    "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask",
)
FEATURE_LABELS = {
    "elevation": "Высота", "th": "Направление ветра",
    "vs": "Скорость ветра", "tmmn": "Минимальная температура",
    "tmmx": "Максимальная температура", "sph": "Влажность воздуха",
    "pr": "Осадки", "pdsi": "Индекс засухи PDSI",
    "NDVI": "Индекс растительности NDVI", "population": "Плотность населения",
    "erc": "Пожарная опасность ERC", "PrevFireMask": "Пожар в предыдущий день",
}
DATA_STATS = {
    "elevation": (0.0, 3141.0, 657.3003, 649.0147),
    "th": (0.0, 360.0, 190.3298, 72.5985),
    "vs": (0.0, 10.02, 3.8501, 1.4110),
    "tmmn": (253.15, 298.95, 281.0877, 8.9824),
    "tmmx": (253.15, 315.09, 295.1738, 9.8155),
    "sph": (0.0, 1.0, 0.00717, 0.00428),
    "pr": (0.0, 44.53, 1.7398, 4.4828),
    "pdsi": (-6.13, 7.876, -0.00527, 2.6823),
    "NDVI": (-9821.0, 9996.0, 5157.625, 2466.668),
    "population": (0.0, 2534.06, 25.5314, 154.7233),
    "erc": (0.0, 106.249, 37.3263, 20.8460),
    "PrevFireMask": (-1.0, 1.0, 0.0, 1.0),
    "FireMask": (-1.0, 1.0, 0.0, 1.0),
}
DATA_SIZE = 64
SAMPLE_SIZE = 32
DEFAULT_THRESHOLD = 0.5


class ReflectConv2d(nn.Module):
    """Свёртка без искусственного соединения противоположных краёв карты."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 1):
        super().__init__()
        # Вложенное имя conv сохраняет совместимость ключей старых checkpoint.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode="reflect",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WildfireCNN(nn.Module):
    """Архитектура из ноутбука: C -> 32 -> 64 -> 32 -> 1."""
    def __init__(self, in_channels: int = len(ALL_FEATURES)):
        super().__init__()
        self.net = nn.Sequential(
            ReflectConv2d(in_channels, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64), nn.ReLU(),
            ReflectConv2d(64, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
            # Возвращаем logits. Sigmoid применяется только при вычислении
            # вероятностей, а loss использует устойчивую реализацию with_logits.
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WildfireUNet(nn.Module):
    """Два уровня encoder/decoder, skip connections, выход — logits."""
    def __init__(self, in_channels=12):
        super().__init__()
        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1, padding_mode="reflect"),
                nn.GroupNorm(8, b), nn.ReLU(),
                nn.Conv2d(b, b, 3, padding=1, padding_mode="reflect"),
                nn.GroupNorm(8, b), nn.ReLU(),
            )
        self.enc1 = block(in_channels, 32)
        self.enc2 = block(32, 64)
        self.bridge = block(64, 128)
        self.dec2 = block(128 + 64, 64)
        self.dec1 = block(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bridge(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([
            F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2
        ], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1
        ], dim=1))
        return self.out(d1)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)
class UNetPlusPlus(nn.Module):

    def __init__(
        self,
        in_channels=12,
        num_classes=1
    ):

        super().__init__()

        filters = [64,128,256,512]

        self.pool = nn.MaxPool2d(2)

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

        self.conv0_0 = ConvBlock(
            in_channels,
            filters[0]
        )

        self.conv1_0 = ConvBlock(
            filters[0],
            filters[1]
        )

        self.conv2_0 = ConvBlock(
            filters[1],
            filters[2]
        )

        self.conv3_0 = ConvBlock(
            filters[2],
            filters[3]
        )

        self.conv0_1 = ConvBlock(
            filters[0] + filters[1],
            filters[0]
        )

        self.conv1_1 = ConvBlock(
            filters[1] + filters[2],
            filters[1]
        )

        self.conv2_1 = ConvBlock(
            filters[2] + filters[3],
            filters[2]
        )

        self.conv0_2 = ConvBlock(
            filters[0]*2 + filters[1],
            filters[0]
        )

        self.conv1_2 = ConvBlock(
            filters[1]*2 + filters[2],
            filters[1]
        )

        self.conv0_3 = ConvBlock(
            filters[0]*3 + filters[1],
            filters[0]
        )

        self.final = nn.Conv2d(
            filters[0],
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(
            self.pool(x0_0)
        )

        x2_0 = self.conv2_0(
            self.pool(x1_0)
        )

        x3_0 = self.conv3_0(
            self.pool(x2_0)
        )

        x0_1 = self.conv0_1(
            torch.cat([
                x0_0,
                self.up(x1_0)
            ], dim=1)
        )

        x1_1 = self.conv1_1(
            torch.cat([
                x1_0,
                self.up(x2_0)
            ], dim=1)
        )

        x2_1 = self.conv2_1(
            torch.cat([
                x2_0,
                self.up(x3_0)
            ], dim=1)
        )

        x0_2 = self.conv0_2(
            torch.cat([
                x0_0,
                x0_1,
                self.up(x1_1)
            ], dim=1)
        )

        x1_2 = self.conv1_2(
            torch.cat([
                x1_0,
                x1_1,
                self.up(x2_1)
            ], dim=1)
        )

        x0_3 = self.conv0_3(
            torch.cat([
                x0_0,
                x0_1,
                x0_2,
                self.up(x1_2)
            ], dim=1)
        )

        return self.final(x0_3)


def build_model(architecture, in_channels):
    if architecture == "unetpp":
        return UNetPlusPlus(in_channels)
    if architecture == "unet":
        return WildfireUNet(in_channels)
    if architecture == "cnn":
        return WildfireCNN(in_channels)
    raise ValueError(f"Unknown architecture: {architecture}")


LOSS_NAMES = ("source_bce_dice", "bce_dice", "focal", "bce_tversky", "focal_tversky")


class WeightedBCEDiceLoss(nn.Module):
    """Набор функций потерь для несбалансированной сегментации пожара."""

    def __init__(
        self,
        fire_weight: float = 10.0,
        loss_name: str = "bce_dice",
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        if loss_name not in LOSS_NAMES:
            raise ValueError(f"Неизвестный loss: {loss_name}")
        self.fire_weight = fire_weight
        self.loss_name = loss_name
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Считает loss только там, где целевая метка достоверна."""
        eps = 1e-7
        if valid_mask is None:
            valid_mask = torch.ones_like(target)
        valid_mask = valid_mask.to(dtype=logits.dtype)

        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            return logits.sum() * 0.0

        probabilities = torch.sigmoid(logits)
        if self.loss_name == "source_bce_dice":
            per_pixel = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none",
                pos_weight=logits.new_tensor(self.fire_weight),
            )
            bce = (per_pixel * valid_mask).sum() / valid_count
            p = probabilities * valid_mask
            y = target * valid_mask
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            return 0.7 * bce + 0.3 * dice
        weights = 1 + target * (self.fire_weight - 1)
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        bce = (bce_per_pixel * weights * valid_mask).sum() / valid_count

        true_positive = (probabilities * target * valid_mask).sum((1, 2, 3))
        false_positive = (
            probabilities * (1 - target) * valid_mask
        ).sum((1, 2, 3))
        false_negative = (
            (1 - probabilities) * target * valid_mask
        ).sum((1, 2, 3))

        if self.loss_name == "focal":
            probability_true = (
                probabilities * target + (1 - probabilities) * (1 - target)
            )
            focal_factor = (1 - probability_true).pow(self.focal_gamma)
            return (
                bce_per_pixel * weights * focal_factor * valid_mask
            ).sum() / valid_count

        if self.loss_name == "bce_dice":
            dice = 1 - (
                (2 * true_positive + eps)
                / (2 * true_positive + false_positive + false_negative + eps)
            ).mean()
            return bce + dice

        tversky = (
            (true_positive + eps)
            / (
                true_positive
                + self.tversky_alpha * false_positive
                + self.tversky_beta * false_negative
                + eps
            )
        )
        tversky_loss = 1 - tversky
        if self.loss_name == "bce_tversky":
            return bce + tversky_loss.mean()
        return tversky_loss.pow(self.focal_gamma).mean()


def iou_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> float:
    """Считает IoU, полностью исключая неопределённые пиксели."""
    prediction = prediction >= threshold
    target = target >= 0.5
    if valid_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = valid_mask >= 0.5
    intersection = (prediction & target & valid).sum((1, 2, 3)).float()
    union = ((prediction | target) & valid).sum((1, 2, 3)).float()
    return float(torch.where(union > 0, intersection / union, 1.0).mean().item())


def save_wildfire_checkpoint(path: str | Path, model: WildfireCNN,
                             channel_indices: Sequence[int], **metadata: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "architecture": "unetpp" if isinstance(model, UNetPlusPlus) else ("unet" if isinstance(model, WildfireUNet) else "cnn"), "model_state_dict": model.state_dict(),
        "channel_indices": list(channel_indices), "all_features": list(ALL_FEATURES),
        "threshold": float(metadata.pop("threshold", DEFAULT_THRESHOLD)), **metadata,
    }, path)
    return path


def load_wildfire_checkpoint(path: str | Path) -> tuple[WildfireCNN, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл весов: {path}")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    if "model_state_dict" in saved:
        state = saved["model_state_dict"]
        metadata = {key: value for key, value in saved.items() if key != "model_state_dict"}
        indices = tuple(metadata["channel_indices"])
    else:  # Совместимость с обычным state_dict.
        state = saved
        count = int(state["net.0.conv.weight"].shape[1])
        indices = tuple(range(count))
        metadata = {"channel_indices": list(indices), "threshold": DEFAULT_THRESHOLD}
    model = build_model(metadata.get("architecture", "cnn"), len(indices))
    model.input_size = metadata.get("input_size", 32)
    model.output_size = metadata.get("output_size", 32)
    model.load_state_dict(state)
    model.eval()
    return model, metadata


def predict_wildfire(model: WildfireCNN, sample: np.ndarray,
                     channel_indices: Sequence[int]) -> np.ndarray:
    expected = getattr(model, "input_size", 32)
    if sample.shape[-2:] != (expected, expected):
        raise ValueError(f"Модель ожидает {expected}x{expected}, получено {sample.shape[-2:]}")
    selected = np.ascontiguousarray(sample[list(channel_indices)], dtype=np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(selected).unsqueeze(0))
        size = getattr(model, "output_size", 32)
        top = (logits.shape[-2] - size) // 2
        left = (logits.shape[-1] - size) // 2
        return torch.sigmoid(logits)[0, 0, top:top+size, left:left+size].numpy()

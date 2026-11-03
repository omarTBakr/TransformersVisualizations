import torch

from config import get_settings
from transformer import VisionTransformer


def resolve_device(requested: str) -> torch.device:
    """Turn the configured device string into a usable torch.device, with fallbacks."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Notice: Configured device '{requested}' not available. Using 'cpu' instead.")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("Notice: Configured device 'mps' not available. Using 'cpu' instead.")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    settings = get_settings()
    m = settings.model
    torch.manual_seed(settings.training.seed)

    print("=" * 60)
    print("Vision Transformer configuration (from config.yml)")
    print("=" * 60)
    print(f"Model                : {m.name}")
    print(f"Image                : {m.image_size}×{m.image_size}×{m.in_channels}")
    print(f"Patches              : {m.patch_size}×{m.patch_size} → N = {m.n_patches} (+ [CLS])")
    print(f"d_model / heads      : {m.d_model} / {m.n_heads}")
    print(f"Encoder layers       : {m.n_encoder_layers}")
    print(f"MLP size             : {m.d_ff} ({m.activation})")
    print(f"Classes              : {m.num_classes}")
    print("=" * 60)

    device = resolve_device(settings.system.device)
    print(f"Running on device: {device}")

    model = VisionTransformer().to(device)
    print(f"Parameters           : {sum(p.numel() for p in model.parameters()):,}")

    # A dummy batch of images
    batch_size = 2
    images = torch.randn(batch_size, m.in_channels, m.image_size, m.image_size, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(images)

    expected = (batch_size, m.num_classes)
    print(f"Input shape          : {tuple(images.shape)}")
    print(f"Logits shape         : {tuple(logits.shape)}")
    if logits.shape != expected:
        raise RuntimeError(f"Expected logits of shape {expected}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Logits contain NaN or Inf values")
    print("Forward pass successful!")


if __name__ == "__main__":
    main()

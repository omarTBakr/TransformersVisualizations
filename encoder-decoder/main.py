import torch

from config import get_settings
from transformer import Transformer


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
    # 1. Load settings parsed from config.yml
    settings = get_settings()
    model_cfg = settings.model
    train_cfg = settings.training
    tok_cfg = settings.tokenizer

    torch.manual_seed(train_cfg.seed)

    print("=" * 60)
    print("Transformer Model Configuration (Loaded from config.yml)")
    print("=" * 60)
    print(f"Model Name           : {model_cfg.name}")
    print(f"Source Vocab Size    : {model_cfg.src_vocab_size}")
    print(f"Target Vocab Size    : {model_cfg.tgt_vocab_size}")
    print(f"d_model              : {model_cfg.d_model}")
    print(f"Attention Heads      : {model_cfg.n_heads}")
    print(f"Encoder Layers       : {model_cfg.n_encoder_layers}")
    print(f"Decoder Layers       : {model_cfg.n_decoder_layers}")
    print(f"Feed-Forward Dim     : {model_cfg.d_ff}")
    print(f"Dropout Rate         : {model_cfg.dropout}")
    print(f"Max Sequence Length  : {model_cfg.max_seq_len}")
    print(f"Positional Encoding  : {model_cfg.positional_encoding}")
    print(f"Activation           : {model_cfg.activation}")
    print(f"Share Embeddings     : {model_cfg.share_embeddings}")
    print(f"Batch Size (Train)   : {train_cfg.batch_size}")
    print(f"Optimizer / LR       : {train_cfg.optimizer.name} / {train_cfg.optimizer.lr}")
    print("=" * 60)

    # 2. Select device (falls back to CPU if the requested accelerator is unavailable)
    device = resolve_device(settings.system.device)
    print(f"Running on device: {device}")

    # 3. Instantiate model with parameters from config.yml
    model = Transformer().to(device)
    # parameters() de-duplicates shared (tied) tensors, so counts are accurate.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # 4. Create a dummy input batch; the last two source tokens of sample 1 are padding.
    batch_size, src_len, tgt_len = 2, 16, 12
    first_regular_id = max(tok_cfg.pad_token_id, tok_cfg.eos_token_id) + 1

    src = torch.randint(
        first_regular_id, model_cfg.src_vocab_size, (batch_size, src_len), device=device
    )
    tgt = torch.randint(
        first_regular_id, model_cfg.tgt_vocab_size, (batch_size, tgt_len), device=device
    )
    src[1, -2:] = tok_cfg.pad_token_id

    src_padding_mask = Transformer.make_padding_mask(src, tok_cfg.pad_token_id)
    tgt_padding_mask = Transformer.make_padding_mask(tgt, tok_cfg.pad_token_id)

    print("\nRunning test forward pass...")
    print(f"Source input shape : {tuple(src.shape)}")
    print(f"Target input shape : {tuple(tgt.shape)}")

    # 5. Forward pass (eval mode + no_grad: this is only a smoke test)
    model.eval()
    with torch.no_grad():
        output = model(
            src,
            tgt,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

    print(f"Output logits shape: {tuple(output.shape)}")
    expected_shape = (batch_size, tgt_len, model_cfg.tgt_vocab_size)
    if output.shape != expected_shape:
        raise RuntimeError(f"Expected output shape {expected_shape}, got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Output contains NaN or Inf values")
    print("Forward pass successful! All parameters from config.yml applied properly.")


if __name__ == "__main__":
    main()

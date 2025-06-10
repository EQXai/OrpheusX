import torch

def concat_with_fade(chunks, sample_rate: int = 24000, fade_ms: int = 20, gap_ms: int = 0, dtype: torch.dtype | None = torch.float32):
    """Concatenate audio tensors with optional cross-fade and silence gap.

    The helper now validates the input list and, when *dtype* is provided,
    casts every tensor to the requested dtype to avoid type mismatches.

    Parameters
    ----------
    chunks : list[torch.Tensor]
        List of 1D or 2D audio tensors to join. 2D tensors must be
        shaped ``(channels, samples)``.
    sample_rate : int, optional
        Sample rate of the audio, by default 24000.
    fade_ms : int, optional
        Duration of the crossfade in milliseconds, by default 20.
    gap_ms : int, optional
        Silence inserted between chunks in milliseconds, by default 0.
    dtype : torch.dtype | None, optional
        The desired data type of the output tensor. If None, the default dtype is used.
    Returns
    -------
    torch.Tensor
        The concatenated audio tensor.
    """
    if not chunks:
        # Return an empty tensor in the requested precision.
        return torch.tensor([], dtype=dtype or torch.float32)
    if len(chunks) == 1:
        return chunks[0].to(dtype) if dtype is not None else chunks[0]

    # Ensure all inputs are tensors of rank 1 or 2 and move to requested dtype.
    cleaned: list[torch.Tensor] = []
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, torch.Tensor):
            raise TypeError(f"Chunk {idx} is not a torch.Tensor")
        if ch.dim() not in (1, 2):
            raise ValueError("Each chunk must be a 1D (mono) or 2D (channels, samples) tensor")
        cleaned.append(ch.to(dtype) if dtype is not None else ch)

    chunks = cleaned

    fade_samples = int(sample_rate * fade_ms / 1000)
    gap_samples = int(sample_rate * gap_ms / 1000)
    output = chunks[0]
    for chunk in chunks[1:]:
        overlap = (
            min(fade_samples, output.shape[-1], chunk.shape[-1]) if fade_samples > 0 else 0
        )
        if gap_samples > 0:
            silence_shape = list(output.shape)
            silence_shape[-1] = gap_samples
            silence = torch.zeros(silence_shape, dtype=output.dtype, device=output.device)
            if overlap > 0:
                fade_out = torch.linspace(1.0, 0.0, overlap, device=output.device, dtype=output.dtype)
                fade_in = torch.linspace(0.0, 1.0, overlap, device=chunk.device, dtype=chunk.dtype)
                if output.dim() == 2:
                    fade_out = fade_out.unsqueeze(0)
                if chunk.dim() == 2:
                    fade_in = fade_in.unsqueeze(0)
                output = torch.cat([
                    output[..., :-overlap],
                    output[..., -overlap:] * fade_out,
                    silence,
                    chunk[..., :overlap] * fade_in,
                    chunk[..., overlap:],
                ], dim=-1)
            else:
                output = torch.cat([output, silence, chunk], dim=-1)
        else:
            if overlap > 0:
                fade_out = torch.linspace(1.0, 0.0, overlap, device=output.device, dtype=output.dtype)
                fade_in = torch.linspace(0.0, 1.0, overlap, device=chunk.device, dtype=chunk.dtype)
                if output.dim() == 2:
                    fade_out = fade_out.unsqueeze(0)
                if chunk.dim() == 2:
                    fade_in = fade_in.unsqueeze(0)
                mixed = output[..., -overlap:] * fade_out + chunk[..., :overlap] * fade_in
                output = torch.cat([
                    output[..., :-overlap],
                    mixed,
                    chunk[..., overlap:],
                ], dim=-1)
            else:
                output = torch.cat([output, chunk], dim=-1)

    return output

import torch

def concat_with_fade(chunks, sample_rate=24000, fade_ms=20):
    """Concatenate audio tensors with a short crossfade.


    Parameters
    ----------
    chunks : list[torch.Tensor]
        List of 1D or 2D audio tensors to join. 2D tensors must be
        shaped ``(channels, samples)``.
    sample_rate : int, optional
        Sample rate of the audio, by default 24000.
    fade_ms : int, optional
        Duration of the crossfade in milliseconds, by default 20.
    Returns
    -------
    torch.Tensor
        The concatenated audio tensor.
    """
    if not chunks:
        return torch.tensor([], dtype=torch.float32)
    if len(chunks) == 1:
        return chunks[0]

    fade_samples = int(sample_rate * fade_ms / 1000)
    output = chunks[0]
    for chunk in chunks[1:]:
        overlap = (
            min(fade_samples, output.shape[-1], chunk.shape[-1]) if fade_samples > 0 else 0
        )
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

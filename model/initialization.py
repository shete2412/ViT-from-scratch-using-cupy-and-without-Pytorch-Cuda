import cupy as cp


def xavier_normal(fan_in,fan_out):
    std = cp.sqrt(
        cp.float32(
            2.0
            / (
                fan_in
                + fan_out
            )
        )
    )

    return (
        cp.random.randn(
            fan_in,
            fan_out,
        ).astype(
            cp.float32
        )
        * std
    )


def small_normal(shape,std=0.02):
    """
    Small normal initialization for learnable embeddings.

    Used for:
        CLS token
        positional embedding

    """

    return (
        cp.random.randn(*shape).astype(cp.float32) * cp.float32(std)
    )

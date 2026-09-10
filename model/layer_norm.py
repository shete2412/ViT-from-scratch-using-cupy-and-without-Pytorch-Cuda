import cupy as cp


class LayerNorm:

    def __init__(
        self,
        embed_dim,
        eps=1e-5,
    ):

        self.embed_dim = (
            embed_dim
        )

        self.eps = cp.float32(
            eps
        )

        self.gamma = cp.ones(
            embed_dim,
            dtype=cp.float32,
        )

        self.beta = cp.zeros(
            embed_dim,
            dtype=cp.float32,
        )

        self.dgamma = None
        self.dbeta = None

        self.cache = None

    def forward(
        self,
        x,
    ):
        """
        Input:
            (B,T,D)

        Output:
            (B,T,D)
        """

        mean = cp.mean(
            x,
            axis=-1,
            keepdims=True,
        )

        centered = (x - mean)

        variance = cp.mean(
            centered ** 2,
            axis=-1,
            keepdims=True,
        )

        inv_std = (
            cp.float32(1.0) / cp.sqrt(variance + self.eps)
        )

        x_norm = (
            centered
            * inv_std
        )

        output = (
            self.gamma
            * x_norm
            + self.beta
        )

        self.cache = {
            "x_norm":
                x_norm,
            "inv_std":
                inv_std,
        }

        return output

    def backward(self,dout):
        x_norm = (
            self.cache[
                "x_norm"
            ]
        )

        inv_std = (
            self.cache[
                "inv_std"
            ]
        )

        self.dgamma = cp.sum(
            dout
            * x_norm,
            axis=(0, 1),
        )

        self.dbeta = cp.sum(
            dout,
            axis=(0, 1),
        )

        dx_norm = (
            dout
            * self.gamma
        )

        sum_dx_norm = cp.sum(
            dx_norm,
            axis=-1,
            keepdims=True,
        )

        sum_dx_norm_xnorm = cp.sum(
            dx_norm
            * x_norm,
            axis=-1,
            keepdims=True,
        )

        D = (
            self.embed_dim
        )

        dx = (
            inv_std
            / D
            * (
                D
                * dx_norm
                - sum_dx_norm
                - x_norm
                * sum_dx_norm_xnorm
            )
        )

        return dx

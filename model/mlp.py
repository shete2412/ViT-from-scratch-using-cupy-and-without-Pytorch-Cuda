"""
Input:
    (B,T,D)

Current:
    (B,197,256)

With MLP_RATIO = 4:
    hidden dimension M = 4D = 1024

Flow
----
    (B,T,256)
        ↓
    Linear W1
    256 -> 1024
        ↓
    (B,T,1024)
        ↓
    GELU
        ↓
    (B,T,1024)
        ↓
    Linear W2
    1024 -> 256
        ↓
    (B,T,256)
"""

import cupy as cp

from .initialization import (
    xavier_normal,
)


class MLP:

    def __init__(
        self,
        embed_dim,
        mlp_ratio=4,
    ):

        self.embed_dim = (
            embed_dim
        )

        self.hidden_dim = (
            mlp_ratio
            * embed_dim
        )

        self.W1 = (
            xavier_normal(
                self.embed_dim,
                self.hidden_dim,
            )
        )

        self.b1 = cp.zeros(
            self.hidden_dim,
            dtype=cp.float32,
        )

        self.W2 = (
            xavier_normal(
                self.hidden_dim,
                self.embed_dim,
            )
        )

        self.b2 = cp.zeros(
            self.embed_dim,
            dtype=cp.float32,
        )

        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

        self.cache = None

    @staticmethod
    def gelu(x):
        a = cp.float32(
            0.7978845608028654
        )

        b = cp.float32(
            0.044715
        )

        u = a * (x + b * x ** 3)

        return (
            cp.float32(
                0.5
            )
            * x
            * (
                cp.float32(
                    1.0
                )
                + cp.tanh(
                    u
                )
            )
        )

    @staticmethod
    def gelu_backward(
        x,
        dout,
    ):

        a = cp.float32(
            0.7978845608028654
        )

        b = cp.float32(
            0.044715
        )

        u = a * ( x + b * x ** 3)

        tanh_u = cp.tanh(u)

        du_dx = a * (cp.float32(1.0) + cp.float32( 3.0) * b * x ** 2)

        dgelu_dx = cp.float32( 0.5 ) * ( cp.float32(1.0) + tanh_u)+ cp.float32(0.5) * x * (cp.float32( 1.0 ) - tanh_u ** 2 ) * du_dx

        return (
            dout
            * dgelu_dx
        )

    def forward(self,x):

        # (B,T,D) @ (D,M) -> (B,T,M)
        z1 = (
            x
            @ self.W1
            + self.b1
        )

        # (B,T,M) -> (B,T,M)
        a1 = (
            self.gelu(
                z1
            )
        )

        # (B,T,M) @ (M,D) -> (B,T,D)
        output = (
            a1
            @ self.W2
            + self.b2
        )

        self.cache = {
            "x":
                x,
            "z1":
                z1,
            "a1":
                a1,
        }

        return output

    def backward(self,dout):

        x = (
            self.cache[
                "x"
            ]
        )

        z1 = (
            self.cache[
                "z1"
            ]
        )

        a1 = (
            self.cache[
                "a1"
            ]
        )

        B, T, D = (
            x.shape
        )

        M = (
            self.hidden_dim
        )

        a1_flat = (
            a1.reshape(
                B * T,
                M,
            )
        )

        dout_flat = (
            dout.reshape(
                B * T,
                D,
            )
        )

        self.dW2 = (
            a1_flat.T
            @ dout_flat
        )

        self.db2 = cp.sum(
            dout,
            axis=(0, 1),
        )

        da1 = (
            dout
            @ self.W2.T
        )

        dz1 = (
            self.gelu_backward(
                z1,
                da1,
            )
        )

        x_flat = (
            x.reshape(
                B * T,
                D,
            )
        )

        dz1_flat = (
            dz1.reshape(
                B * T,
                M,
            )
        )

        self.dW1 = (
            x_flat.T
            @ dz1_flat
        )

        self.db1 = cp.sum(
            dz1,
            axis=(0, 1),
        )

        return (
            dz1
            @ self.W1.T
        )

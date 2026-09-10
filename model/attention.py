import cupy as cp

from .initialization import (xavier_normal,)


class MultiHeadAttention:

    def __init__(self,embed_dim,num_heads):

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        if (embed_dim % num_heads != 0):
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.head_dim = embed_dim // num_heads

        self.Wq = (
            xavier_normal(
                embed_dim,
                embed_dim,
            )
        )

        self.Wk = (
            xavier_normal(
                embed_dim,
                embed_dim,
            )
        )

        self.Wv = (
            xavier_normal(
                embed_dim,
                embed_dim,
            )
        )

        self.Wo = (
            xavier_normal(
                embed_dim,
                embed_dim,
            )
        )

        self.dWq = None
        self.dWk = None
        self.dWv = None
        self.dWo = None

        self.cache = None

    @staticmethod
    def softmax(x):

        shifted = (x - cp.max(x, axis=-1, keepdims=True))

        exp_x = cp.exp(shifted)

        return (
            exp_x/ cp.sum(exp_x, axis=-1, keepdims=True)
        )

    def forward(self, x):

        B, T, D = x.shape

        H = self.num_heads

        d = self.head_dim

        #get Q,K,V matrices
        Q = x @ self.Wq

        K = x @ self.Wk

        V = x @ self.Wv

        # ---------------------------------------------
        # Split D into H heads.
        #
        # (B,T,D)
        #    ->
        # (B,H,T,d)
        # ---------------------------------------------

        Q = Q.reshape(B,T,H,d).transpose(0,2,1,3)

        K = K.reshape(B,T,H,d).transpose(0,2,1,3)

        V = V.reshape(B,T,H,d).transpose(0,2,1,3)

        # ---------------------------------------------
        # Scores:
        #
        # Q:   (B,H,T,d)
        # K^T: (B,H,d,T)
        #
        # output:
        #      (B,H,T,T)
        # ---------------------------------------------

        scores = Q @ K.transpose(0,1,3,2)

        scores = scores / cp.sqrt(cp.float32(d))

        attention_weights = self.softmax(scores)

        # ---------------------------------------------
        # Weighted V:
        #
        # (B,H,T,T) @ (B,H,T,d)
        #    ->
        # (B,H,T,d)
        # ---------------------------------------------

        head_output = attention_weights @ V

        # ---------------------------------------------
        # Merge heads.
        #
        # (B,H,T,d)
        #    ->
        # (B,T,D)
        # ---------------------------------------------

        concat = head_output.transpose(0,2,1,3).reshape(B,T,D)

        # Output projection:
        # (B,T,D) @ (D,D)
        #    ->
        # (B,T,D)
        output = concat @ self.Wo

        self.cache = {
            "x":
                x,
            "Q":
                Q,
            "K":
                K,
            "V":
                V,
            "attention":
                attention_weights,
            "concat":
                concat,
        }

        return output

    def backward(self,dout):
        cache = self.cache

        x = cache["x"]

        Q = cache["Q"]

        K = cache["K"]

        V = cache["V"]

        attention_weights = cache["attention"]

        concat = cache["concat"]

        B, T, D = x.shape

        H = self.num_heads

        d = self.head_dim

        # ---------------------------------------------
        # Output projection backward.
        # ---------------------------------------------

        concat_flat = concat.reshape(B * T,D)

        dout_flat = dout.reshape(B * T,D)

        self.dWo = concat_flat.T @ dout_flat

        dconcat = dout @ self.Wo.T

        # Split gradient back into heads:
        # (B,T,D) -> (B,H,T,d)
        dhead = dconcat.reshape(B,T,H,d,).transpose(0,2,1,3)

        # head_output = attention @ V
        dAttention = dhead @ V.transpose(0,1,3,2)

        dV = attention_weights.transpose(0,1,3,2) @ dhead

        # Softmax backward.
        softmax_dot = cp.sum(
            dAttention
            * attention_weights,
            axis=-1,
            keepdims=True,
        )

        dScores = (
            attention_weights * (dAttention - softmax_dot)
        )

        scale = cp.sqrt(cp.float32(d))

        # scores = QK^T / sqrt(d)
        dQ = (dScores @ K) / scale

        dK = (dScores.transpose(0,1,3,2) @ Q) / scale

        # (B,H,T,d) -> (B,T,D)
        dQ = dQ.transpose(0,2,1,3).reshape(B,T,D)

        dK = dK.transpose(0,2,1,3).reshape(B,T,D)

        dV = dV.transpose(0,2,1,3).reshape(B,T,D)

        x_flat = x.reshape(B * T,D)

        dQ_flat = dQ.reshape(B * T,D)

        dK_flat = dK.reshape(B * T,D)

        dV_flat = dV.reshape(B * T,D)

        self.dWq = x_flat.T @ dQ_flat

        self.dWk = x_flat.T @ dK_flat

        self.dWv = x_flat.T @ dV_flat

        # x contributed to Q, K and V,
        # therefore all three gradients add.
        dx_q = dQ @ self.Wq.T

        dx_k = dK @ self.Wk.T

        dx_v = dV @ self.Wv.T

        return (dx_q + dx_k + dx_v)

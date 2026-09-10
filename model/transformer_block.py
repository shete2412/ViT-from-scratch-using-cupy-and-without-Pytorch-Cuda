from .layer_norm import LayerNorm
from .attention import MultiHeadAttention
from .mlp import MLP
from .dropout import Dropout


class TransformerBlock:

    def __init__(self, embed_dim, num_heads, mlp_ratio, eps, dropout_rate):
        self.ln1 = LayerNorm(embed_dim, eps)
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Residual-branch dropout after attention.
        self.attention_dropout = Dropout(dropout_rate)

        self.ln2 = LayerNorm(embed_dim, eps)
        self.mlp = MLP(embed_dim, mlp_ratio)

        # Residual-branch dropout after MLP.
        self.mlp_dropout = Dropout(dropout_rate)

    def forward(self, x, training=True):
        """
        Input:
            x:
                (B,T,D)

        Output:
            x2:
                (B,T,D)
        """

        # =============================================
        # ATTENTION SUBLAYER
        #
        # x1 = x + Dropout(MHA(LN1(x)))
        # =============================================

        norm1 = self.ln1.forward(x)
        attention_output = self.attention.forward(norm1)
        attention_output = self.attention_dropout.forward(attention_output, training=training)

        # Residual ADD, not concatenation.
        x1 = x + attention_output

        # =============================================
        # MLP SUBLAYER
        #
        # x2 = x1 + Dropout(MLP(LN2(x1)))
        # =============================================

        norm2 = self.ln2.forward(x1)
        mlp_output = self.mlp.forward(norm2)
        mlp_output = self.mlp_dropout.forward(mlp_output, training=training)

        x2 = x1 + mlp_output

        return x2

    def backward(self, dout):
        """
        Backward in exact reverse order.

        Input:
            dout:
                (B,T,D)

        Output:
            dx:
                (B,T,D)
        """

        # =============================================
        # SECOND RESIDUAL
        #
        # x2 = x1 + Dropout(MLP(LN2(x1)))
        # =============================================

        dx1_residual = dout

        dmlp_output = self.mlp_dropout.backward(dout)
        dnorm2 = self.mlp.backward(dmlp_output)
        dx1_ln = self.ln2.backward(dnorm2)

        dx1 = dx1_residual + dx1_ln

        # =============================================
        # FIRST RESIDUAL
        #
        # x1 = x + Dropout(MHA(LN1(x)))
        # =============================================

        dx_residual = dx1

        dattention = self.attention_dropout.backward(dx1)
        dnorm1 = self.attention.backward(dattention)
        dx_ln = self.ln1.backward(dnorm1)

        dx = dx_residual + dx_ln

        return dx
import cupy as cp

from .initialization import (
    xavier_normal,
)


class Classifier:

    def __init__(
        self,
        embed_dim,
        num_classes,
    ):

        self.embed_dim = (
            embed_dim
        )

        self.num_classes = (
            num_classes
        )

        self.W = (
            xavier_normal(
                embed_dim,
                num_classes,
            )
        )

        self.b = cp.zeros(
            num_classes,
            dtype=cp.float32,
        )

        self.dW = None
        self.db = None

        self.cache = None

    def forward(self,transformer_output):
        """
        Input:
            (B,T,D)

        Output:
            logits:
                (B,C)
        """

        cls_features = (
            transformer_output[:,0,:]
        )

        logits = (cls_features @ self.W + self.b)

        self.cache = {
            "cls_features":
                cls_features,
            "transformer_shape":
                transformer_output.shape,
        }

        return logits

    def backward(self,dlogits):

        cls_features = (
            self.cache[
                "cls_features"
            ]
        )

        transformer_shape = (
            self.cache[
                "transformer_shape"
            ]
        )

        self.dW = (
            cls_features.T
            @ dlogits
        )

        self.db = cp.sum(
            dlogits,
            axis=0,
        )

        dcls = (
            dlogits
            @ self.W.T
        )

        dtransformer = cp.zeros(
            transformer_shape,
            dtype=cp.float32,
        )

        dtransformer[:,0,:] = dcls

        return dtransformer

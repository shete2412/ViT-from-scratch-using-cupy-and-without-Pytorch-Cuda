import cupy as cp

from .initialization import small_normal


class TokenEmbedding:

    def __init__(self, num_tokens, embed_dim):
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Learnable CLS:
        # (1,D)
        self.cls_token = small_normal((1, self.embed_dim), std=0.02)

        # Learnable absolute positional embedding:
        # (T,D)
        self.pos_embed = small_normal((self.num_tokens, self.embed_dim), std=0.02)

        self.dcls_token = None
        self.dpos_embed = None

    def forward(self, patch_embeddings):
        """
        Input:
            patch_embeddings:
                (B,196,D)

        Output:
            tokens:
                (B,197,D)
        """

        B = patch_embeddings.shape[0]

        # (1,D) -> (B,1,D)
        cls_batch = cp.broadcast_to(self.cls_token[None, :, :], (B, 1, self.embed_dim))

        # (B,1,D) + (B,196,D)
        #    ->
        # (B,197,D)
        tokens = cp.concatenate((cls_batch, patch_embeddings), axis=1)

        # Position values are added, not concatenated.
        #
        # (B,197,D) + (1,197,D)
        #    ->
        # (B,197,D)
        tokens = tokens + self.pos_embed[None, :, :]

        return tokens

    def backward(self, dtokens):
        """
        Input:
            dtokens:
                (B,197,D)

        Gradients:
            dpos_embed:
                (197,D)

            dcls_token:
                (1,D)

        Output:
            gradients for patch tokens:
                (B,196,D)
        """

        # pos_embed is shared across all B images.
        self.dpos_embed = cp.sum(dtokens, axis=0)

        # CLS token is shared across all B images.
        self.dcls_token = cp.sum(dtokens[:, 0, :], axis=0, keepdims=True)

        return dtokens[:, 1:, :]
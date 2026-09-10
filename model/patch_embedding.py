import cupy as cp

from .initialization import xavier_normal


class PatchEmbedding:

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side * self.patches_per_side
        self.patch_vector_dim = patch_size * patch_size * in_channels

        # W_patch:
        #     (patch_vector_dim, embed_dim)
        #
        # Current:
        #     (768, 256)
        self.W = xavier_normal(self.patch_vector_dim, self.embed_dim)

        # Bias:
        #     (embed_dim,)
        self.b = cp.zeros(self.embed_dim, dtype=cp.float32)

        self.dW = None
        self.db = None
        self.cache = None

    def patchify(self, images):
        """
        Divide each image into non-overlapping patches.

        Input:
            images:
                (B, image_size, image_size, channels)

            Current:
                (B,224,224,3)

        Output:
            patches:
                (B, num_patches, patch_vector_dim)

            Current:
                (B,196,768)
        """

        B, H, W, C = images.shape

        if H != self.image_size or W != self.image_size or C != self.in_channels:
            raise ValueError(f"Unexpected image shape: {images.shape}")

        # ----------------------------------------------------
        # Separate H and W into:
        #
        # number of patch rows
        # patch height
        # number of patch columns
        # patch width
        #
        # Current:
        # (B,224,224,3)
        #    ->
        # (B,14,16,14,16,3)
        # ----------------------------------------------------
        patches = images.reshape(B, self.patches_per_side, self.patch_size, self.patches_per_side, self.patch_size, self.in_channels)

        # ----------------------------------------------------
        # Move patch row/column axes together.
        #
        # (B,14,16,14,16,3)
        #    ->
        # (B,14,14,16,16,3)
        # ----------------------------------------------------
        patches = patches.transpose(0, 1, 3, 2, 4, 5)

        # ----------------------------------------------------
        # Flatten each 16x16x3 patch.
        #
        # (B,14,14,16,16,3)
        #    ->
        # (B,196,768)
        # ----------------------------------------------------
        patches = patches.reshape(B, self.num_patches, self.patch_vector_dim)

        return patches

    def forward(self, images):
        """
        Complete patch embedding.

        Input:
            images:
                current shape (B,224,224,3)

        Output:
            patch_embeddings:
                current shape (B,196,256)
        """

        patches = self.patchify(images)

        # Every patch independently goes through the SAME linear projection.
        #
        # (B,196,768) @ (768,256)
        #     ->
        # (B,196,256)
        output = patches @ self.W + self.b

        self.cache = {"patches": patches}

        return output

    def backward(self, dout):
        """
        Backward through the linear patch projection.

        Input:
            dout:
                (B,196,256)

        Gradients produced:
            dW:
                (768,256)

            db:
                (256,)

        Output:
            dpatches:
                (B,196,768)
        """

        patches = self.cache["patches"]

        B, P, patch_dim = patches.shape

        patches_flat = patches.reshape(B * P, patch_dim)
        dout_flat = dout.reshape(B * P, self.embed_dim)

        self.dW = patches_flat.T @ dout_flat
        self.db = cp.sum(dout, axis=(0, 1))

        dpatches = dout @ self.W.T

        return dpatches
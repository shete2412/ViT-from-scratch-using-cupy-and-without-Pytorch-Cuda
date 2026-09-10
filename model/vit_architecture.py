from .patch_embedding import PatchEmbedding
from .token_embedding import TokenEmbedding
from .dropout import Dropout
from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm
from .classifier import Classifier
from .optimizer import AdamW


class VisionTransformer:

    def __init__(
        self,
        num_classes,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_heads=8,
        num_blocks=3,
        mlp_ratio=4,
        eps=1e-5,
        dropout_rate=0.10,
        embedding_dropout_rate=0.10,
    ):

        # ====================================================
        # ARCHITECTURE SETTINGS
        # ====================================================

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.D = embed_dim
        self.H = num_heads

        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = mlp_ratio * embed_dim

        self.num_classes = num_classes

        self.dropout_rate = float(dropout_rate)
        self.embedding_dropout_rate = float(embedding_dropout_rate)

        # ====================================================
        # MODULE 1: PATCH EMBEDDING
        # ====================================================

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Useful architecture information.
        self.patches_per_side = self.patch_embedding.patches_per_side
        self.num_patches = self.patch_embedding.num_patches
        self.patch_vector_dim = self.patch_embedding.patch_vector_dim

        self.num_tokens = self.num_patches + 1

        # ====================================================
        # MODULE 2: CLS + POSITION
        # ====================================================

        self.token_embedding = TokenEmbedding(
            num_tokens=self.num_tokens,
            embed_dim=self.D,
        )

        # ====================================================
        # MODULE 3: EMBEDDING DROPOUT
        #
        # It runs AFTER:
        #     patch embedding
        #     CLS token
        #     positional embedding
        #
        # It runs BEFORE:
        #     Transformer Block 0
        #
        # Input/output:
        #     (B,197,D) -> (B,197,D)
        # ====================================================

        self.embedding_dropout = Dropout(self.embedding_dropout_rate)

        # ====================================================
        # MODULE 4: N TRANSFORMER BLOCKS
        # ====================================================

        self.blocks = [
            TransformerBlock(
                embed_dim=self.D,
                num_heads=self.H,
                mlp_ratio=self.mlp_ratio,
                eps=eps,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_blocks)
        ]

        # ====================================================
        # MODULE 5: FINAL LAYERNORM
        # ====================================================

        self.final_ln = LayerNorm(self.D, eps)

        # ====================================================
        # MODULE 6: CLASSIFIER
        # ====================================================

        self.classifier = Classifier(self.D, self.num_classes)

        # ====================================================
        # MODULE 7: OPTIMIZER
        # ====================================================

        self.optimizer = AdamW()

        # Expose old familiar parameter names so checkpoint/test
        # code can remain easy to write.
        self._create_parameter_aliases()

    # ========================================================
    # COMPLETE MODEL FORWARD
    # ========================================================

    def forward(self, images, training=True):
        """
        Complete ViT forward pass.

        INPUT:
            images:
                (B,224,224,3)

        OUTPUT:
            logits:
                (B,num_classes)

        training=True:
            embedding dropout ON
            block dropout ON

        training=False:
            all dropout OFF
        """

        # ----------------------------------------------------
        # 1. PATCHIFY + PATCH PROJECTION
        #
        # (B,224,224,3)
        #     ->
        # (B,196,768)
        #     ->
        # (B,196,D)
        # ----------------------------------------------------

        x = self.patch_embedding.forward(images)

        # ----------------------------------------------------
        # 2. CLS + POSITIONAL EMBEDDING
        #
        # (B,196,D)
        #     ->
        # (B,197,D)
        # ----------------------------------------------------

        x = self.token_embedding.forward(x)

        # ----------------------------------------------------
        # 3. EMBEDDING DROPOUT
        #
        # (B,197,D)
        #     ->
        # (B,197,D)
        # ----------------------------------------------------

        x = self.embedding_dropout.forward(x, training=training)

        # ----------------------------------------------------
        # 4. ALL TRANSFORMER BLOCKS
        #
        # Each block:
        #     (B,197,D) -> (B,197,D)
        # ----------------------------------------------------

        for block in self.blocks:
            x = block.forward(x, training=training)

        # ----------------------------------------------------
        # 5. FINAL LAYERNORM
        #
        # (B,197,D) -> (B,197,D)
        # ----------------------------------------------------

        x = self.final_ln.forward(x)

        # ----------------------------------------------------
        # 6. CLS CLASSIFIER
        #
        # CLS:
        #     (B,D)
        #
        # logits:
        #     (B,C)
        # ----------------------------------------------------

        logits = self.classifier.forward(x)

        return logits

    # ========================================================
    # COMPLETE MODEL BACKWARD
    # ========================================================

    def backward(self, dlogits):

        # Classifier:
        # (B,C) -> (B,197,D)
        dx = self.classifier.backward(dlogits)

        # Final LN:
        # (B,197,D) -> (B,197,D)
        dx = self.final_ln.backward(dx)

        # Reverse all Transformer blocks automatically.
        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # Backward through the SAME embedding dropout mask
        # generated by the training forward pass.
        dx = self.embedding_dropout.backward(dx)

        # Separate CLS/position gradients from patch-token gradients.
        dx = self.token_embedding.backward(dx)

        # Patch-projection gradients.
        dpatches = self.patch_embedding.backward(dx)

        return dpatches

    # ========================================================
    # PARAMETERS + GRADIENTS FOR ADAMW
    # ========================================================

    def named_parameters_and_gradients(self):
        """
        Returns tuples:

            (
                name,
                parameter,
                gradient,
                use_weight_decay,
            )

        Dropout has NO trainable parameters, so embedding dropout
        does not appear here.
        """

        patch = self.patch_embedding
        token = self.token_embedding

        parameters = [
            # Patch projection.
            (
                "W_patch",
                patch.W,
                patch.dW,
                True,
            ),
            (
                "b_patch",
                patch.b,
                patch.db,
                False,
            ),

            # CLS and positional embedding.
            (
                "cls_token",
                token.cls_token,
                token.dcls_token,
                False,
            ),
            (
                "pos_embed",
                token.pos_embed,
                token.dpos_embed,
                False,
            ),
        ]

        for block_idx, block in enumerate(self.blocks):

            parameters.extend(
                [
                    # LN1.
                    (
                        f"ln_gamma_{block_idx}_0",
                        block.ln1.gamma,
                        block.ln1.dgamma,
                        False,
                    ),
                    (
                        f"ln_beta_{block_idx}_0",
                        block.ln1.beta,
                        block.ln1.dbeta,
                        False,
                    ),

                    # LN2.
                    (
                        f"ln_gamma_{block_idx}_1",
                        block.ln2.gamma,
                        block.ln2.dgamma,
                        False,
                    ),
                    (
                        f"ln_beta_{block_idx}_1",
                        block.ln2.beta,
                        block.ln2.dbeta,
                        False,
                    ),

                    # Attention.
                    (
                        f"Wq_{block_idx}",
                        block.attention.Wq,
                        block.attention.dWq,
                        True,
                    ),
                    (
                        f"Wk_{block_idx}",
                        block.attention.Wk,
                        block.attention.dWk,
                        True,
                    ),
                    (
                        f"Wv_{block_idx}",
                        block.attention.Wv,
                        block.attention.dWv,
                        True,
                    ),
                    (
                        f"Wo_{block_idx}",
                        block.attention.Wo,
                        block.attention.dWo,
                        True,
                    ),

                    # MLP.
                    (
                        f"W1_{block_idx}",
                        block.mlp.W1,
                        block.mlp.dW1,
                        True,
                    ),
                    (
                        f"b1_{block_idx}",
                        block.mlp.b1,
                        block.mlp.db1,
                        False,
                    ),
                    (
                        f"W2_{block_idx}",
                        block.mlp.W2,
                        block.mlp.dW2,
                        True,
                    ),
                    (
                        f"b2_{block_idx}",
                        block.mlp.b2,
                        block.mlp.db2,
                        False,
                    ),
                ]
            )

        parameters.extend(
            [
                # Final LayerNorm.
                (
                    "final_ln_gamma",
                    self.final_ln.gamma,
                    self.final_ln.dgamma,
                    False,
                ),
                (
                    "final_ln_beta",
                    self.final_ln.beta,
                    self.final_ln.dbeta,
                    False,
                ),

                # Classifier.
                (
                    "W_classifier",
                    self.classifier.W,
                    self.classifier.dW,
                    True,
                ),
                (
                    "b_classifier",
                    self.classifier.b,
                    self.classifier.db,
                    False,
                ),
            ]
        )

        return parameters

    # ========================================================
    # OPTIMIZER WRAPPER
    # ========================================================

    def adamw_step(
        self,
        learning_rate,
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.05,
    ):
        """
        Update all trainable parameters once.
        """

        self.optimizer.step(
            self.named_parameters_and_gradients(),
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def state_dict(self):
        """
        Return model parameters using the SAME key names as your
        original single-file model.

        Embedding dropout has no parameter, so it does not need
        a tensor in state_dict().
        """

        state = {
            "W_patch": self.patch_embedding.W,
            "b_patch": self.patch_embedding.b,
            "cls_token": self.token_embedding.cls_token,
            "pos_embed": self.token_embedding.pos_embed,
            "final_ln_gamma": self.final_ln.gamma,
            "final_ln_beta": self.final_ln.beta,
            "W_classifier": self.classifier.W,
            "b_classifier": self.classifier.b,
        }

        for block_idx, block in enumerate(self.blocks):
            state[f"Wq_{block_idx}"] = block.attention.Wq
            state[f"Wk_{block_idx}"] = block.attention.Wk
            state[f"Wv_{block_idx}"] = block.attention.Wv
            state[f"Wo_{block_idx}"] = block.attention.Wo

            state[f"W1_{block_idx}"] = block.mlp.W1
            state[f"b1_{block_idx}"] = block.mlp.b1
            state[f"W2_{block_idx}"] = block.mlp.W2
            state[f"b2_{block_idx}"] = block.mlp.b2

            state[f"ln_gamma_{block_idx}_0"] = block.ln1.gamma
            state[f"ln_beta_{block_idx}_0"] = block.ln1.beta
            state[f"ln_gamma_{block_idx}_1"] = block.ln2.gamma
            state[f"ln_beta_{block_idx}_1"] = block.ln2.beta

        return state

    # ========================================================
    # COMPATIBILITY PARAMETER ALIASES
    # ========================================================

    def _create_parameter_aliases(self):
        """
        Keep familiar names such as:
            model.W_patch
            model.Wq[0]
            model.W1[0]

        This makes future checkpoint/test code easier and preserves
        the naming style of your original implementation.
        """

        self.W_patch = self.patch_embedding.W
        self.b_patch = self.patch_embedding.b

        self.cls_token = self.token_embedding.cls_token
        self.pos_embed = self.token_embedding.pos_embed

        self.Wq = [block.attention.Wq for block in self.blocks]
        self.Wk = [block.attention.Wk for block in self.blocks]
        self.Wv = [block.attention.Wv for block in self.blocks]
        self.Wo = [block.attention.Wo for block in self.blocks]

        self.W1 = [block.mlp.W1 for block in self.blocks]
        self.b1 = [block.mlp.b1 for block in self.blocks]
        self.W2 = [block.mlp.W2 for block in self.blocks]
        self.b2 = [block.mlp.b2 for block in self.blocks]

        self.ln_gamma = [
            [block.ln1.gamma, block.ln2.gamma]
            for block in self.blocks
        ]

        self.ln_beta = [
            [block.ln1.beta, block.ln2.beta]
            for block in self.blocks
        ]

        self.final_ln_gamma = self.final_ln.gamma
        self.final_ln_beta = self.final_ln.beta

        self.W_classifier = self.classifier.W
        self.b_classifier = self.classifier.b

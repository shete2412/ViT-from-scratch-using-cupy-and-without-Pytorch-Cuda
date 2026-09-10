"""
MANUAL ADAMW OPTIMIZER
======================
For every parameter gradient g:

    m_t = beta1*m_(t-1) + (1-beta1)*g
    v_t = beta2*v_(t-1) + (1-beta2)*g^2

Bias correction:

    m_hat = m_t / (1-beta1^t)
    v_hat = v_t / (1-beta2^t)

Adam update:

    theta -= lr * m_hat / (sqrt(v_hat) + eps)

For actual weight matrices only, decoupled weight decay is also:

    theta -= lr * weight_decay * theta

Weight decay IS used for:
    W_patch
    Wq/Wk/Wv/Wo
    W1/W2
    W_classifier

Weight decay is NOT used for:
    biases
    LayerNorm gamma/beta
    CLS
    positional embedding
"""

import cupy as cp


class AdamW:

    def __init__(self):
        self.step_count = 0
        self.first_moment = {}
        self.second_moment = {}

    def step(
        self,
        named_parameters,
        learning_rate,
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.05,
    ):
        self.step_count += 1
        t = self.step_count

        lr = cp.float32(learning_rate)
        beta1 = cp.float32(beta1)
        beta2 = cp.float32(beta2)
        adam_eps = cp.float32(adam_eps)
        weight_decay = cp.float32(weight_decay)
        one = cp.float32(1.0)

        bias_correction1 = one - beta1 ** t
        bias_correction2 = one - beta2 ** t

        for name, parameter, gradient, use_weight_decay in named_parameters:

            if gradient is None:
                raise RuntimeError(
                    f"Gradient is None for parameter {name}. Did backward run?"
                )

            # Initialize first and second moments for this parameter.
            if name not in self.first_moment:
                self.first_moment[name] = cp.zeros_like(parameter)
                self.second_moment[name] = cp.zeros_like(parameter)

            m = self.first_moment[name]
            v = self.second_moment[name]

            # First moment update.
            m *= beta1
            m += (one - beta1) * gradient

            # Second moment update.
            v *= beta2
            v += (one - beta2) * (gradient * gradient)

            # Bias correction.
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Decoupled weight decay.
            if use_weight_decay:
                parameter -= lr * weight_decay * parameter

            # Adam update.
            parameter -= lr * m_hat / (cp.sqrt(v_hat) + adam_eps)


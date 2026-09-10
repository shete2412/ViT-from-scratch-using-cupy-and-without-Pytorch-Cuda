import cupy as cp


class Dropout:

    def __init__(self,probability=0.10):

        if not (0.0 <= probability < 1.0):
            raise ValueError(
                "Dropout probability must be in [0, 1)."
            )

        self.probability = float(probability)

        # Training mask saved for backward.
        self.mask = None

    def forward(self,x,training=True):
        if (not training or self.probability == 0.0):
            self.mask = None
            return x

        keep_probability = cp.float32(1.0 - self.probability)

        self.mask = (
            (cp.random.random(x.shape) < keep_probability).astype(cp.float32) / keep_probability
        )

        return (x * self.mask)

    def backward(self,dout):
        if self.mask is None:
            return dout

        return (dout * self.mask)

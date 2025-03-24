import jax
from flax import nnx
from typing import Optional
import optax

LR = 3e-4
MOMENTUM = 0.9

class MLP(nnx.Module):
    def __init__(self, dim: int, lora_rank: Optional[int] = None):
        self.dim = dim
        self.lora_rank = lora_rank

    def __call__(self, x):
        x = nnx.LoRALinear(in_features=self.dim, out_features=self.dim, lora_rank=self.lora_rank)(x)
        x = nnx.relu(x)
        return x

def main():
    devices = jax.devices()
    model = MLP(dim=10, lora_rank=4)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=LR))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), 
        loss=nnx.metrics.Average('loss'),
    )
    print(devices)
    print("Hello from jtx!")


if __name__ == "__main__":
    main()

import jax
from flax import nnx
from typing import Optional
class MLP(nnx.Module):
    def __init__(self, dim: int, lora_rank: Optional[int] = None):
        self.dim = dim
        self.lora_rank = lora_rank

    def __call__(self, x):
        x = nnx.LoRALinear()
        x = nnx.nn.relu(x)
        x = nnx.nn.Dense(features=10)(x)
        return x

def main():
    devices = jax.devices()
    print(devices)
    print("Hello from jtx!")


if __name__ == "__main__":
    main()

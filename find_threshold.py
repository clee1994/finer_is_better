import jax.numpy as jnp

def find_threshold():
    x = jnp.linspace(5.0, 5.1, 10000)
    q_x = jnp.float4_e2m1fn(x)
    for v, q in zip(x, q_x):
        if q == 6.0:
            print(f"Switches to 6.0 at: {v}")
            return
            
if __name__ == "__main__":
    find_threshold()

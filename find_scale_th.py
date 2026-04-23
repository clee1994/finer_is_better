import jax.numpy as jnp

def find_scale_threshold():
    x = jnp.linspace(8.0, 9.0, 10000)
    q_x = jnp.float8_e4m3fn(x)
    for v, q in zip(x, q_x):
        if q == 9.0:
            print(f"Switches to 9.0 at: {v}")
            return

if __name__ == "__main__":
    find_scale_threshold()

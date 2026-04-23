import jax.numpy as jnp

def find_threshold(target_val, search_range):
    x = jnp.linspace(search_range[0], search_range[1], 10000)
    q_x = jnp.float4_e2m1fn(x)
    for v, q in zip(x, q_x):
        if q == target_val:
            return float(v)
    return None

if __name__ == "__main__":
    ranges = [
        (0.5, (0.2, 0.3)),
        (1.0, (0.7, 0.8)),
        (1.5, (1.2, 1.3)),
        (2.0, (1.7, 1.8)),
        (3.0, (2.4, 2.6)),
        (4.0, (3.4, 3.6)),
        (6.0, (4.9, 5.1)),
    ]
    
    for val, r in ranges:
        th = find_threshold(val, r)
        print(f"Threshold for {val}: {th}")

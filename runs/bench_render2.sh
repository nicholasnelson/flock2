cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
timeout 900 python - <<'P' 2>&1 | grep -v "^W\|^I0\|^E1\|cuda_"
import time, jax, jax.numpy as jnp
from flock.meadow2 import render as R
from flock.meadow2.world import random_start
st = random_start(jax.random.PRNGKey(0))
for n in (1, 12):
    f = jax.jit(lambda s: jax.vmap(lambda i: R.render(s, i))(jnp.arange(n)))
    t0 = time.time(); c = f.lower(st).compile(); tc = time.time() - t0
    y = c(st); y.block_until_ready()
    t0 = time.time()
    for _ in range(5): y = c(st)
    y.block_until_ready()
    print(f"{n} views: compile {tc:.1f} s, run {(time.time()-t0)/5*1000:.1f} ms", flush=True)
P

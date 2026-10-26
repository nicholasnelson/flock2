cd /mnt/c/Users/nnels/Documents/projects/flock2 && . ~/.venvs/flock2/bin/activate && export XLA_PYTHON_CLIENT_PREALLOCATE=false
timeout 900 python - <<'P' 2>&1 | grep -v "^W\|^I0\|^E1\|cuda_"
import time, jax, jax.numpy as jnp
from flock.meadow2 import render as R
from flock.meadow2.world import random_start
print("jax", jax.__version__)
def bench(name, fn, arg):
    t0 = time.time(); c = jax.jit(fn).lower(arg).compile(); tc = time.time() - t0
    y = c(arg); y.block_until_ready(); t0 = time.time()
    for _ in range(5): y = c(arg)
    y.block_until_ready(); print(f"{name}: compile {tc:.1f} s, run {(time.time()-t0)/5*1000:.1f} ms", flush=True)
st = random_start(jax.random.PRNGKey(0))
bench("12 views, lax.map", lambda s: jax.lax.map(lambda i: R.render(s, i), jnp.arange(12)), st)
for bs in (4, 12):
    bench(f"12 views, lax.map batch {bs}", lambda s: jax.lax.map(lambda i: R.render(s, i), jnp.arange(12), batch_size=bs), st)
sts = jax.vmap(random_start)(jax.random.split(jax.random.PRNGKey(1), 8))
pairs = (jnp.repeat(jnp.arange(8), 12), jnp.tile(jnp.arange(12), 8))
def many(ss):
    return jax.lax.map(lambda mi: R.render(jax.tree.map(lambda a: a[mi[0]], ss), mi[1]), pairs, batch_size=4)
bench("96 views (8 meadows), lax.map batch 4", many, sts)
P

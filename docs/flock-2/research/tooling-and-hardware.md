# Tooling and hardware feasibility for FLOCK 2.0: vectorised MARL simulation/training (JAX, PyTorch) vs Rust ML crates for in-installation inference and online learning

Research date: 2026-10-22. All prices and version numbers are as observed on that date unless stated otherwise. Items marked **ESTIMATE** are the researcher's own arithmetic, with assumptions stated; everything else is cited.

## Key Question 1: JAX vectorised sim/MARL (JaxMARL, PureJaxRL, Craftax, Jumanji, Brax, gymnax, Kinetix, XLand-MiniGrid, Navix): published steps/sec, wall-clock, and CPU-only behaviour

### Takeaway
JAX grid-world-style environments run at 10^5-10^7 env-steps/s on a single consumer or datacenter GPU when thousands of envs are vmapped, and end-to-end PPO for ~1B steps takes under an hour on an RTX 4090 (Craftax) or ~17 min on an A100 (XLand-MiniGrid, derived). Published CPU-only throughput for these JAX envs is essentially absent; the one single-env data point (JaxMARL, 1 env) shows JAX is *slower* than the original Python env at batch size 1, which is the regime that matters on CPU.

### Cited Findings
- JaxMARL Table 3 (random actions, "single NVIDIA A100 GPU and AMD EPYC 7763 64-core processor"): Overcooked original 1.9×10^3 steps/s (1 env) vs JAX 3.6×10^3 (1 env), 3.0×10^5 (100 envs), 1.7×10^7 (10k envs); MPE Simple Spread original 8.3×10^4 vs JAX 5.5×10^3 (1 env), 5.2×10^5 (100 envs), 4.0×10^7 (10k envs); Hanabi 2.1×10^3 vs 1.4×10^3 / 1.1×10^5 / 5.0×10^6; Coin Game 2.0×10^4 vs 4.7×10^3 / 4.1×10^5 / 4.0×10^7 — [JaxMARL arXiv v5](https://arxiv.org/html/2311.10090v5)
- JaxMARL: "our JAX-based training pipeline is around 14 times faster than existing approaches, and up to 12500x when multiple training runs are vectorized"; QMIX on MPE "complete 1024 individual training runs in 198.4 seconds, compared to 1 hour and 10 minutes for a single training run with PyMARL"; a single Q-learning run on MPE Simple Spread takes 130 s — [JaxMARL arXiv v5](https://arxiv.org/html/2311.10090v5)
- Craftax Table 1 (average steps/s over full PPO training, best config; hardware RTX 4090 24GB + i9-13900K + 32GB RAM): Craftax-Classic 405,618 steps/s at 4096 workers; Craftax 266,961 at 4096; Procgen 7,638 at 1024; NetHack 5,628 at 64; Crafter (Python) 1,580 at 1024; MineRL 133 at 8. Speedups over Crafter: 257x (Classic) and 169x (Craftax). "a run on a single RTX 4090 can finish in an hour" for 1B steps; Craftax-Classic solved to 90% by PPO in 51 min. Paper reports no separate CPU-only numbers — [Craftax arXiv v2](https://arxiv.org/html/2402.16801v2); abstract: "up to 250x faster than the Python-native original" — [Craftax arXiv abs](https://arxiv.org/abs/2402.16801)
- Craftax README: default pip install puts JAX on CPU; first render ~30 s and first action ~20 s of compile time, "then it should be very quick" (no CPU throughput figure) — [Craftax GitHub](https://github.com/MichaelTMatthews/Craftax)
- PureJaxRL blog: "over 4000x training speedups in RL by vectorizing agent training on GPUs"; CartPole-v1 NumPy, 10 parallel envs: ~46 s per 1M frames; gymnax on A100 with 2k envs: ~0.05 s (~1000x); MinAtar-Breakout CPU ~50 s/1M frames vs gymnax ~0.2 s; PPO in JAX "over 10x faster" wall clock than CleanRL PyTorch at the same env count; 2048 CartPole agents train in about half the time of one CleanRL agent; meta-evolution of 512 agents x 1024 generations on a single A40 in ~9 h (>100B frames). The post makes no claims about JAX on CPU — [PureJaxRL blog](https://chrislu.page/blog/meta-disco/)
- gymnax README: speeds reported only as "Secs/1M steps on A100 with 2k envs": CartPole-v1 0.05 s, Breakout-MinAtar 0.19 s, Asterix-MinAtar 0.92 s, FourRooms-misc 0.07 s, MetaMaze-misc 0.09 s; detailed comparisons "on different accelerators (CPU, RTX 2080Ti)" are deferred to gymnax-blines docs (not fetched) — [gymnax GitHub](https://github.com/RobertTLange/gymnax)
- XLand-MiniGrid (A100, random policy, auto-reset): throughput scales "almost log-log linear" with parallel envs and saturates around 2^13 (8,192) envs on one device; multi-device "tens of millions of steps per second"; PPO training ~1.0M steps/s (A100), ~1.2M (H100), ~160k (2x T4), ~400k (RTX 3090), ~800k (RTX 4090); bf16: 1.5M (H100), 2.1M (4090); image observations ~20k steps/s on one GPU at 1,024 envs. CPU-only throughput: not reported — [XLand-MiniGrid arXiv v4](https://arxiv.org/html/2312.12044v4); README: "scales to 2^16 parallel environments and millions of steps per second on a single GPU", "1 trillion environment steps under two days" multi-GPU — [xland-minigrid GitHub](https://github.com/dunnolab/xland-minigrid)
- NAVIX (JAX MiniGrid): ~670M steps/s in batch mode vs ~3,145 steps/s for original MiniGrid; NeurIPS 2025 version states >160,000x speedup in batch mode with up to 2048 agents on a single A100 80GB (earlier arXiv version said >200,000x) — [NAVIX arXiv](https://arxiv.org/html/2407.19396v1); [NAVIX NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/c108bda95007522456be2e4b0794b836-Abstract-Datasets_and_Benchmarks_Track.html)
- Kinetix / Jax2D: "easily scale to thousands of parallel environments on a single GPU"; ~4x faster than Box2D for physics alone and ~30x faster when training RL because the whole pipeline is on GPU; each method allotted 5B env interactions; exact steps/s and GPU in Appendix B (not retrieved) — [Kinetix arXiv v2](https://arxiv.org/html/2410.23208v2)
- Gigastep (NeurIPS 2023): "up to one billion environment steps per second on consumer-grade hardware" (MARL) — [Gigastep NeurIPS poster](https://neurips.cc/virtual/2023/poster/73577)
- Jumanji: 22-24 JAX environments (Maze, Snake, RobotWarehouse, LevelBasedForaging, SearchAndRescue swarms, etc.); only qualitative speed claims, no steps/s — [Jumanji GitHub](https://github.com/instadeepai/jumanji)
- Brax: "millions of physics steps per second on TPU", "train agents in seconds to minutes"; CPU not mentioned — [Brax GitHub](https://github.com/google/brax)
- Madrona: "thousands of virtual environment instances" and "millions of aggregate simulation steps per second, on a single GPU"; GPU backend needs Volta+ NVIDIA, CUDA 12.4/12.8+, Linux-only; a CPU backend exists "for debugging and visualization"; envs are C++ (ECS); state exported as PyTorch tensors — [Madrona GitHub](https://github.com/shacklettbp/madrona)
- DreamerV3 (Nature 2025 / arXiv v2): six model sizes "from 12M to 400M parameters"; "one Nvidia A100 GPU per Dreamer agent"; Minecraft ~9 days on 1 GPU — [DreamerV3 arXiv v2](https://arxiv.org/html/2301.04104v2)

### Inferences
- The JaxMARL "1 env" column is the best public proxy for JAX-on-CPU-at-small-batch behaviour: at batch 1, JAX is 0.7x-15x *slower* than hand-written Python/NumPy envs (MPE: 5.5k vs 83k steps/s; Overcooked: 3.6k vs 1.9k). The big wins only appear at ≥100-10k vmapped envs, i.e. on an accelerator. For FLOCK's 20-60 sheep on one meadow at a few Hz, JAX-on-CPU buys nothing over plain Rust.
- A grid world the size of FLOCK's (128x128 meadow, 16x16 egocentric view, ~50 agents) is structurally between XLand-MiniGrid (small grids, ~1M PPO steps/s on A100) and Craftax (~270k steps/s on 4090 with 4096 envs, symbolic obs). ESTIMATE: a JAX port of flock-sim could plausibly reach 10^5-10^6 agent-steps/s on a rented 4090 class GPU with 1k-4k meadows, so 1B agent-steps is on the order of an hour of GPU time (assumption: obs are symbolic 16x16x16 tensors, not rendered pixels; XLand's image-obs path drops to 20k steps/s).
- All the JAX wins presuppose a pure-functional, vmappable env; porting flock-sim (Rust, imperative) to JAX is a full rewrite, not a wrap.

### Gaps
- No source publishes JAX env-stepping or PPO throughput on CPU-only for any of these libraries; gymnax-blines allegedly has CPU/RTX2080Ti comparisons but was not fetched. Treat "JAX on CPU" as unknown beyond the batch-1 JaxMARL data points.
- Kinetix Appendix B steps/s numbers and GPU were not retrieved.
- DreamerV3 per-size table (GRU units, CNN depth per variant) was in a section not returned by the fetch.

## Key Question 2: PyTorch-based fast RL (PufferLib, Sample Factory, EnvPool, Madrona, CleanRL): numbers

### Takeaway
PufferLib (C envs + CUDA C trainer) advertises 1M steps/s (v2.0), 4M steps/s (v3.0) and "up to 60M step/s" training (v5.0 docs, hardware unstated), while EnvPool's own benchmarks show 37k-49k Atari FPS on a 12-core laptop and ~1.07M FPS on a 256-core DGX-A100; custom envs for PufferLib are written in C against a minimal template, which is the closest analogue to driving a Rust sim from Python.

### Cited Findings
- PufferLib docs (5.0): "Up to 60,000,000 step/second training in only ~10k lines of CUDA C" (hardware not stated); "Train a policy in under a second (RTX 5090)"; custom env = copy the `ocean/minimal` template, header at `ocean/your_env/your_env.h`, declare obs/action metadata in `binding.c`, build with `./build.sh ENV --cpu`; only C and CUDA C are mentioned for envs; observation buffers are not zeroed automatically — [PufferLib docs](https://puffer.ai/docs.html)
- PufferLib site: "PufferLib 2.0: Reinforcement Learning at 1M sps" (RLJ 2025 paper; arXiv 2406.12905) and "PufferLib 3.0: Better Reinforcement Learning at 4M sps"; 4.0 adds "Environment vectorization" and a "high-throughput recurrent network" post — [puffer.ai](https://puffer.ai/)
- PufferLib GitHub: 6.5k stars, `5.0` branch, 4,297 commits, MIT; "can train tiny, super-human models in seconds"; support via Discord — [PufferLib GitHub](https://github.com/PufferAI/PufferLib)
- Third-party docs mirrors claim Ocean envs are C/CUDA and hit "1M-10M+ steps per second on a single CPU core" and PuffeRL (PPO+LSTM) trains at "1M-4M steps/second" — secondhand, unverified hardware — [Mintlify PufferLib mirror](https://www.mintlify.com/PufferAI/PufferLib/ocean/overview)
- EnvPool (random actions, raw sim throughput): Atari on a 12-core laptop: Sample-Factory ~28,216 FPS vs EnvPool sync 37,396 / async 49,439; Atari on 256-core DGX-A100: Sample-Factory ~707,494 vs EnvPool async 891,286 / NUMA+async 1,069,922; MuJoCo laptop: SF 62,510 vs EnvPool async 105,126; DGX-A100: SF 1,573,262 vs EnvPool NUMA+async 3,134,287 — [EnvPool PyPI 1.2.4](https://pypi.org/project/envpool/1.2.4/); [EnvPool wiki](https://github.com/sail-sg/envpool/wiki); [EnvPool arXiv](https://arxiv.org/pdf/2206.10558)
- Madrona: millions of aggregate steps/s on one GPU; C++ ECS envs; PyTorch tensor export; GPU backend Linux-only, Volta+ — [Madrona GitHub](https://github.com/shacklettbp/madrona)
- PureJaxRL's own comparison: JAX PPO "over 10x faster" than CleanRL PyTorch at equal env counts on CartPole/MinAtar — [PureJaxRL blog](https://chrislu.page/blog/meta-disco/)

### Inferences
- PufferLib's architecture (compiled env + minimal `binding.c` + Python trainer) is a direct template for FLOCK: flock-sim could be exposed with a C ABI (`#[no_mangle] extern "C"`) or PyO3 and stepped in batch from Python. PufferLib does not officially support non-C envs, so this would mean either (a) writing the thin C shim that calls into a Rust `cdylib`, or (b) using PyO3 + numpy zero-copy with a different trainer (CleanRL/own PPO). Both are unproven for PufferLib specifically.
- EnvPool's laptop numbers (~40k Atari FPS on 12 cores) are a realistic ceiling for a CPU-side batched Rust sim stepping from Python; FLOCK's sim is cheaper than Atari emulation per step, so 10^5 agent-steps/s on a 16-core desktop is plausible (ESTIMATE, assumes sub-10 µs per agent-step in Rust, which is unmeasured for flock-sim).

### Gaps
- No PufferLib per-environment steps/s table with named hardware was retrievable from first-party pages; the 1M/4M/60M figures are headline claims.
- No precedent found of PufferLib binding a Rust env.

## Key Question 3: Rust-side ML crates (candle, burn, tract, ort, dfdx): maturity, training support, CPU performance, WASM, weight loading, online-learning realism

### Takeaway
As of October 2026: burn (0.22) and candle both have autodiff and can load PyTorch/safetensors weights; tract (0.23.9) and ort (2.0.0-rc.14/15, ONNX Runtime 1.31) are the mature CPU inference paths with WASM support, and ort additionally exposes ONNX Runtime on-device training; dfdx is pre-alpha. No public benchmark of small CNN+GRU nets on x86 CPU exists for any of them, so online-learning feasibility must be estimated from FLOPs (see Q5), where it looks comfortably feasible.

### Cited Findings
**burn (tracel-ai/burn)**
- Current version 0.22; "LibTorch was deprecated in 0.22.0 and has been removed from main"; backends: CUDA, ROCm, Metal, Vulkan, WebGPU (CubeCL), "Cpu (CubeCL)" which "JIT-compiles the same kernels as the GPU backends through LLVM and supports fusion", and "Flex", "a pure-Rust eager backend with no native dependencies"; autodiff is a decorator ("autodiff cannot execute by itself") with gradient checkpointing; terminal training dashboard; ONNX import converts models to Rust code and "currently supports a limited set of ONNX operators"; "load weights from PyTorch or Safetensors formats directly into your Burn-defined models" (burn-store); WASM via Flex (CPU) and WGPU; no_std via Flex only; "recompiles in under 5 seconds"; benchmarks via burn-bench, no numbers in README — [burn GitHub](https://github.com/tracel-ai/burn); docs.rs shows 0.21.0 built, 0.22.0 docs build failed — [docs.rs burn](https://docs.rs/crate/burn/latest)
- burn-import 0.8.0 (old) checklist marked GRU and LSTM as unchecked/unimplemented for ONNX import — [burn-import 0.8.0 mirror](https://crates-io.ganymede-static.progval.net/crates/burn-import/0.8.0). The current SUPPORTED-ONNX-OPS.md could not be fetched (404 at the old path), so current GRU/LSTM ONNX-import status is unverified.
- burn's Candle backend README: "alpha, not all operations are supported. It is usable for some use cases, like for inference" — [search result summary](https://cdn05042023.gitlink.org.cn/dnrops/burn/src/branch/main/crates/burn-candle)

**candle (huggingface/candle)**
- Features: "Simple syntax (looks and feels like PyTorch)", "Model training", "Optimized CPU backend with optional MKL support for x86 and Accelerate for macs", CUDA (multi-GPU via NCCL), "WASM support, run your models in a browser"; loads safetensors, npz, ggml, PyTorch files; FAQ: "Candle's core goal is to make serverless inference possible"; 2,726 commits — [candle GitHub](https://github.com/huggingface/candle)
- candle-core exposes a `backprop` module — [docs.rs candle-core](https://docs.rs/candle-core)
- candle-nn provides `GRU`, `LSTM` structs (with `GRUConfig`, `LSTMConfig`, state types) and an `RNN` trait — [docs.rs candle-nn rnn](https://docs.rs/candle-nn/latest/candle_nn/rnn/index.html)

**tract (sonos/tract)**
- "Sonos' neural-network inference engine"; loads ONNX and NNEF (and TFLite); default CPU runtime for x86, ARMv6/7/8, ARM SVE with "hand-rolled SIMD micro-kernels"; Metal and CUDA runtimes; WebAssembly for browser/WASI; "pulsified" streaming inference; nightly benchmark trend examples (e.g. Parakeet preprocessor on i9-11900KB 14.7x faster since Jun 2026; MobileNet v2 multi-thread on Cortex-A53 3.2x since Aug 2026); min rustc 1.91; no training API — [tract GitHub](https://github.com/sonos/tract); [tract README raw](https://raw.githubusercontent.com/sonos/tract/main/README.md)
- tract-onnx latest 0.23.9 dated 2026-10-15 — [docs.rs tract-onnx](https://docs.rs/crate/tract-onnx/latest)
- Historical (Oct 2020 fork README): GRU and LSTM listed among implemented and tested ONNX ops; ~85% ONNX backend tests passing at that time — [tract fork README (2020)](https://github.com/apachecn/tract); early GRU loading issue #141 with optional-input quirks — [tract issue #141](https://github.com/sonos/tract/issues/141)
- FOSDEM 2026 talk: tract started 2018, used across Sonos devices, reads ONNX/NNEF/TFLite — [FOSDEM 2026 tract](https://archive.fosdem.org/2026/schedule/event/YJJQTD-tract-and-torch-to-nnef/)

**ort (pykeio/ort)**
- "Fast ML inference & training for ONNX models in Rust"; badge shows ONNX Runtime v1.31.0; docs link to 2.0.0-rc.15; 2.6k stars, 1,097 commits, Apache-2.0/MIT — [ort GitHub](https://github.com/pykeio/ort)
- Docs: "These docs are for the latest version of ort, 2.0.0-rc.14", called "production-ready"; execution providers include CUDA, TensorRT, OpenVINO, QNN, CANN "and much more"; "Deploy your application to the Web with the ort-web backend" — [ort.pyke.io](https://ort.pyke.io/)
- Training: README describes a `Trainer` API for on-device training (manual loop and a HF-style loop; `train-clm` / `train-clm-simple` examples); rc.14 released Oct 12 2026 with ONNX Runtime 1.31.0 and "training memory fixes"; rc.13 Jul 28 2026 — [ort README on docs.rs](https://docs.rs/crate/ort/2.0.0-rc.13/source/README.md); [pyke listing](https://www.everydev.ai/developers/pyke)

**dfdx (coreylowman/dfdx)**
- Reverse-mode autodiff via gradient tape checked at compile time; CPU and CUDA backends; Linear, Conv2D, Transformer modules; Sgd/Adam/AdamW/RMSprop; README: "still in pre-alpha state", breaking releases ahead; crates.io 0.13.0; saves `.npz` (no safetensors mentioned) — [dfdx GitHub](https://github.com/coreylowman/dfdx)

**Other**
- No head-to-head burn vs candle CPU benchmark exists; burn ships `burnbench` for `candle-cpu`, `ndarray`, `tch-cpu` comparisons — [burn backend-comparison README](https://cdn05042023.gitlink.org.cn/dnrops/burn/raw/branch/main/backend-comparison/README.md); a 2026 blog characterises burn as backend-agnostic (CPU/GPU/WASM via CubeCL) and candle as inference/quantisation-focused — [dasroot.net comparison (Apr 2026)](https://dasroot.net/posts/2026/04/rust-machine-learning-burn-vs-candle-framework-comparison/)
- `candle-lstm` third-party crate exists — [candle-lstm](https://rust-digger.code-maven.com/crates/candle-lstm)

### Inferences
- For *training in Rust* (online learning in the installation): burn is the only crate positioned as a full training framework (autodiff + optimisers + dashboard + CPU JIT backend); candle can also train (backprop + GRU/LSTM in candle-nn) but its stated goal is inference; ort can do on-device training only if the training graph is produced by ONNX Runtime Training's Python artifact tooling (pattern: build artifacts in Python, step optimizer in Rust); dfdx should be excluded (pre-alpha).
- For *inference only* with weights trained in PyTorch: ort (ONNX) is the most op-complete and best-benchmarked path (ONNX Runtime itself); tract is pure-Rust, lighter, WASM-capable and has long-standing GRU/LSTM ONNX ops; burn ONNX import is codegen-based and historically lacked GRU/LSTM, so for a GRU-based RSSM the safer burn route is to define the model in burn and load safetensors, not import ONNX.
- Online learning realism: see Q5 ESTIMATE; a few SGD steps/s on a ~1-2M-param CNN+GRU on 8-16 x86 cores is well within budget in any autodiff-capable crate, but *no public benchmark confirms burn/candle CPU training throughput for such nets* — this needs a 1-day spike.

### Gaps
- Current burn SUPPORTED-ONNX-OPS.md (GRU/LSTM status) not retrievable; current tract ONNX opset/pass-rate statement not found on README.
- No CPU benchmarks of small CNN/GRU nets in any Rust crate; no published burn-bench results for CPU.
- candle-nn docs do not state whether GRU/LSTM support backprop (they are built from differentiable tensor ops, so likely yes, but unverified).

## Key Question 4: Interop pattern (train in Python, export safetensors/ONNX, infer in Rust): practice and pitfalls, especially GRU/RNN export

### Takeaway
Train-in-Python/infer-in-Rust via safetensors (burn, candle) or ONNX (ort, tract) is standard; the recurring pitfall is RNN export: PyTorch GRU ONNX export historically bakes in batch size/sequence length, cannot export packed sequences or `GRUCell`, and the new onnxscript exporter only gained `aten_gru`/`aten_lstm` in Nov 2025. The robust pattern is to export a single-step cell (h_t = f(x_t, h_{t-1})) with h as an explicit input/output.

### Cited Findings
- PyTorch GRU export warning: "Exporting a model to ONNX with a batch_size other than 1, with a variable length with GRU can cause an error when running the ONNX model with a different batch size. Make sure to save the model with a batch size of 1, or define the initial states (h0/c0) as inputs of the model"; dynamic axes on length had no effect for that user — [PyTorch forum](https://discuss.pytorch.org/t/export-rnn-gru-as-onnx-how-to-set-the-batch-size/111661)
- Exported GRU only ran at the dummy-input sequence length in ONNX Runtime — [PyTorch forum](https://discuss.pytorch.org/t/unable-to-export-gru-model-in-onnx-accepting-variable-length-sequence/64223)
- "Cannot export individual pack_padded_sequence or pad_packed_sequence; these operations must occur in pairs" — [PyTorch forum](https://discuss.pytorch.org/t/pad-packed-sequence-export-to-onnx/56089)
- `GRUCell` export fails with missing `_thnn_fused_gru_cell` symbolic — [PyTorch forum](https://discuss.pytorch.org/t/onnx-export-failed-on-aten-operator-thnn-fused-gru-cell/110616)
- microsoft/onnxscript PR (Nov 2025) adds `aten_lstm` and `aten_gru` to enable export of PyTorch LSTM/GRU layers (multi-layer, bidirectional, batch_first) and maps PyTorch gate order to ONNX's — [onnxscript PR #2674](https://github.com/microsoft/onnxscript/pull/2674)
- burn: "load weights from PyTorch or Safetensors formats directly into your Burn-defined models"; ONNX import is codegen with limited op set — [burn GitHub](https://github.com/tracel-ai/burn)
- candle: loads safetensors/npz/ggml/PyTorch files; `candle::safetensors::load` — [candle GitHub](https://github.com/huggingface/candle)
- tract: loads ONNX, NNEF, TFLite; FOSDEM 2026 talk is literally titled "tract and torch-to-nnef" — [FOSDEM 2026](https://archive.fosdem.org/2026/schedule/event/YJJQTD-tract-and-torch-to-nnef/)
- Precedents for Rust envs driven from Python RL: entity-gym-rs ("lets Rust programs be used as EntityGym training environments", on PyPI) — [entity-gym-rs](https://github.com/entity-neural-network/entity-gym-rs); rlgym-sim-rs (Rust port of RLGym "adapted ... for the ease of use of bindings via PyO3") — [rlgym-sim-rs](https://github.com/jeffa233/rlgym-sim-rs); rlox (Rust data plane for env stepping/buffers/advantages, Python control plane via PyO3, maturin, "zero-copy where possible") — [rlox PyPI](https://pypi.org/project/rlox/); twisterl-rs (generic Env trait, parallel collectors, optional PyO3 bindings; "proof of concept") — [twisterl-rs](https://rust-digger.code-maven.com/crates/twisterl-rs); RelayRL (Rust core, PyO3 bindings; "prototype") — [relayrl_framework](https://docs.rs/relayrl_framework)
- PufferLib's own compiled-env pattern: C header + `binding.c` metadata + `build.sh` — [PufferLib docs](https://puffer.ai/docs.html)

### Inferences
- Safest FLOCK pattern: implement the world model as an explicit single-step cell in PyTorch (`h' = GRUCell(x, h)` written out with Linear ops rather than `nn.GRUCell`, or `nn.GRU` with seq_len=1 and h0 as input), export ONNX with batch=N sheep and h as input/output, and run it per tick in ort/tract. Rollout over time is then a Rust loop, which also matches the installation's "one tick at a time" execution. The same cell can be written in burn and loaded from safetensors if online learning in Rust is wanted.
- No mature "Rust env for PufferLib" exists; the realistic path is PyO3 + maturin with batched numpy I/O (as rlox/rlgym-sim-rs do), driving a PyTorch PPO/Dreamer loop.

### Gaps
- Whether the Nov 2025 onnxscript GRU exporter is merged/released in a stable PyTorch version was not confirmed.
- No documented case of loading the *same* safetensors into both a PyTorch and a burn model with a GRU (name-mapping details unverified).

## Key Question 5: CPU feasibility numbers (i7-7820X, i7-13700KF, Pi 5) for ~1M-param CNN+GRU inference at 50 agents x 5 Hz plus modest training

### Takeaway
No published benchmark matches FLOCK's net, so this is an ESTIMATE from FLOPs: a ~1.5-2M-param CNN+GRU on 16x16x16 input costs ~15-25 MFLOP per agent forward; 60 agents x 5 Hz = ~6-8 GFLOP/s, i.e. well under 1% of an i7-7820X's ~1.8 TFLOPS FP32 peak and under ~10% of a Pi 5's realistic ~30-50 GFLOPS effective, so inference is feasible everywhere; a few SGD steps/s on batches of a few hundred samples (~30 GFLOP/step) is feasible on the x86 boxes but marginal on the Pi.

### Cited Findings
- i7-7820X: 8 cores/16 threads, base 3.6 GHz, turbo 4.3 GHz (4.5 Turbo Max), 2 AVX-512 FMA units — [WikiChip i7-7820X](https://en.wikichip.org/wiki/intel/core_i7/i7-7820x); [TechPowerUp](https://techpowerup.com/cpu-specs/core-i7-7820x.c1933); Puget Linpack AVX-512 vs AVX2 comparison on Skylake-X exists — [Puget Systems](https://www.pugetsystems.com/labs/hpc/Skylake-X-7800X-vs-Coffee-Lake-8700K-for-compute-AVX512-vs-AVX2-Linpack-benchmark-1068/)
- i7-13700KF: 16 cores (8 P + 8 E), 24 threads; listed instruction sets are SSE4.2/AVX/AVX2/FMA/VAES etc. with no AVX-512 — [AVADirect listing](https://www.avadirect.com/Core-i7-13700KF-16-8P-8E-Cores-2-5-5-4GHz-Turbo-LGA-1700-253W-MTP-Processor/Product/15147608); [TechPowerUp 13700K review](https://techpowerup.com/review/intel-core-i7-13700k/)
- Raspberry Pi 5: Broadcom BCM2712, quad-core Cortex-A76 at 2.4 GHz, 512 KB L2/core, 2 MB L3, LPDDR4X-4267, 1-16 GB; "2-3x the speed of the previous generation" — [raspberrypi.com Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)
- Pi 5 inference benchmark (Ultralytics 8.4.108, YOLO26n-class model, FP32, 640 px, pre/post-processing excluded): PyTorch 299.09 ms, TorchScript 353.20, ONNX (Runtime) 125.99, OpenVINO 104.55, LiteRT 123.30, MNN 91.87, NCNN 67.03, ExecuTorch 144.83 ms/image — [Ultralytics Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/)
- Pi 5, ncnn Vulkan (GPU) backend, Phoronix: MobileNet v1 ~41-44 ms, v2 ~11-12.4 ms, v3 ~8.1-8.8 ms (not CPU numbers) — [OpenBenchmarking Pi 5 ncnn](https://mail.openbenchmarking.org/result/2501280-NE-DJD60178083&grs)
- Pi 5 TFLite MobileNetV1 224x224 ~25 ms/inference vs ~55 ms on Pi 4 (vendor blog, thread count unstated) — [Zbotic](https://zbotic.in/raspberry-pi-ai-ml-projects-tensorflow-lite-on-pi-5/); independent edge benchmark: Pi 5 TensorFlow MobileNet v2 148.9 ms and MobileNet v1 SSD 66.2 ms at 300x300 (detection) — [aallan benchmarking-ml-on-the-edge](https://github.com/aallan/benchmarking-ml-on-the-edge)
- ncnn's benchmark table has no Pi 5 rows; closest Cortex-A76-class data point is Snapdragon 855+ (Kryo 485, 4 threads): mobilenet 5.38 ms, mobilenet_v2 4.68 ms, squeezenet 3.64 ms, resnet18 10.35 ms — [ncnn benchmark README](https://raw.githubusercontent.com/Tencent/ncnn/master/benchmark/README.md)
- ONNX Runtime provides `onnxruntime_perf_test` for latency measurement — [ONNX Runtime benchmarking docs](https://www.mintlify.com/microsoft/onnxruntime/performance/benchmarking); official Pi tutorial runs MobileNet via ONNX Runtime without timings — [ONNX Runtime Pi tutorial](https://onnxruntime.ai/docs/tutorials/iot-edge/rasp-pi-cv.html)

### Inferences (all ESTIMATE, assumptions stated)
- **Peak FP32 compute.** i7-7820X: 2 AVX-512 FMA units x 16 lanes x 2 FLOP = 64 FLOP/cycle/core; 8 cores x 3.6 GHz = ~1.84 TFLOPS (4.3 GHz: ~2.2 TFLOPS); sustained AVX-512 clocks are lower, so assume ~1.5 TFLOPS peak and 10-30% achieved on small-batch nets = ~150-450 GFLOPS effective. i7-13700KF: P-cores 2 x AVX2 FMA x 8 lanes x 2 = 32 FLOP/cycle x 8 cores x ~5 GHz = ~1.3 TFLOPS plus E-cores (~16 FLOP/cycle x 8 x ~4 GHz = ~0.5 TFLOPS); effective ~150-400 GFLOPS. Pi 5: assume Cortex-A76 has 2 x 128-bit NEON FMA pipes = 16 FLOP/cycle x 4 cores x 2.4 GHz = ~154 GFLOPS peak; the Ultralytics ONNX figure (126 ms for a ~6-9 GFLOP YOLO-n class model; FLOPs figure from memory, unverified) implies ~50-70 GFLOPS effective in ONNX Runtime and ~100 GFLOPS in NCNN.
- **Per-agent forward cost.** Assumed net: input 16x16x16 one-hot -> conv3x3 16->32 (2.4 MFLOP) -> conv3x3 s2 32->64 (2.4 MFLOP) -> conv3x3 s2 64->64 (1.2 MFLOP) -> dense 1024->256 (0.5 MFLOP) -> GRU(512) with 3 x 512 x (512+256+~16) weights = ~1.2M params (2.4 MFLOP/step) -> policy/value MLPs (~0.3 MFLOP) -> drawing decoder (dense 512->1024 + 3 transposed convs, ~6 MFLOP) -> bleat head (negligible). Total ~1.6-2M params, ~15-20 MFLOP per agent forward (counting multiply-add as 2 FLOP). The world-model prior/posterior MLPs of an RSSM add perhaps 1-2 MFLOP.
- **Online inference load.** 60 sheep x 5 Hz x 20 MFLOP = 6 GFLOP/s. On the i7-7820X at 150 GFLOPS effective that is ~4% of one socket's budget (or ~0.3 of one core); batching all 60 sheep into one [60, ...] forward per tick amortises overhead. On Pi 5 at ~50 GFLOPS effective in ONNX Runtime: ~12% utilisation, i.e. ~25 ms of CPU per 200 ms tick, leaving headroom but competing with the Chromium/WASM renderer. Verdict: inference is feasible on all three; the Pi should only be a fallback because the renderer already lives there.
- **Online learning load.** One SGD step on a batch of 32 sequences x 16 timesteps = 512 samples; forward+backward ~3x forward = 512 x 20 MFLOP x 3 = ~30 GFLOP; Adam update on 2M params is negligible. At 150-300 GFLOPS effective, that is 0.1-0.2 s per step, so ~5-10 SGD steps/s on the i7-7820X or 13700KF while inference uses a fraction of a core. "A few SGD steps/s" is therefore realistic with a 2-4x safety margin. On Pi 5 (~50 GFLOPS) a step takes ~0.6 s and would starve the renderer: not recommended.
- **Memory.** 2M params fp32 = 8 MB weights (+ 16 MB Adam moments); 60 sheep x 512-float GRU states is trivial; a replay buffer of 1M agent-steps of 16x16x16 uint8 obs = 4 GB, or 256 MB if stored as 16x16 palette indices (1 byte/pixel). Fits a homelab server easily.
- **Offline pretraining scale.** Thousands of sheep across many meadows in Rust on the 13700KF: if flock-sim steps at ~5 µs/agent-step (unmeasured) on 16 threads that is ~10^6 agent-steps/s of *simulation*, but the policy forward at 20 MFLOP each caps the CPU at ~10^4 agent-steps/s with learning. That is 100x slower than the ~10^5-10^6 steps/s JAX-on-4090 regime; 1B steps would take ~1 day on GPU vs ~weeks on CPU.

### Gaps
- No direct benchmark of a ~1M-param CNN+GRU on x86 or Pi 5 in ONNX Runtime/tract/burn; all per-model numbers above are derived.
- YOLO26n FLOP count used to back out Pi 5 effective GFLOPS was not fetched.
- Actual flock-sim v0.4 step cost is unmeasured in this research.

## Key Question 6: JAX vs PyTorch for an individual developer doing MARL with world models (2025-2026), DreamerV3 implementations, and whether flock-sim can be the vectorised env

### Takeaway
JAX gives 10-100x cheaper offline training *if* the env is rewritten purely functionally (Craftax/JaxMARL/XLand all did this), while PyTorch (PufferLib/CleanRL/PyO3) lets the existing Rust sim be reused with a thin binding at EnvPool-class CPU throughput; the reference DreamerV3 is JAX (danijar), the best-known PyTorch port (NM512) was archived in Jul 2026 in favour of r2dreamer, and both are far bigger (12M-400M params) than FLOCK's 0.5-2M target.

### Cited Findings
- danijar/dreamerv3 is JAX, Python 3.11+, GPU by default with `--jax.platform cpu` available; 32 commits; `size50m` config mentioned; no throughput figures — [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- DreamerV3 paper: six model sizes 12M-400M params; one A100 per agent; Minecraft ~9 GPU-days — [DreamerV3 arXiv v2](https://arxiv.org/html/2301.04104v2)
- NM512/dreamerv3-torch: PyTorch DreamerV3 (DMC, Atari100k, Crafter, Minecraft, Memory Maze); 888 stars; archived read-only on Jul 5 2026 with a note that it predates major DreamerV3 updates; author points to r2dreamer, claimed "~5x faster" — [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- JaxMARL: envs "can be very easily parallelised by using JAX's vmap"; RL "slowed down by CPU-GPU data transfers" otherwise — [JaxMARL arXiv v4](https://arxiv.org/html/2311.10090v4); gymnax executes env and policy both on the accelerator (Anakin pattern) — [gymnax PyPI](https://pypi.org/project/gymnax/)
- Craftax hardware for all results: one RTX 4090 + i9-13900K — [Craftax arXiv v2](https://arxiv.org/html/2402.16801v2)
- PureJaxRL meta-evolution of 512 agents x 1024 generations on one A40 in ~9 h — [PureJaxRL blog](https://chrislu.page/blog/meta-disco/)
- PufferLib custom-env workflow is C-only; "hook the C/C++ for Atari and maybe ProcGen into our low-level interface when we have time" is a future plan — [PufferLib docs](https://puffer.ai/docs.html)
- Rust-from-Python RL precedents: entity-gym-rs, rlgym-sim-rs (PyO3), rlox (PyO3 + maturin, zero-copy), twisterl-rs, RelayRL — see Q4 citations
- Madrona offers a C++ ECS + PyTorch tensor route with millions of steps/s on GPU, Linux-only, Volta+ — [Madrona GitHub](https://github.com/shacklettbp/madrona)

### Inferences
- Two coherent stacks exist. **Stack A (JAX):** port flock-sim rules to a pure-functional JAX env (Craftax-style), use PureJaxRL/JaxMARL-style PPO or a trimmed DreamerV3 in JAX, train on a rented 4090/A100 at ~10^5-10^6 agent-steps/s, export safetensors, infer in Rust. Cost: a second implementation of the meadow that must stay in sync with flock-sim; benefit: evolution/meta-learning across thousands of meadows becomes cheap (PureJaxRL's 512-agent x 1024-generation run is exactly this). **Stack B (PyTorch + Rust env):** expose flock-sim via PyO3/maturin with batched stepping (many meadows per call, numpy out), train with CleanRL-style PPO or r2dreamer-style world model in PyTorch; CPU-bound at roughly EnvPool laptop rates (~10^4-10^5 agent-steps/s) unless a GPU runs the nets. Benefit: one simulation implementation; cost: 10-100x less throughput than Stack A for population-scale evolution.
- DreamerV3 reference sizes (12M+) are 6-20x larger than FLOCK's budget; a from-scratch small RSSM (e.g. GRU 512, 32x32 or 16x16 categorical latents) in either framework is more realistic than adapting the reference repos.
- For a solo developer in 2026, the PyTorch ecosystem for world models is in flux (NM512 archived; r2dreamer new), whereas the JAX grid-world ecosystem (Craftax, XLand, NAVIX, JaxMARL) is actively maintained and benchmarked; but every JAX option assumes rewriting the env.

### Gaps
- r2dreamer's repo, size, and benchmarks were not fetched.
- No published example of a Rust env compiled to be stepped in batch by PufferLib or by PureJaxRL (JAX cannot call into Rust inside jit except via experimental FFI; not researched).

## Key Question 7: Cloud GPU pricing (Oct 2026) for RTX 4090 / A100 / H100 on RunPod, Lambda, Vast.ai

### Takeaway
As of Oct 2026 an RTX 4090 rents for ~$0.30-0.90/h (Vast.ai/RunPod Community ~$0.29-0.34, RunPod Secure $0.89), an A100 80GB for ~$1.19-1.79/h (RunPod) or $1.99-2.79 (Lambda, 40/80GB), and an H100 for ~$1.99-4.29/h; a 1B-step Craftax-class run (~1 h on a 4090) therefore costs well under $1-2.

### Cited Findings
- RunPod ("Updated October 8, 2026"), on-demand per hour, Community / Secure: RTX 4090 $0.34 / $0.89; RTX 5090 $0.69 / $1.19; A100 80GB PCIe $1.19 / $1.79; A100 80GB SXM $1.39 / $1.79; H100 PCIe $1.99 / $2.89; H100 SXM $2.69 / $3.99 — [RunPod pricing](https://www.runpod.io/gpu-cloud/pricing)
- Lambda on-demand (page undated, © 2026), per GPU/h for 1x: H100 SXM $4.29 (8x $3.99), H100 PCIe $3.29, A100 SXM 40GB $1.99, A100 PCIe 40GB $1.99, A100 SXM 80GB $2.79 (8x only), B200 $6.99, A6000 $1.09, A10 $1.29 — [Lambda GPU cloud](https://lambda.ai/service/gpu-cloud)
- Vast.ai (marketplace, prices set by supply/demand; live grid did not load) — [Vast.ai pricing](https://vast.ai/pricing); third-party trackers: on-demand RTX 4090 from $0.29/h, interruptible from ~$0.15/h — [Costbench Vast.ai 2026](https://costbench.com/software/ai-gpu-cloud/vast-ai/); $0.309/GPU-h on-demand (updated Aug 5 2026) — [ComputePrices](https://computeprices.com/providers/vast/gpus/rtx4090); individual listings $0.61-0.69/h on-demand, spot ~$0.13-0.16/h — [GetDeploying RTX 4090](https://getdeploying.com/gpus/nvidia-rtx-4090)
- GetDeploying RTX 4090 cross-provider table (updated Oct 22 2026): Salad $0.33 (spot $0.16), TensorDock $0.46, RunPod Secure $0.89 (Community $0.34 out of stock at check time) — [GetDeploying cloud GPU reference](https://getdeploying.com/reference/cloud-gpu/nvidia-rtx-4090)

### Inferences
- ESTIMATE: offline pretraining budget for FLOCK at JAX-on-4090 rates (~3x10^5 steps/s): 10B agent-steps = ~9 h = ~$3-8 on Community/Vast tiers; even 100 such runs for evolutionary sweeps are < $1k. GPU cost is not the constraint; developer time to build the vectorised env is.

### Gaps
- Vast.ai first-party live prices not captured (JS grid); only aggregator snapshots.

## FLOCK 2.0 RECOMMENDATION (researcher's assessment; clearly labelled, separate from cited findings)

### Takeaway
Keep the Rust sim as the single source of truth and the Rust renderer on the Pi; do offline training in Python (PyTorch by default, with an optional JAX port of the meadow only if population-scale evolution across thousands of meadows becomes central); run installation inference and modest online learning on the x86 server in Rust via ort (ONNX) or burn (safetensors), with the Pi doing rendering only.

### Recommendation details
1. **Phase B (installation) on i7-7820X or i7-13700KF: feasible with large margin.** ESTIMATE: 60 sheep x 5 Hz of a ~2M-param CNN+GRU is ~6 GFLOP/s, <5% of either CPU; a few SGD steps/s (~30 GFLOP each) is ~5-10 steps/s on either box. AVX-512 on the 7820X vs AVX2-only on the 13700KF does not change the verdict; the 13700KF's 16 cores give more headroom for sim + render encode + training. Keep the brain on x86, not the Pi.
2. **Rust crate choice.** Inference-only, weights from PyTorch: `ort` (ONNX Runtime 1.31 via ort 2.0.0-rc.14/15, "production-ready" per docs) first; `tract` 0.23.9 (pure Rust, WASM, long-standing GRU/LSTM ONNX ops) as the dependency-light alternative, especially if any brain code must run in the Pi's browser (unlikely). Online learning in Rust: `burn` 0.22 (autodiff + CPU CubeCL/Flex backends + PyTorch/safetensors weight loading); `candle` is a viable second choice (has backprop and GRU/LSTM) but is inference-focused by its own statement. Exclude `dfdx` (pre-alpha). Export the recurrent cell as a single-step function with hidden state as explicit I/O to dodge the PyTorch GRU ONNX pitfalls.
3. **Phase A (offline) default: PyTorch + PyO3-wrapped flock-sim, batched over many meadows.** This reuses v0.4 and follows the PufferLib/rlox/rlgym-sim-rs pattern; expect CPU-bound ~10^4-10^5 agent-steps/s (ESTIMATE), adequate for pretraining a 1-2M-param model but slow for thousands-of-sheep evolution. Rent a 4090 (~$0.3-0.9/h) only for the network side.
4. **Phase A upgrade path: JAX port of the meadow (Craftax/XLand style)** if population-scale evolution is the artistic core. Published numbers (Craftax 2.7-4.1x10^5 steps/s on a 4090; XLand ~0.8M PPO steps/s on a 4090; PureJaxRL 512 agents x 1024 generations on one A40 in 9 h) show 1B-step runs cost ~1 GPU-hour (~$0.5-1). The price is maintaining two meadow implementations and accepting that JAX on CPU is no faster (and at batch 1 slower) than Rust, so the JAX env is for the cloud only.
5. **Pi 5:** rendering only. Inference there is feasible in principle (~12% of CPU for the 60x5 Hz load, ESTIMATE) but would compete with Chromium/WASM; online learning on the Pi is not recommended (~0.6 s/SGD step, ESTIMATE).

### Open questions the research could not resolve
- Does the i7-13700KF desktop have a discrete GPU? If yes (any 8GB+ NVIDIA card), Phase A on-prem becomes viable with PyTorch CUDA and the cloud is optional; the recommendation above assumes no GPU.
- Actual flock-sim v0.4 cost per agent-step (needed to validate the 10^4-10^5 steps/s estimate for the PyO3 route).
- Current burn ONNX-import support for GRU/LSTM (file not fetched); irrelevant if burn models are defined natively and loaded from safetensors.
- Whether PyTorch's new onnxscript GRU/LSTM exporter is in a stable release; the single-step-cell export pattern sidesteps this.
- Which Unraid host will run the brain, and whether it is acceptable to pin ~2-4 cores to it continuously.
- No empirical CPU benchmark exists for small CNN+GRU nets in ort/tract/burn; a half-day spike (export a 2M-param cell, time batch-60 forward and a 512-sample train step on both CPUs) should precede architecture lock-in.

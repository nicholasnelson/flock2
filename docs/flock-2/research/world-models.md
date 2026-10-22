# World models for small embodied agents — survey and FLOCK 2.0 fit

Research notes, 2026-10-22. Scope: learned world models (DreamerV3 / Dreamer 4 / TD-MPC2 / Ha&Schmidhuber / IRIS / STORM / DIAMOND / SimPLe), surprise-as-signal (Plan2Explore, curiosity), multi-agent world models (MAMBA, MARIE, CoDreamer, Dreamer-CPC), and CPU-feasibility arithmetic for tens of agents with 16x16, 16-colour egocentric views. Anything marked **ESTIMATE** is my own derivation, not a published number.

---

## KQ1. DreamerV3: architecture, smallest published sizes, compute, sample efficiency, tiny-image work

### Takeaway
DreamerV3 is an RSSM (GRU + 32x32 categorical latents) trained with symlog inputs, free bits (1 nat) and KL balancing; the smallest *published* configs are XS = 8M params (v1 paper) and `size1m` ≈ 1M params (official repo, used for proprioceptive DMC). All published image results use 64x64 inputs on a single V100/A100; no paper-level numbers exist for 16x16 inputs or for CPU training throughput.

### Cited Findings
- **Architecture.** World model is an RSSM: CNN encoder/decoder for images, MLPs for vectors; dynamics, reward and continue predictors are MLPs. Representations are "a vector of softmax distributions" (categorical latents); free bits "clip the dynamics and representation losses below the value of 1 nat ≈ 1.44 bits"; loss weights βpred = 1, βdyn (0.5 in v1 table, 1 in v2 text), βrep = 0.1; categoricals are a mixture of 1% uniform + 99% network output ("unimix"); vector inputs use symlog; rewards/values use symexp twohot regression over bins from −20 to +20; return normalisation S = EMA(Per(R,95) − Per(R,5), 0.99), divisor max(1, S); entropy scale 3e-4, γ = 0.997, imagination horizon H = 15 (v1 Table W.1) / 16 (v2) — [DreamerV3 arXiv:2301.04104 v2 HTML](https://arxiv.org/html/2301.04104v2); [v1 PDF](https://arxiv.org/pdf/2301.04104v1).
- **Latent size and hyperparameters (v1 Table W.1):** 32 latents × 32 classes; replay capacity 1e6 (FIFO); batch size 16; batch length 64; LayerNorm + SiLU; world-model LR 1e-4, actor-critic LR 3e-5; critic EMA decay 0.98 — [v1 PDF](https://arxiv.org/pdf/2301.04104v1).
- **Model-size table (v1 Table B.1):** XS: 256 GRU units, CNN multiplier 24, dense 256, 1 MLP layer, **8M params**; S: 512/32/512/2, **18M**; M: 1024/48/640/3, **37M**; L: 2048/64/768/4, **77M**; XL: 4096/96/1024/5, **200M**. "The encoder consists of stride 2 convolutions of doubling depth until resolution 4 × 4 followed by flattening. The decoder starts with a dense layer, followed by reshaping to 4 × 4 × C and then inverts the encoder" — [v1 PDF](https://arxiv.org/pdf/2301.04104v1). The v2/Nature revision instead reports "six model sizes ... ranging from 12M to 400M parameters" — [v2 HTML](https://arxiv.org/html/2301.04104v2).
- **Official repo size configs (configs.yaml):** `size1m`: rssm deter 512, hidden 64, classes 4, enc/dec depth 4, units 64; `size12m`: deter 2048, hidden 256, classes 16, depth 16, units 256; `size25m`: 3072/384/24/24/384; `size50m`: 4096/512/32/32/512; `size100m`: 6144/768/48/48/768; `size200m` (default): 8192/1024/64/64/1024; `size400m`: 12288/1536/96/96/1536. Defaults: batch_size 16, batch_length 64, replay size 5e6, train_ratio 32; encoder/decoder "simple" CNN with kernel 5, depth 64, mults [2,3,4,4], 3 layers. The `dmc_proprio` task config uses `size1m` with train_ratio 1024; `crafter` uses train_ratio 512 with 1 env; `atari100k` train_ratio 256. There is also a `debug` config that runs on `jax.platform: cpu` with a toy agent (deter 8, hidden 3) and a `multicpu` config using 8 mock devices — [danijar/dreamerv3 configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml).
- **Compute (v1 Table A.1):** every agent trained on "1 Nvidia V100 GPU each". GPU-days per run: DMC proprio <1 (S, train ratio 512), DMC vision <1 (S, 512), Crafter 2 (XL, 512, 1M env steps), BSuite <1 (XL, 1024), Atari 100K <1 (S, 1024), Atari 200M 16 (XL, 64), DMLab 4 (XL, 64), Minecraft 17 (XL, 16). "The train ratio is the number of replayed steps per policy steps" — [v1 PDF](https://arxiv.org/pdf/2301.04104v1). v2 states "All Dreamer agents are trained on a single Nvidia A100 GPU each" — [v2 HTML](https://arxiv.org/html/2301.04104v2).
- **Atari-100k compute comparison (v1 Table T.1, V100-days):** SimPLe 10, EfficientZero 1.2, SPR 0.2, IRIS 7, TWM 0.8, DreamerV3 0.5 — [v1 PDF](https://arxiv.org/pdf/2301.04104v1).
- **Sample efficiency / scaling:** "Higher training ratios result in substantially improved data-efficiency. Notably, larger models achieve not only higher final performance but also higher data-efficiency" (Figure 6, train ratios 1–64, sizes 8M–200M). "DreamerV3 learns to collect diamonds in 17 GPU days from sparse rewards and without human data" vs VPT's 720 V100 GPUs for 9 days — [v1 PDF](https://arxiv.org/pdf/2301.04104v1). DMLab: "In 100M frames, Dreamer exceeds the performance of the scalable IMPALA and R2D2+ agents at 1B environment steps" — [v2 HTML](https://arxiv.org/html/2301.04104v2).
- **Crafter (64x64 top-down, closest proxy to FLOCK):** DreamerV3 score 14.5 ± 1.6 %, reward 11.7 ± 1.9; DreamerV2 10.0 ± 1.2 %; PPO 4.6 %; Rainbow 4.3 %; Plan2Explore (unsupervised) 2.1 %; human experts 50.5 ± 6.8 % (v1 Table M.1) — [v1 PDF](https://arxiv.org/pdf/2301.04104v1). Crafter itself: 22 achievements, 17 discrete actions, score = geometric mean of success rates — [Crafter arXiv:2109.06780](https://arxiv.org/abs/2109.06780).
- **MiniGrid with DreamerV3:** a 2025 paper ran DreamerV3 on MiniGrid for 3M steps using the XL (200M) config with train ratio 64, ~12 h on one V100 + Xeon Gold 6150 (48 h for Crafter) — [arXiv:2503.21047](https://arxiv.org/pdf/2503.21047). No published MiniGrid result with the small configs was found.
- **Small DreamerV3 on CPU at inference (closest hard datapoint):** an air-hockey robot team shrank DreamerV3 to "a total of 9.7 M learnable parameters spread across the world model, actor and critic networks. With this model, the mean inference time of our agents was tested to be 5.2 ms on the CPU of the evaluation server" (20 ms control cycle; proprioceptive inputs, not images). Training: single workstation, Ryzen 9 7950X + RTX 4090, batch 32, train ratio 512, replay 1e7, 100M sim steps — [arXiv:2406.00518](https://arxiv.org/pdf/2406.00518).
- **Other small-model uses:** DIAMOND's comparison table lists the DreamerV3 Atari-100k agent at **18M params, < 1 training day** — [DIAMOND arXiv:2405.12399 HTML](https://arxiv.org/html/2405.12399v2); STORM's table lists DreamerV3 Atari-100k at 12 V100-hours — [STORM arXiv:2310.09615](https://ar5iv.labs.arxiv.org/html/2310.09615).

### Inferences
- The XS (8M) and `size1m` configs show the authors consider ~1–8M-parameter RSSMs legitimate; `size1m` is only used officially for proprioceptive inputs, so a ~1M image-RSSM would be an extrapolation (nothing in the literature says it fails, nothing says it works).
- The "doubling depth until 4×4" encoder rule means a 16x16 input needs only two stride-2 convs; most of DreamerV3's CNN cost (which is sized for 64x64) disappears at 16x16.
- Train ratio dominates compute: at train_ratio 512 (Crafter) each environment step costs 512 replayed training steps. For FLOCK the lever is lowering the train ratio, not just the model size.

### Gaps
- No published FLOPs-per-step or steps/second for any DreamerV3 size on CPU; no published results at 16x16 resolution. The Nature 2025 version is paywalled (redirect to login) so I could not quote its revised tables; the v2 arXiv text is the same revision.

---

## KQ2. Dreamer 4 — what changed, relevance at FLOCK scale

### Takeaway
Dreamer 4 replaces the RSSM with a 2B-parameter transformer video world model (400M causal tokenizer + 1.6B block-causal dynamics trained with "shortcut forcing"), trained on 256–1024 TPU-v5p; its ideas (few-step shortcut sampling, x-prediction, learning from unlabeled video) are relevant conceptually, but the system is orders of magnitude beyond FLOCK's compute envelope.

### Cited Findings
- **Tokenizer:** causal (in time) encoder/decoder with bottleneck so frames decode one at a time; per-timestep patch tokens plus learned latent tokens projected to a small channel dim with tanh; loss MSE + 0.2·LPIPS; masked autoencoding drops input patches with p ~ U(0, 0.9) which improved spatial consistency — [Dreamer 4 arXiv:2509.24527 HTML](https://arxiv.org/html/2509.24527v1).
- **Dynamics:** block-causal transformer over interleaved action, signal-level/step-size tokens and tokenizer representations, with register tokens; trained with *shortcut forcing* (builds on diffusion forcing and shortcut models); inference uses **K = 4 forward passes per generated frame** (step size 1/4); uses x-prediction (clean-representation prediction) because it "enables high-quality rollouts of arbitrary length"; context corrupted to τctx = 0.1 — [same](https://arxiv.org/html/2509.24527v1).
- **Size and compute:** "2B parameters: 400M for the tokenizer and 1.6B for the dynamics model"; trained on 256 to 1024 TPU-v5p with batch 1 per device and FSDP; Minecraft: 256 spatial tokens, 192-frame context; source video 360p at 20 FPS; VPT contractor dataset ≈ 2541 h — [same](https://arxiv.org/html/2509.24527v1).
- **Changes vs DreamerV3:** transformer world model instead of RSSM; policy objective PMPO (sign of advantages); one imagined rollout per context; reward/value heads keep symexp twohot; three-phase training (pretrain world model on video+actions → finetune with task embeddings, BC + reward → imagination training with frozen transformer) — [same](https://arxiv.org/html/2509.24527v1).
- **Claims:** "real-time interactive inference on a single GPU"; first agent to obtain diamonds in Minecraft purely from offline data (diamonds in 0.7 % of episodes; iron pickaxe 29 %); 100× less data than VPT — [abstract](https://arxiv.org/abs/2509.24527), [HTML](https://arxiv.org/html/2509.24527v1).

### Inferences
- Clearly too large for FLOCK: 2B params and TPU pods vs an i7 with no GPU. The transferable ideas are (a) few-step shortcut/consistency sampling if FLOCK ever used a diffusion-style predictor, and (b) that offline pre-training on logged video of the sim is a sound way to spend the one-off cloud GPU budget.

### Gaps
- Exact inference FPS and GPU were not stated in the HTML text I read.

---

## KQ3. TD-MPC2 — decoder-free latent model; sizes; pixel inputs; CPU inference

### Takeaway
TD-MPC2 is an all-MLP, decoder-free latent world model (SimNorm latents, 5-member Q ensemble, MPPI planning with 512 samples × 6 iterations × horizon 3); published sizes run 1M → 317M; pixel tasks use 64x64 with a 4-layer conv encoder; no CPU numbers are published, and the planner's 512-sample MPPI is the dominant per-step cost.

### Cited Findings
- **Architecture:** five components — encoder, latent dynamics, reward, terminal value (Q), policy prior — all MLPs with LayerNorm + Mish; encoder 2–5 layers by size, others 3-layer MLPs; latent normalised by SimNorm (simplices of V = 8, τ = 1); Q ensemble of 5 with 1 % dropout, TD target = min of 2 random Qs; rewards/values use 101-bin discrete regression in log space — [TD-MPC2 arXiv:2310.16828 HTML](https://arxiv.org/html/2310.16828v2).
- **Sizes (Table 9):** 1M: enc dim 256, MLP 384, latent 128, 2 enc layers, 2 Qs; 5M (default): 256/512/512/2/5; 19M: 1024/1024/768/3/5; 48M: 1792/1792/768/4/5; 317M: 4096/4096/1376/5/8 — [same](https://arxiv.org/html/2310.16828v2).
- **Planning (Table 8):** horizon 3, 6 iterations (+2 if action dim ≥ 20), population 512, 24 policy-prior samples, 64 elites, temperature 0.5 — [same](https://arxiv.org/html/2310.16828v2).
- **Compute:** multi-task training on one RTX 3090: 3.7 GPU-days (1M), 4.2 (5M), 5.3 (19M), 12 (48M), 33 (317M); planning throughput improved "approx. 2×" via code optimisations; no steps/second or CPU figures — [same](https://arxiv.org/html/2310.16828v2).
- **Pixels:** 10 DMControl tasks at 64×64 with random shift; "we replace the default MLP encoder of TD-MPC2 with a shallow (4 layers) convolutional encoder"; comparable to DrQ-v2 and DreamerV3 with no hyperparameter change — [same](https://arxiv.org/html/2310.16828v2). Abstract: 317M agent on 80 tasks; 104 online tasks with one hyperparameter set — [abstract](https://arxiv.org/abs/2310.16828).

### Inferences
- Decoder-free means no pixel reconstruction, so "surprise" would have to be latent-space prediction error or Q/reward disagreement rather than a visible predicted frame. That is a disadvantage for an art installation that wants to *show* what the sheep expected.
- The 1M config is tiny, but 512 × 6 × 3 latent rollouts per action step (~9k dynamics evaluations) is expensive compared with Dreamer's single actor forward pass. See KQ7 for arithmetic.

### Gaps
- No published inference latency; the appendix's "55M default" vs "5M" is an inconsistency the paper itself does not resolve.

---

## KQ4. Simpler alternatives: Ha & Schmidhuber, PlaNet/SimPLe, IRIS, STORM, DIAMOND, MuZero-style

### Takeaway
The original "World Models" (VAE 32-dim latent + 256-unit MDN-RNN ≈ 0.42M params + 867-param linear controller) is the explicit precedent for tiny per-agent models and already documents the dream-exploitation failure mode; transformer/diffusion world models (IRIS 30M, STORM 2L×512D, DIAMOND 4–13M with 3 denoising steps) buy Atari fidelity at 3–7 GPU-days per game and are over-budget for FLOCK.

### Cited Findings
- **Ha & Schmidhuber 2018:** input resized to 64×64; z = 32 (CarRacing) / 64 (VizDoom); MDN-RNN LSTM 256 (CarRacing) / 512 (VizDoom) hidden units, 5 Gaussian mixtures; controller "a simple single layer linear model" with 867 / 1,088 params. Parameter counts: VAE 4,348,547 / 4,446,915; MDN-RNN **422,368** / 1,678,785; controller 867 / 1,088. Procedure: 10,000 random rollouts → train VAE (1 epoch) → train MDN-RNN (20 epochs) → CMA-ES on controller (population 64, 16 rollouts each, 64-core machine); each model under an hour on one GPU — [arXiv:1803.10122 HTML](https://arxiv.org/html/1803.10122v4).
- **Temperature / exploitation:** raising τ "adds uncertainty to the dream, making it harder"; at τ = 0.10 dream score 2086 ± 140 but real score 193 ± 58 (below random 210 ± 108); best transfer at τ = 1.15 (1092 ± 556); too high τ makes the dream too hard to learn. The agent "found an adversarial policy in which the model's monsters never fired fireballs" — imperfect model exploited. Paper proposes an iterative loop and suggests flipping the sign of M's loss for curiosity; C is deliberately "as simple and small as possible" — [same](https://arxiv.org/html/1803.10122v4).
- **SimPLe:** video-prediction world model, 100K agent interactions (400K frames, ~2 h play); "about 3 weeks on a P100" per IRIS's comparison; 10 V100-days per DreamerV3's table; 33 mean HNS (DreamerV3 Table T.1) — [SimPLe arXiv:1903.00374](https://arxiv.org/abs/1903.00374v5); [IRIS](https://ar5iv.labs.arxiv.org/html/2209.00588); [DreamerV3 v1](https://arxiv.org/pdf/2301.04104v1).
- **IRIS:** VQGAN-style discrete autoencoder (no discriminator, perceptual loss), 64×64 frames → K = 16 tokens, vocab 512, embedding 512; GPT (minGPT) 10 layers, D = 256, 4 heads, context 20 timesteps; actor-critic LSTM 512; imagination horizon 20; 600 epochs; "training takes around 7 days, resulting in an average of 3.5 days per environment" on 8× A100 40GB; mean HNS 1.046, IQM 0.501. Failure modes: with 16 tokens "the autoencoder often erases the player, switches colors, and misplaces rewards" (64 tokens helps); "double exploration" (must discover an event before the model can imagine it); sparse-reward Freeway needed temperature 0.01 — [IRIS ar5iv](https://ar5iv.labs.arxiv.org/html/2209.00588). DIAMOND lists IRIS at 30M params, 4.1 training days, 16 NFE per frame — [DIAMOND HTML](https://arxiv.org/html/2405.12399v2).
- **STORM:** categorical VAE → 32 categoricals × 32 classes per frame; GPT-like transformer K = 2 layers, D = 512, 8 heads, dropout 0.1; WM batch 16 × length 64; imagination batch 1024, context 8, horizon 16; 126.7 % mean HNS; "4.3 hours" on one RTX 3090 for Atari-100k (9.3 V100-h) vs DreamerV3 12 V100-h, TWM 10 A100-h, IRIS "7 days for two runs" on A100; KV-cache for imagination; parameter count not stated — [STORM ar5iv](https://ar5iv.labs.arxiv.org/html/2310.09615); official config `wm_2L512D8H` — [weipu-zhang/STORM](https://github.com/weipu-zhang/STORM).
- **DIAMOND:** EDM diffusion, 2D U-Net conditioned on last 4 frames + actions via adaptive group norm; 64×64×3; n = 3 denoising steps (1 step blurs stochastic games like Boxing; "3 NFE per frame compared to 16 NFE per frame for IRIS"); Table 4: 13M params DIAMOND / 30M IRIS / 18M DreamerV3 (text elsewhere says "4M for our Atari models", unresolved); 2.9 days per run on an RTX 4090 ("1.03 GPU years in total"); per-update 543 ms; imagination step 20.4 ms, next-obs prediction 12.7 ms, 4.2 ms per denoising step; Atari-100k mean HNS DIAMOND 1.459, STORM 1.266, DreamerV3 1.097, IRIS 1.046; CS:GO model 381M (incl. 51M upsampler), 10 Hz on an RTX 3090 — [DIAMOND HTML](https://arxiv.org/html/2405.12399v2).
- **Craftax (JAX Crafter) as cheap sim/benchmark:** pixel obs 63×63×3 (Classic) / 110×130×3; symbolic obs 1345 / 8268 floats; up to 405,618 steps/s (Classic) with 4096 workers vs Crafter 1,580 steps/s; "symbolic ... around 10x faster" than pixels; 1B steps in ~1 h on one RTX 4090; baselines PPO, PPO-RNN, ICM, E3B (no model-based baselines in the main text) — [Craftax arXiv:2402.16801](https://arxiv.org/html/2402.16801v1).

### Inferences
- The Ha & Schmidhuber MDN-RNN at ~0.4M params on a 32-dim latent is the closest published analogue to a "1M-param per-sheep model" and it produced convincing dream rollouts; its decoupled V/M/C training (train V and M offline, cheap controller online) maps well onto FLOCK's "offline pre-train on rented GPU, cheap online adaptation" plan.
- IRIS/STORM/DIAMOND are optimised for Atari-100k fidelity on one GPU per game; none is designed for tens of concurrent CPU-resident models.
- MuZero/Muesli-style value-equivalent models were not fetched in this pass (budget); they predict reward/value/policy only, so they give no visible "expected frame" — the same drawback as TD-MPC2 for an art piece.

### Gaps
- PlaNet (Hafner 2019) full text was not fetched; it is the RSSM origin (DreamerV3 inherits it). No MuZero/Muesli primary-source numbers are included here.

---

## KQ5. Surprise / prediction error as a visible behavioural signal; ensembles and their cost

### Takeaway
Plan2Explore defines surprise as the variance across a 5-member ensemble of lightweight 2-layer MLP one-step heads predicting the 1024-dim next encoder embedding from the RSSM state, which "adds little computational overhead" and separates epistemic novelty from aleatoric noise; plain prediction-error curiosity (Burda et al.) works but is attracted to unpredictable "noisy-TV" stimuli.

### Cited Findings
- **Plan2Explore mechanism:** bootstrap ensemble of one-step models takes RSSM state s_t and action a_t and predicts the next image embedding h_{t+1}; trained with squared error (Gaussian, fixed variance); **5 members, each a 2-hidden-layer MLP**, predicting 1024-dim encoder features; intrinsic reward D(s_t,a_t) = 1/(K−1) Σ_k (μ_k − μ')², scaled by 10,000; computed in latent space so "no images are generated during policy learning"; chosen over variants predicting deterministic/stochastic RSSM features because of "the large parametrization that comes with predicting the large encoder features"; ensemble "adds little computational overhead" (members train in parallel); no ensemble-size ablation and no timing numbers — [Plan2Explore arXiv:2005.05960 HTML](https://arxiv.org/html/2005.05960v2).
- **Why disagreement rather than raw prediction error:** prediction error is "retrospective" (computed after reaching a state); disagreement approximates expected information gain, targets epistemic uncertainty, and "given enough samples, it eventually reduces to zero even for stochastic environments" — [same](https://arxiv.org/html/2005.05960v2).
- **Results:** zero-shot DMC average Plan2Explore 563.58 vs Curiosity 489.26 vs Random 342.11 vs supervised Dreamer 694.77; few-shot 643.73 vs Dreamer 700.27 — [same](https://arxiv.org/html/2005.05960v2). On Crafter, Plan2Explore (unsupervised) scored 2.1 % vs RND 2.0 % vs random 1.6 % — [DreamerV3 v1 Table M.1](https://arxiv.org/pdf/2301.04104v1).
- **Prediction-error curiosity at scale (Burda et al.):** curiosity-only agents across 54 environments; random features work surprisingly well, learned features generalise better; identifies the "noisy TV" weakness: agents are drawn to unpredictable stimuli, which slows progress — [arXiv:1808.04355](https://arxiv.org/abs/1808.04355).
- **Change-based intrinsic motivation inside DreamerV3:** a 2025 paper adds a change-based intrinsic reward to DreamerV3 and evaluates on MiniGrid and Crafter (XL config, ~12 h / 48 h on one V100) — [arXiv:2503.21047](https://arxiv.org/pdf/2503.21047).
- **Ha & Schmidhuber curiosity suggestion:** flip the sign of the M model's loss so the controller seeks states the model predicts badly — [arXiv:1803.10122](https://arxiv.org/html/1803.10122v4).

### Inferences
- For a *visible* surprise signal FLOCK does not need a correct exploration objective; it needs a well-scaled scalar per step. Three cheap candidates, in increasing cost: (1) KL(posterior ‖ prior) of the RSSM, free in DreamerV3's loss; (2) reconstruction error of the decoded predicted frame (gives a displayable "expected view"); (3) a K = 3–5 ensemble of small MLP heads à la Plan2Explore for epistemic-only surprise. The Plan2Explore heads are MLPs on the latent, so cost scales with K × (latent dim × hidden) and is small relative to the encoder/decoder (see KQ7 ESTIMATE).
- "Noisy TV" matters for FLOCK: other sheep are a stochastic stimulus; pure prediction-error curiosity will make sheep stare at each other. Disagreement (epistemic) or a learned-feature target damps this.

### Gaps
- No paper quantifies ensemble cost in FLOPs; no work studies prediction error as an *aesthetic* / behavioural display signal.

---

## KQ6. Multi-agent world models: per-agent vs shared, parameter sharing with individual state

### Takeaway
All multi-agent Dreamer variants found (MAMBA, MARIE, MA-Dreamer, CoDreamer, Dreamer-CPC) share world-model parameters across agents while keeping a separate recurrent/latent state per agent; the shared models are then conditioned by communication/attention over other agents' latents. Nothing was found that trains tens of *independent* per-agent world models, and nothing at FLOCK's agent count (SMAC maps are 2–~27 agents).

### Cited Findings
- **MAMBA (Egorov & Shpilman 2022, AAMAS):** DreamerV2 backbone; per-agent RSSM state hⁱ_t, zⁱ_t (32 categoricals × 32 classes, "160 bits per message"); "Recurrent model consists of 1 layer GRU"; "In order to facilitate independence between hidden states h_t, we predict individual reward, discount and observation only from agent's latent state"; the transition function uses a 3-layer Transformer "Communication Block" over all agents' (z, a) with positional encoding "to distinguish agents in homogeneous settings"; mutual-information loss (predict own previous action) to disentangle agents' latents; "Each agent has its own version of the world model, which can be updated using the Communication Block" at execution; imagination horizon H = 15; PPO actor with "parameter sharing for the actor network"; critic attends over all latents; absorbing states for dead agents — [MAMBA arXiv:2205.15023 PDF](https://arxiv.org/pdf/2205.15023). Abstract: SMAC and Flatland, "up to an orders of magnitude" fewer interactions, "scale gracefully with the number of agents" — [abstract](https://arxiv.org/abs/2205.15023).
- **MARIE (2024):** VQ-VAE "(E,D,Z) is shared across all agents" (MLP enc/dec, 3 layers × 512, codebook 512, K = 16 tokens, code dim 128, EMA to avoid codebook collapse); "a shared Transformer as a local dynamic model" (256 emb, 10 layers, 4 heads); Perceiver cross-attention compresses all agents' n(K+1) tokens to n features, each inserted into that agent's own token stream; each agent's prediction conditions on its own history; horizons 15/8/5 by agent count; param size MARIE 55.35 MB vs MAMBA 23.72 MB vs MAPPO 0.24 MB; trained on one RTX 3090; fully centralised ablation "demands over twice the original computational time"; ablation across "2 to 7 agents"; win rates e.g. 3s_vs_5z 78.4 vs MAMBA 13.4 — [MARIE arXiv:2406.15836 HTML](https://arxiv.org/html/2406.15836v2).
- **Per-agent independent models:** CoDreamer/IDreamer create "distinct Dreamer agents, each using its own separate world model", then add GNN communication — [CoDreamer arXiv:2406.13600](https://arxiv.org/html/2406.13600v1); Dreamer-CPC gives each agent its own DreamerV3 RSSM trained on local obs + received messages — [arXiv:2607.19809](https://arxiv.org/pdf/2607.19809); MA-Dreamer uses a standard world model per agent plus a global model used only in training — [arXiv:2204.04687](https://arxiv.org/pdf/2204.04687).
- **Interactive local models:** "Models as Agents" optimises multi-step predictions of interactive local models in MBMARL — [arXiv:2303.17984](https://arxiv.org/pdf/2303.17984).

### Inferences
- The field's standard answer to "individual memory, shared learning" is exactly MAMBA's: one parameter set, N recurrent states, with any agent identity carried by the state (and optionally an ID embedding). This is also the only configuration that makes CPU training affordable for tens of agents (KQ7).
- Independent per-agent models (CoDreamer/Dreamer-CPC) exist but are run with a handful of agents on GPUs; none reports cost at 20–50 agents.

### Gaps
- No study of behavioural *diversity* arising from per-agent vs shared world models; no agent counts above SMAC scale; MAMBA's Flatland agent counts were not in the text I extracted.

---

## KQ7. Practical numbers: FLOPs for a ~1M-param RSSM on 16x16x16 input; online training on 8 cores for 20–50 agents

### Takeaway
No published numbers exist for this regime; **ESTIMATE**: a ~1–2M-param DreamerV3-style RSSM on 16×16×16 one-hot input costs ≈ 4 MFLOP per step for acting (encoder + RSSM + actor) and ≈ 9 MFLOP with the decoder, so 50 sheep at 10 Hz is ≈ 5 GFLOP/s — trivial on either i7. Training is the constraint: ≈ 50–70 GFLOP per gradient update (batch 16 × 32 steps + imagination), i.e. roughly 1–3 updates/s on 8 cores, which is enough for **one shared model** at a low train ratio but not for 50 independently trained models.

### Cited Findings (anchors used for the estimate)
- DreamerV3 encoder: stride-2 convs doubling depth until 4×4, then flatten; XS = 256 GRU units, CNN multiplier 24, dense 256, 1 MLP layer, 8M params — [v1 PDF](https://arxiv.org/pdf/2301.04104v1).
- Repo `size1m`: deter 512, hidden 64, 4 classes; batch 16 × 64, train ratio 32 default, 512 for Crafter — [configs.yaml](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml).
- 9.7M-param DreamerV3 agent: 5.2 ms mean CPU inference (vector input) — [arXiv:2406.00518](https://arxiv.org/pdf/2406.00518).
- Ha & Schmidhuber MDN-RNN 422k params; CMA-ES controller optimisation ran on a 64-core CPU machine — [arXiv:1803.10122](https://arxiv.org/html/1803.10122v4).
- DIAMOND per-update 543 ms and 20.4 ms per imagination step for a 13M diffusion model on an RTX 4090 — [DIAMOND](https://arxiv.org/html/2405.12399v2).

### ESTIMATE (my derivation; multiply-accumulates (MAC) ×2 = FLOPs)
Assumed "FLOCK-S" model: input 16×16×16 (one-hot colour planes), conv 3×3 s2 → 8×8×32, conv 3×3 s2 → 4×4×64, flatten 1024 → dense 256; RSSM with 16 categoricals × 16 classes (stoch = 256), GRU deter 256, prior/posterior MLPs 256; mirrored decoder; reward/continue/actor/critic 2×256 MLPs.
- Parameters: encoder ≈ 0.29M; RSSM ≈ 0.8M; decoder ≈ 0.29M; heads ≈ 0.5M → **≈ 1.9M params** (a 128-wide version lands near 0.6M).
- Forward MACs per step: encoder ≈ 0.85M; RSSM (input dense + GRU + posterior) ≈ 0.7M; actor ≈ 0.13M → **≈ 1.7M MAC ≈ 3.5 MFLOP to act**. Adding the decoder (≈ 2.6M MAC) and prior head for a displayed "expected frame" → **≈ 4.4M MAC ≈ 9 MFLOP**. A Plan2Explore-style K = 5 ensemble of 2×256 MLP heads predicting a 256-dim embedding adds ≈ 5 × 0.13M ≈ 0.65M MAC (≈ 15 %).
- Inference load: 50 agents × 10 Hz × 9 MFLOP ≈ **4.5 GFLOP/s**; at 30 Hz ≈ 14 GFLOP/s. An 8-core AVX-512 i7-7820X or a 13700KF realistically sustains 100–300 GFLOP/s in PyTorch/JAX on batched small matmuls (my assumption, not measured), so inference for all sheep, batched into one call per tick, is < 10 % of one machine. The Pi 5 is not in the loop for inference under FLOCK's plan.
- Training: one world-model update on batch 16 × length 32 = 512 timesteps × 9 MFLOP × ~3 (fwd+bwd) ≈ 14 GFLOP; actor-critic imagination from 512 start states × horizon 15 × (RSSM prior + heads ≈ 1 MFLOP) × 3 ≈ 25–45 GFLOP → **≈ 50–70 GFLOP per gradient update**. At 100–200 GFLOP/s effective: **≈ 1.5–4 updates/s for a single shared model.** Fifty independently-trained copies would need 50× that (≈ 3 TFLOP/s for 1 update/s each) — not feasible on CPU; it is feasible only if per-agent models update at ≈ 1 update every 15–50 s each, or are tiny (≤ 0.2M params) and updated round-robin.
- Train ratio implied: at 50 sheep × 10 Hz = 500 env steps/s and ~2 updates/s × 512 replayed steps = ~1000 replayed steps/s → **train ratio ≈ 2**, versus 32–512 in DreamerV3's configs. Sample efficiency will be far below published Dreamer behaviour unless the model is pre-trained offline on the rented GPU and only fine-tuned online.
- Memory: replay of 16×16×16 one-hot = 4 KB/step (or 128 B if stored as 4-bit palette indices); 50 sheep × 1 h at 10 Hz = 1.8M steps = 230 MB as palette indices. Trivial.

### Gaps
- All figures above are estimates; nobody has published throughput for an RSSM at this resolution on CPU. The real numbers depend heavily on framework overhead for small ops (JAX/PyTorch dispatch can dominate at 1M-param scale), so a 1-day microbenchmark is needed before committing.

---

## KQ8. Failure modes with small models / small data and remedies

### Takeaway
Documented failures are: policies exploiting model errors (dream-hallucination), compounding rollout error, posterior/representation collapse, codebook collapse in discrete tokenizers, erased small objects in low-capacity tokenizers, and "double exploration" (can't imagine what hasn't been seen); remedies are free bits + KL balancing + unimix (DreamerV3), sampling temperature/noise in dreams (Ha & Schmidhuber), short or branched horizons (MBPO/Dreamer H = 15), EMA codebooks (MARIE), and more tokens/capacity for small objects (IRIS).

### Cited Findings
- **Exploiting the model:** at low dream temperature the CarRacing/VizDoom agent scores 2086 in the dream but 193 in reality (worse than random); monsters "never fired fireballs" in the dream; raising τ to 1.15 fixed transfer — [arXiv:1803.10122](https://arxiv.org/html/1803.10122v4).
- **Collapse guards in DreamerV3:** free bits (1 nat), small representation loss (βrep = 0.1 vs βdyn), 1 % unimix "so the encoder and dynamics distributions can't become deterministic", symlog on inputs to limit large reconstruction gradients — [v2 HTML](https://arxiv.org/html/2301.04104v2); [v1 PDF](https://arxiv.org/pdf/2301.04104v1). KL balancing: prior pulled toward posterior faster than posterior toward prior (δ ≈ 0.8 in DreamerV2-style write-ups) — [DITTO arXiv:2302.03086](https://arxiv.org/pdf/2302.03086); TorchRL notes the reference clamps the summed KL over categoricals rather than per-categorical — [TorchRL DreamerV3ModelLoss](https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.DreamerV3ModelLoss.html).
- **Compounding error / horizon:** MAMBA sets H = 15 explicitly "to account for ... possible compounding error of the world model" — [MAMBA PDF](https://arxiv.org/pdf/2205.15023); MARIE reports lower compounding prediction error than MAMBA and attributes its win-rate gap to "more accurate, consistent imaginations" — [MARIE](https://arxiv.org/html/2406.15836v2); a 2026 preprint argues DreamerV3 imagined rollouts are "kinematic, not dynamic" on walker-walk (unreviewed) — [arXiv:2607.05966](https://arxiv.org/abs/2607.05966).
- **Tokenizer capacity:** IRIS with 16 tokens "often erases the player, switches colors, and misplaces rewards"; 64 tokens helps; "double exploration" and sparse-reward discovery problems — [IRIS](https://ar5iv.labs.arxiv.org/html/2209.00588). MARIE uses EMA codebook updates "to avoid codebook collapse" — [MARIE](https://arxiv.org/html/2406.15836v2).
- **Single-step diffusion blurs stochastic outcomes** (interpolates modes) — DIAMOND uses 3 steps — [DIAMOND](https://arxiv.org/html/2405.12399v2).
- **Latent disagreement vanishes under aleatoric noise given enough data**, unlike prediction error — [Plan2Explore](https://arxiv.org/html/2005.05960v2).

### Inferences
- For FLOCK, "hallucination" is partly a feature (visible wrong expectations), but model exploitation by the policy is still a risk: homeostatic drives learned in imagination could make sheep "eat" imaginary grass. Keeping imagination horizons short (≤ 15), training actors partly on real replay, and the DreamerV3 regularisers address this.
- 16-colour, 16×16 inputs make the IRIS small-object problem acute: a single-pixel predator or grass tile is exactly what a low-capacity tokenizer erases. Reconstruct in one-hot colour space with cross-entropy (categorical pixels) rather than MSE on RGB.

### Gaps
- No paper studies RSSMs at ≤ 2M params on images for collapse behaviour; remedies above are extrapolated from 8M+ models.

---

## FLOCK 2.0 fit assessment (my assessment, not from the literature)

**Minimum model that plausibly gives believable anticipation and surprise.** A DreamerV3-style RSSM with: 2-conv encoder on 16×16×16 one-hot planes, 16×16 categorical latent, GRU 256, mirrored categorical decoder (cross-entropy over 16 colours), reward/continue heads driven by homeostatic drives, and the standard DreamerV3 regularisers (free bits 1 nat, βrep 0.1, 1 % unimix, symlog/twohot where applicable). ESTIMATE ≈ 1–2M params, ≈ 9 MFLOP/step with decoder. This is the smallest architecture in the survey that (a) yields a *displayable* predicted next view, (b) gives a free per-step surprise scalar (posterior‖prior KL and/or decoded-frame error), and (c) has a published lineage down to 8M (XS) and 1M (`size1m`) configs. Ha & Schmidhuber's 0.42M MDN-RNN is the only smaller precedent and lacks the collapse guards.

**Sharing vs. individual.** Follow MAMBA/MARIE: **one shared parameter set, one recurrent state per sheep**, optionally an agent-ID / trait embedding for individuality. Per KQ7, 50 independently trained models is roughly 50× over the CPU training budget; a shared model trained at ≈ 1–3 updates/s on the 13700KF is within budget. Individuality then comes from each sheep's own memory, drives, trait embedding, and — if wanted — a tiny per-sheep head (e.g. last layer of the actor or a per-sheep reward weighting) fine-tuned online, mirroring Ha & Schmidhuber's "small C, big shared M".

**Surprise signal.** Use the RSSM KL and decoded-frame error for the exhibit's visible surprise; add a K = 3 Plan2Explore-style latent-disagreement ensemble (≈ +10–15 % inference FLOPs, ESTIMATE) as the *curiosity reward* so sheep are not perpetually drawn to each other's unpredictable motion (noisy-TV).

**What to pre-train offline.** Spend the one-off GPU on (1) the world model on logged sim video from scripted/random sheep (as Ha & Schmidhuber did with 10k random rollouts and Dreamer 4 does with unlabeled video), and (2) an initial actor-critic; run online only a low train ratio (≈ 1–4) for drift/adaptation. Expect online-only learning from scratch at train ratio ≈ 2 to be far slower than any published Dreamer result (which use 32–1024).

**Clearly too large / wrong shape for FLOCK:** Dreamer 4 (2B, TPU pod), DIAMOND (3 denoising passes, 20 ms/step on a 4090), IRIS (10-layer GPT per agent, days per run), STORM (transformer context + KV cache per agent). TD-MPC2's 1M config is small but its MPPI planner (512 × 6 × 3 latent rollouts per action) and lack of a decoder make it a poor fit for a visible-anticipation art piece.

**Risks to check first (1–2 day microbenchmarks):** framework overhead at ~1M params on CPU (measure JAX vs PyTorch vs a hand-rolled NumPy/ONNX inference loop); whether a 16×16 latent decoder keeps 1-pixel objects (predators/grass) rather than erasing them (IRIS failure mode); and posterior collapse at very small data before pre-training is in place.

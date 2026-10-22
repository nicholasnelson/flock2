# Predator-prey learning, rare-event learning, and death/population dynamics for FLOCK 2.0

Research notes, 2026-10-22. Scope: predator-prey MARL/ALife, learning from rare events, scripted vs learned adversaries, death/respawn and population stability in long-running agent sims, injury-vs-death, fear/one-shot avoidance, and animal-behaviour grounding. Every literature claim carries a link; anything marked **[speculation]** or in the "Inferences" subsections is my own reasoning, not a sourced claim. The final section ("FLOCK 2.0 fit assessment") is the design recommendation and is clearly labelled.

---

## KQ1. Predator-prey MARL / ALife: which setups produced flocking, vigilance, or alarm from learning rather than hand-coding?

### Takeaway
Flocking/selfish-herd behaviour emerges reliably from a pure survival objective **only when the predator's attack rule makes grouping pay** (predator confusion: capture probability falls with nearby prey; or density-dependent/random-launch attack). Vigilance-vs-foraging trade-offs have been evolved but only under specific conditions; **no source I found shows alarm calls emerging from RL in a predator-prey setting** (the only learned-communication predator-prey results are on the predator side, i.e. collaborative hunting).

### Cited Findings

**Evolutionary (Olson/Adami line)**
- Olson et al. 2013 (J R Soc Interface): 2D continuous 512x512 world, 1 coevolved predator vs 50 prey, Markov-network brains, 1,200 generations, 180 replicates per condition, 2,000-step evaluations. Predator moves 3x faster than prey; attack within 5 units with 10-step handling time. **Confusion rule: capture probability P = 1/A_NV, where A_NV = number of prey visible to the predator within 30 units of the target.** Without confusion: swarm density 0.69 ± 0.02, dispersion 46.69 ± 0.44, no replicate formed a cohesive swarm. With confusion: density 12.48 ± 0.8, dispersion 22.54 ± 1.32, cohesive swarms in ~70% of replicates; prey survivorship 34.7 ± 0.6 vs 25.54 ± 0.49 out of 50. Swarming only favoured when predator visual field ≥120° (density 6.13 ± 0.76) and not at ≤60° (0.68 ± 0.02). Functional response was Type II in both cases but the plateau dropped from 24.01 ± 0.49 to 15.18 ± 0.57 prey consumed with confusion. Predators were coevolved (seeded with a pre-evolved tracker) not scripted — [arXiv 1209.3330 full text](https://ar5iv.labs.arxiv.org/html/1209.3330); [abstract](https://arxiv.org/abs/1209.3330v2); [journal entry, DOI 10.1098/rsif.2013.0305](https://gpbib.pmacs.upenn.edu/gp-html/olson_2013_RSIF.html); [MSU press release noting >100 replicates per experiment](https://msutoday.msu.edu/news/2013/swarming-offers-clues-on-how-intelligence-evolved)
- Olson, Knoester & Adami 2016 (Artificial Life), "Evolution of swarming behavior is shaped by how predators attack": **how predators attack is critical** to selfish-herd evolution; density-dependent predation is a sufficient selective advantage for prey to evolve the selfish herd against coevolving predators; the authors present this as corroborating Hamilton's selfish-herd hypothesis in a digital evolutionary model — [arXiv 1310.6012](https://arxiv.org/pdf/1310.6012); [ar5iv full text](https://ar5iv.arxiv.org/html/1310.6012); [CoMSES model entry](https://catalog.comses.net/publications/112097)
- Olson, Haley & Dyer 2015 (J R Soc Interface), vigilance/foraging trade-off: tests gregarious foraging under the "many eyes" hypothesis; finds highly related groups with a semelparous reproductive strategy are the most likely to evolve gregarious foraging mediated by the vigilance benefit — [arXiv 1408.1906](https://arxiv.org/pdf/1408.1906); [PMC4593673](https://pmc.ncbi.nlm.nih.gov/articles/PMC4593673)
- Olson et al. 2016, "Exploring the coevolution of predator and prey morphology and behavior" (arms race incl. prey vision/morphology) — [arXiv 1602.08802](https://arxiv.org/pdf/1602.08802)

**Deep RL (prey learn; predator scripted)**
- Hahn et al. 2019 "SELFish" (ALIFE 2019): prey trained with DQN/DDPG, reward **+1 per step alive, −1000 on collision with predator which ends the agent's life**; when caught, "a new agent is spawned immediately at a random position to keep the number of agents in the system constant"; predator is **static/hand-coded**: moves toward closest agent, but if several agents are within a certain distance it picks a target at random (distraction), holds a target for a while, and occasionally "leaps" (accelerates) since by default prey and predator move at the same speed. Torus world, no walls, no prey-prey collisions. One policy trained (500,000 steps, DQN observes 5 nearest neighbours, DDPG 1) then copied to all agents; only 10 agents present during training, 60 in demos. Episodes end at capture or 10,000 steps. Result: flocking similar to Boids emerges with no flocking reward; cohesion/alignment measured via DBSCAN clusters; a hand-coded "TurnAway" baseline that ignores neighbours produces many noise points; separation from the swarm precedes capture (their Fig. 11). The authors note TurnAway was possibly never found because learning got stuck in a local optimum once all other agents stayed close — [arXiv 1905.04077 PDF](https://arxiv.org/pdf/1905.04077); [ALIFE 2019 talk](https://2019.alife.org/programme/contrib/104/); [code: fish-rl-alife](https://github.com/instance01/fish-rl-alife/)

**Deep RL (both sides learn)**
- Li, Li & Zhao 2023 (New J. Phys. 25:092001), "Predator–prey survival pressure is sufficient to evolve swarming behaviors": modified MADDPG (decentralised critic, actor shared within species, one replay buffer per species); 3 predators + 10 prey in training (3 vs 5 at evaluation); reward only **−1 to prey / +1 to predator per step of contact ("bleeding"; prey are not removed)**, plus small movement cost and −0.1 boundary penalty in the confined case; 2,000 episodes x 100 steps, batch 256, buffer 5e5. Prey sparsity (DoS) fell ~22% → ~19% and alignment (DoA) rose ~0.65 → ~0.82 (with loaded policies DoS ~15%, DoA ~0.96). With **0 predators no swarming emerged**; 1 predator slowed emergence; a 5:3 predator:prey speed ratio gave DoS ~18%, DoA ~0.85. Perception range of one-third environment size weakened swarming (DoS ~20.5%, DoA ~0.75). Predators showed dispersion tactics (break the swarm), confusion, marginal predation (target edge/isolated prey) — qualitatively. The "swirling" experiment used a **fixed-rule predator (steer at nearest prey at max speed) to speed prey learning**. Target networks were used against non-stationarity; training ran until "dynamic equilibrium"; no further arms-race discussion — [arXiv 2308.12624 HTML](https://arxiv.org/html/2308.12624v1); [journal PDF](https://shiyuzhao.westlake.edu.cn/Li_2023_New_J._Phys._25_092001.pdf); [DOAJ record](https://doaj.org/article/05e3f1e47d65468794a51ea056f2873c)
- Lowe et al. 2017 MADDPG predator-prey ("simple tag"): prey learned, metric is prey touches per episode; MADDPG predators ~11.0 touches vs ~9.4 for DDPG predators (table columns ambiguous in snippet). No scripted-prey evaluation in the paper — [arXiv 1706.02275](https://arxiv.org/pdf/1706.02275); independent-learner comparison on simple tag found MADDPG predators "learned to cooperate better" than DQN — [arXiv 2111.01100](https://arxiv.org/pdf/2111.01100)
- "Collective Adaptation in Multi-Agent Systems: How Predator Confusion Shapes Swarm-Like Behaviors" (2022) also uses the confusion mechanism in a MARL setting — [arXiv 2209.06338](https://ar5iv.labs.arxiv.org/html/2209.06338)
- Baker et al. 2020 hide-and-seek (ICLR 2020): six emergent phases (running/chasing, fort building, ramp use, ramp defence, box surfing, surf defence); each phase created new pressure on the opposing team; authors argue multi-agent competition scales better with environment complexity than intrinsic motivation — [arXiv 1909.07528](https://arxiv.org/abs/1909.07528v2); [ICLR page](https://iclr.cc/virtual/2020/poster/1450)

**Learned communication / coordination (predator side only)**
- Tsutsui et al. 2024 (eLife 13:e85694): deep-RL predators with only associative-learning-level capacity learned collaborative hunting; cooperation driven by difficulty of solo capture plus shared food reward; role differentiation (chaser vs ambusher) emerged; agents adapted to hunting alongside humans — [eLife](https://elifesciences.org/articles/85694); [PMC11076040](https://pmc.ncbi.nlm.nih.gov/articles/PMC11076040); [Nagoya press release](https://www.nagoya-u.ac.jp/researchinfo/result-en/2024/05/20240520-01.html)
- Ritz, Hahn et al. 2020 (ALIFE 2020, "Towards Ecosystem Management from Greedy RL in a Predator-Prey Setting"): a single RL predator hunting reproducing prey learned a **restrained hunting strategy** (hunt only while enough prey remain) via curriculum learning, long-horizon discounting and stacked observations; the right amount of past/predictive information was crucial — [Hahn publication list](https://mobile.ifi.lmu.de/team/carsten-hahn); related 2021 MDPI Entropy "Co-Evolution of Predator-Prey Ecosystems by RL Agents" — [MDPI](https://www2.mdpi.com/1099-4300/23/4/461)

### Inferences
- The common ingredient across Olson 2013/2016, SELFish and Li 2023 is that **the environment's capture rule, not the reward, is what makes grouping pay**: 1/N confusion, random-target distraction, or contact-based bleeding where predators target marginal individuals. With a predator that simply chases the nearest prey and always succeeds, grouping has no payoff and prey learn "TurnAway". **[speculation]** For FLOCK this means the fox's capture mechanic is the main lever for lifelike flocking, more so than the sheep's learner.
- Prey survival reward alone yields flocking + flight; vigilance requires a foraging cost to vigilance and a detection benefit (Olson 2015); alarm signalling has no demonstrated RL emergence in predator-prey sims in the sources found (gap below).

### Gaps
- No source found where **alarm calls** (or any prey-to-prey signalling) emerged from learning in a predator-prey simulation; emergent-communication work in this domain is predator-side (Tsutsui). Ito & Ikegami work was not found under that name.
- Li 2023 does not report survival rates or arms-race dynamics; Olson's confusion results are evolutionary (1,200 generations x 180 replicates), not online RL — sample-complexity numbers for RL prey learning flocking against a confusable predator were not found beyond SELFish's 500k training steps.

---

## KQ2. Learning from rare events (a fox ~once per night)

### Takeaway
The standard toolkit is: (a) prioritise or permanently retain the rare transitions (PER; Lipton's never-flushed "danger buffer"), (b) manufacture more encounters offline (parallel sims, curricula, regret-driven environment design), and (c) use a world model to imagine danger. Published evidence for (c) on *rare* dangers specifically is thin; safe-model-based RL (SafeDreamer, Nightmare Dreamer) shows imagination of unsafe futures works for frequent costs.

### Cited Findings
- Schaul et al. 2016 PER: uniform replay "replays transitions at the same frequency that they were originally experienced, regardless of their significance"; PER beat uniform replay on 41/49 Atari games; Blind Cliffwalk shows random exploration needs an exponential number of steps to the first non-zero reward, and good replay ordering gives exponential speed-ups — [arXiv 1511.05952](https://arxiv.org/pdf/1511.05952)
- Limitation: TD-error priority can stop resampling zero-error experiences after the behaviour policy changes, so rare, high-value event trajectories may be lost; Pinto/Stone et al. propose combining PER with event-trajectory focus — [TMLR 2023 PDF](https://www.cs.utexas.edu/%7Epstone/Papers/bib2html-links/TMLR23.pdf); Tavakoli et al. instead keep an episodic buffer of high-reward starting states — [Prioritizing Starting States](https://mila.quebec/en/?p=35322)
- A 2025 safe-driving paper notes risk costs are "typically close to zero, with only occasional spikes", and uses PER with p_i ∝ |δ_i| for exactly this rare-cost regime — [arXiv 2508.14926](https://arxiv.org/pdf/2508.14926)
- Lipton et al. 2016 "Intrinsic Fear": DQNs forget rare catastrophes once flushed from replay and periodically revisit them (the "Sisyphean curse"). Fix: a supervised fear model F(s) trained on a **danger buffer (states within k_r steps before a catastrophe) that is never discarded** vs a safe buffer, 50/50 minibatches; Q target y = r + max Q(s',a') − λF(s'), λ phased in over k_λ steps. Hyperparameters: Adventure Seeker k_r=5, λ=40; Cart-Pole k_r=20; Atari k_r=5, λ=0.5. Results: in Adventure Seeker all IF models stopped dying within ~14 runs; on Freeway a randomly exploring DQN got zero reward while IF learned; Asteroids higher reward and fewer catastrophes; Seaquest similar catastrophe rate but higher reward. Theorem 1: if the optimal policy visits danger states with prob ≤ ε and rewards in [0,1], the fear-optimal policy is within λε of optimal — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1611.01211); [arXiv abstract](https://arxiv.org/abs/1611.01211v8)
- PAIRED (Dennis et al., NeurIPS 2020): environment adversary rewarded for **regret** = antagonist return − protagonist return, which avoids the unsolvable worst cases of minimax and the lack of structure of domain randomisation; produces a natural curriculum of increasingly complex environments and better zero-shot transfer — [arXiv 2012.02096](https://arxiv.org/abs/2012.02096v2); [NeurIPS page](https://neurips.cc/virtual/2020/public/poster_985e9a46e10005356bbaf194249f6856.html)
- SafeDreamer (ICLR 2024): Lagrangian constraints inside Dreamer planning; "nearly zero-cost performance" on Safety-Gymnasium incl. vision-only tasks — [arXiv 2307.07176](https://arxiv.org/html/2307.07176v3). Nightmare Dreamer (2026 preprint): world model imagines future trajectories and switches to a safety actor if predicted cost exceeds budget; reports near-zero violations and ~20x sample efficiency vs model-free (1e6 vs 1e7 steps) — [arXiv 2601.04686](https://arxiv.org/html/2601.04686v1). FOSP adds a cost decoder to DreamerV3 — [arXiv 2407.04942](https://arxiv.org/pdf/2407.04942)
- Hide-and-seek autocurricula and Olson 2013's 180 replicates x 1,200 generations illustrate that emergent anti-predator behaviour in the literature came from **very large offline encounter volumes**, not from live-rate exposure — [Baker et al.](https://arxiv.org/abs/1909.07528v2); [Olson 2013](https://ar5iv.labs.arxiv.org/html/1209.3330)
- Hamon et al. 2023 shows the compute route: 1M steps with thousands of agents in ~20 minutes on one GPU (JAX) — [ar5iv 2302.09334](https://ar5iv.labs.arxiv.org/html/2302.09334)

### Inferences
- **[speculation]** At one fox visit per in-game night, a live exhibition generates ~30 encounters/month per flock; nothing in the RL literature learns a robust avoidance policy from that. Fox competence must therefore be trained offline (many parallel meadows, fox-dense curriculum), and the live run should only fine-tune lightly, if at all.
- Lipton's danger-buffer idea maps directly onto "replaying fox encounters": keep every pre-attack window permanently, train a fear predictor on it, and shape the drive system with −λF(s). Its theorem gives a principled way to cap how much fear distorts grazing (λε).
- **[speculation]** Dreamer-style imagination is plausible for rehearsing fox encounters *if* the world model has seen enough real ones to model the fox; the cited safe-RL results are all for frequent-cost tasks, so expect imagination to hallucinate rather than extrapolate the fox from a handful of samples.

### Gaps
- No paper found that quantifies how few catastrophic samples suffice for avoidance to generalise ("one-shot fear" in deep RL); Lipton's ~14-death figure is in a toy environment.
- No work found applying PAIRED/UED specifically to rare-hazard frequency scheduling (e.g. an adversary that chooses *when* the predator appears).

---

## KQ3. Scripted vs learned predator; arms-race non-stationarity; self-play populations

### Takeaway
The sources that produced clean prey flocking used scripted or frozen predators at least as scaffolding (SELFish fully scripted; Li 2023 froze the predator for the swirling experiment; Olson seeded the predator with a pre-evolved tracker). Learned adversaries deliver richer arms races (hide-and-seek) but need league/fictitious-self-play machinery to avoid cycling, and even AlphaStar's exploiters degraded late in training.

### Cited Findings
- SELFish: "The policy of the predator is static and does not change over time"; flocking still emerged — [arXiv 1905.04077](https://arxiv.org/pdf/1905.04077)
- Li 2023: fixed-rule predator used "to speed up prey evolution" in the swirling experiment; co-learning otherwise, with target networks to mitigate non-stationarity — [arXiv 2308.12624](https://arxiv.org/html/2308.12624v1)
- Olson 2013: predator populations seeded with a pre-evolved tracker to speed evolution "with negligible effect on outcomes" — [ar5iv 1209.3330](https://ar5iv.labs.arxiv.org/html/1209.3330)
- Baker et al. 2020: fully co-learned hide-and-seek produced six strategy phases, each a response to the opponent's last — [arXiv 1909.07528](https://arxiv.org/abs/1909.07528v2)
- Self-play cycling: plain self-play "can chase cycles indefinitely"; FSP trains a best response against a mixture of past policies; AlphaStar's league approximates FSP with main agents, main exploiters and league exploiters; a later critique found that in later iterations AlphaStar's exploiters "lose the ability to identify the weaknesses" and the main agent was fragile to uncommon counter-strategies — [Self-play survey, arXiv 2408.01072](https://arxiv.org/pdf/2408.01072); [NeurIPS 2023 goal-conditioned exploiters](https://papers.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf); [ApproxED](https://arxiv.org/pdf/2301.08830)
- Unity ML-Agents self-play docs: a larger window of saved opponent snapshots gives a more diverse opponent pool — [ML-Agents Training-Self-Play](https://plastichub.unity.cn/unity-tech-cn/ml-agents/src/commit/3f806353-158b-4cf1-a43c-ff4b5f2cd2fc...docs/Training-Self-Play.md); DiffFP notes unimodal RL policies "overfit to recent opponent behaviors while neglecting previously encountered strategies" — [arXiv 2511.13186](https://arxiv.org/html/2511.13186v1)
- Ritz/Hahn 2020: a learned predator with reproducing prey learned restraint (not over-hunting) only with curriculum + long-horizon discounting + stacked observations — [Hahn publications](https://mobile.ifi.lmu.de/team/carsten-hahn)
- Evidence on scripted vs learned opponents in pursuit-evasion: AgilePE reports self-play predators vs scripted prey win rates fluctuating 0.28–0.58 across training checkpoints, i.e. noisy — [arXiv 2608.14135](https://arxiv.org/pdf/2608.14135)

### Inferences
- **[speculation]** For a months-long installation, a learned fox is a liability: the only published long co-learning arms races (hide-and-seek, AlphaStar) ran with huge compute and still needed leagues; a drifting fox could make the exhibit either a massacre or a toothless visitor overnight. A scripted fox with a small parameter set tuned to a target kill rate is the stable choice; a *frozen* learned fox (trained offline against a league of sheep snapshots) is the richer but still stable middle ground.
- Ritz/Hahn 2020 is the one "learned predator with sustainability" result, and it needed careful engineering; it argues against expecting a naive learned fox to self-limit.

### Gaps
- No controlled study found that directly compares prey learning stability under scripted vs learned predators with quantitative outcomes.

---

## KQ4. Death, respawn, and population stability in long-running sims

### Takeaway
Every long-running MARL/ALife system I found either holds population constant by immediate respawn (SELFish, Neural MMO spawn cap) or lets population float with energy-based death and reproduction and then accepts collapse risk (Hamon 2023: 3 of 5 seeds collapsed). Ecological theory and the NetLogo wolf-sheep model identify explicit resource dynamics, refuges with limited capacity, and absorbing-state awareness as the stabilisers.

### Cited Findings
- Neural MMO v1.3: agents spawn with 10 food/water/health, lose 1 food and 1 water per tick, lose health when starving; "agents are added (spawn) within a designated region at a rate of one per timestep" and the training loop spawns "if number of agents alive < spawn cap"; reward is a discounted survival signal with r_T = −1 at death, γ=0.95, "each rollout corresponds to an agent lifetime"; combat has damage/health so injury is graded (health decreases, regenerates when well fed) but death removes the agent; results: population size magnifies exploration (agents spread to avoid competition) and number of unshared populations magnifies niche formation (8 populations) until saturated (32) — [arXiv 2001.12004](https://arxiv.org/pdf/2001.12004); [Neural MMO 1.0](https://arxiv.org/pdf/1903.00784); [OpenAI blog: agents receive reward 1 per tick alive, no environment resets](https://openai.com/blog/neural-mmo). Neural MMO 2.0: 128 agents standard, 1–1024 supported, task-driven competition — [arXiv 2311.03736](https://arxiv.org/pdf/2311.03736)
- SELFish: captured agent's life ends; "a new agent is spawned immediately at a random position to keep the number of agents in the system constant" — [arXiv 1905.04077](https://arxiv.org/pdf/1905.04077)
- Hamon, Nisioti & Moulin-Frier 2023 (GECCO '23 Companion, not ALIFE): 400x200 grid (~100K cells), up to 1,000 agents, start 330 agents and 160,000 resources, no reset over 1,000,000 steps; energy max 3, decays 0.025/step, +1 per resource; **death if energy < threshold for ≥200 consecutive steps or at age 650**; asexual reproduction if energy above threshold ≥140 steps, offspring mutated (σ=0.02); minimal-criterion reproduction so population self-regulates with resources. Dynamics: Lotka-Volterra-like population/resource oscillations at small scale; over-population nearly depletes resources (local tragedy of the commons) which pushes toward sustainable foraging; **of 5 seeds, 3 collapsed after several generations so hyperparameters required tuning**. Stabilisers: sparse spontaneous resource growth (resources never vanish permanently), latitudinal niches with different regrowth, neighbourhood-dependent regrowth, minimal-criterion reproduction. 1M steps with thousands of agents ≈ 20 min on one GPU; 2,445-parameter RNN agents — [ar5iv 2302.09334](https://ar5iv.labs.arxiv.org/html/2302.09334); [arXiv abstract](https://arxiv.org/abs/2302.09334v1); [dblp GECCO Companion 2023 pp.143–146](https://dblp.org/pid/124/0220)
- Jelly Bean World (ICLR 2020): never-ending, non-episodic testbed; PPO+LSTM learned well for ~1M of 10M steps then declined and sometimes got stuck collecting no reward — evidence that standard RL degrades under indefinite non-episodic runs — [arXiv 2002.06306](https://arxiv.org/pdf/2002.06306)
- NetLogo Wolf Sheep Predation: the sheep-wolves-only variant (infinite grass) "is ultimately unstable"; the sheep-wolves-grass variant where eaten grass regrows after a fixed delay is "generally stable"; stability defined as maintaining itself over time despite fluctuations vs instability as extinction of one or more species; sheep+grass with zero wolves is stable; ABMs capture small-population extinctions that continuous LV equations underestimate — [NetLogo model info](https://ccl.netlogo.org/cm/models/predation/info.html); [model page](https://ccl.northwestern.edu/netlogo/models/WolfSheepPredation); [CT-STEM lesson](https://ct-stem.northwestern.edu/curriculum/print/31); community variants with refuge and predator-mediated coexistence — [Refuge variant](https://ccl.northwestern.edu/netlogo/models/community/Wolf%20Sheep%20Predation%20Refuge)
- Refuge theory (Krivan 1998 review): most theory finds refugia stabilising; refuges protecting a **constant number** of prey give a stable equilibrium (stronger than constant proportion); low refuge carrying capacity stabilises, high refuge capacity loses stability — [Krivan TPB 1998](https://fix.prf.jcu.cz/~krivan/clanky/Krivan-TPB-1998.pdf). Over-large refuges can starve predators to extinction — [DDNS 2012](https://emis.univie.ac.at/journals/HOA/DDNS/Volume2012/508962.abs.html)
- Finite stochastic predator-prey systems always eventually hit the absorbing (extinct) state, on a timescale exponential in system size; spatial coupling of patches lets travelling waves re-seed an extinct patch — [APS 2025 abstract](https://archive.aps.org/ses/2025/j04/4)
- A 2018 ABM wolf–rabbit–grass model with refuges found prey escape behaviour had the largest effect on ecosystem dynamics — [CoMSES entry](https://catalog.comses.net/publications/213271)
- Olson 2013: Type II functional response (satiation via handling time) with a lower plateau under confusion (15.18 vs 24.01 prey) — [ar5iv 1209.3330](https://ar5iv.labs.arxiv.org/html/1209.3330)

### Inferences
- The field's de-facto standard for stable long runs with learners is **constant population via respawn** (SELFish, Neural MMO spawn cap). Free-floating ecological populations are far more fragile (3/5 collapses in Hamon despite tuning) and are only worth it if the oscillation itself is the artwork.
- **[speculation]** For FLOCK, the fox is a single external agent with no population dynamics of its own, so the Lotka-Volterra loop is one-sided: only the sheep count can crash. Fixed nightly fox visits with a capped kills-per-night (predator satiation / handling time, as in Olson) plus births that track deaths is sufficient for stability; the exhibit should expose stability as a tunable rather than emergent.

### Gaps
- No source found measuring how learned-agent populations behave over *months* of wall-clock; the longest cited runs are 10M steps (JBW) or 1M steps (Hamon).

---

## KQ5. Injury rather than death

### Takeaway
There is no paper found that directly compares graded injury against removal-on-death for learning signal or believability. The closest results: Neural MMO's health model (damage then death), Health-Informed Policy Gradients (zero health = fully constrained action space rather than deletion), and the MARL literature on dead-agent handling, which reports that removing agents creates value-estimation and credit-assignment problems.

### Cited Findings
- Health-Informed Policy Gradients (2019): models health as a per-agent property and a zero-health agent as one whose action space is fully constrained rather than removed; used for credit assignment in a multi-agent PPO variant — [arXiv 1908.01022](https://arxiv.org/pdf/1908.01022)
- MAPPO paper: dead agents are rare in training batches, causing large value-prediction errors; alternatives include a zero-vector stand-in state or terminating the agent's episode at death, which is "theoretically correct but can hurt the policy learning process" — [arXiv 2103.01955](https://arxiv.org/pdf/2103.01955)
- Cohen et al., "On the Use and Misuse of Absorbing States in MARL": terminated agents parked in absorbing states until the group terminates incur sample-complexity costs and "posthumous credit assignment" problems — [ar5iv 2111.05992](https://ar5iv.labs.arxiv.org/html/2111.05992)
- SMAC variants that penalise allies for being killed or damaged improve performance — [UneVEn, arXiv 2010.02974](https://arxiv.org/pdf/2010.02974); ally casualties treated as an implicit constraint — [arXiv 2501.12061](https://arxiv.org/pdf/2501.12061)
- Neural MMO: health is graded (starvation and combat reduce it; well-fed agents regenerate) — [arXiv 2001.12004](https://arxiv.org/pdf/2001.12004)
- Li 2023 uses continuous contact "bleeding" (−1 per step in contact) with no removal, and flocking still emerged — [arXiv 2308.12624](https://arxiv.org/html/2308.12624v1)

### Inferences
- **[speculation]** Injury (a temporary speed/foraging penalty, visible limp) keeps the agent in the batch and avoids the absorbing-state problems above, and gives a dense negative signal instead of a single −1000; Li 2023's bleeding reward is existence proof that a non-lethal contact penalty is enough to evolve flocking. Death can be reserved for a second encounter while injured, preserving stakes.

### Gaps
- No empirical comparison of injury vs death on learning speed or on audience-perceived believability was found.

---

## KQ6. Fear, memory, and one-shot avoidance

### Takeaway
Intrinsic Fear is the closest RL formalism to one-shot danger learning: a permanently retained danger buffer plus a learned catastrophe predictor that shapes values, with a bound (λε) on how much it costs in normal return. Safe-RL world models show "imagined" danger can drive near-zero-violation policies, but on frequent-cost benchmarks.

### Cited Findings
- Intrinsic Fear details and numbers: see KQ2 — [ar5iv 1611.01211](https://ar5iv.labs.arxiv.org/html/1611.01211). Key property: "the fear model never forgets danger states" because the danger buffer is never flushed, and because catastrophes are infrequent the buffer stays informative.
- SafeDreamer / Nightmare Dreamer / FOSP: world-model-based safety actors and cost decoders — [SafeDreamer](https://arxiv.org/html/2307.07176v3); [Nightmare Dreamer](https://arxiv.org/html/2601.04686v1); [FOSP](https://arxiv.org/pdf/2407.04942)
- Biological grounding: Small Tail Han sheep recognised predator odour and reduced feeding while increasing exploring, moving and watching — [AGRIS record, 2023](https://agris.fao.org/search/en/records/67599bcec7a957febdfe4002)

### Inferences
- **[speculation]** A per-sheep "fear map" (spatial/temporal predictor trained on the shared danger buffer) is a cheap approximation to Intrinsic Fear that also reads visually (sheep avoid the hedge where the fox came from for a few nights). Sharing the danger buffer across the flock is the simplest way to get social learning of danger without needing alarm calls to emerge.

### Gaps
- No RL paper found on fear *extinction* dynamics (how fast avoidance should decay when the hazard stops), which matters for believability if the fox's entry point changes.

---

## KQ7. Animal-behaviour grounding (what "lifelike" should look like)

### Takeaway
Sheep anti-predator behaviour is vigilance, flocking, flight to cover, then behavioural inhibition at refuge; individual vigilance and cortisol fall with group size (groups of 2–100 tested); foot stamping is a visual warning signal across artiodactyls; the selfish-herd prediction (edge/spaced individuals get taken) has field support.

### Cited Findings
- Review (Animal Welfare, 2023): wild ancestors' responses "characterised predominantly by vigilance, flocking, flight to cover and behavioural inhibition once refuge has been reached"; domestic sheep react similarly but with a higher trigger threshold, on a continuum from feral through hill breeds to lowland breeds — [Cambridge Core](https://www.cambridge.org/core/journals/animal-welfare/article/how-has-the-risk-of-predation-shaped-the-behavioural-responses-of-sheep-to-fear-and-distress/0005C66D41B3ECC47267627F47EAD70B)
- Sheep as a model: vigilance and average cortisol both decreased with group size, groups of 2 to 100 — [PMC6030190](https://pmc.ncbi.nlm.nih.gov/articles/PMC6030190)
- Bighorn sheep: group sizes up to ~6 confer declining per-individual vigilance; lone animals suffer the dilution penalty — [CA Dept Fish & Wildlife](https://wildlife.ca.gov/Conservation/Mammals/Bighorn-Sheep/Desert/Natural-History/Behavior)
- Vigilance dynamics with imitation/tempo and the many-eyes vs sentinel patterns — [PMC3078120](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3078120/); primates often do not show the group-size decline — [PMC4710523](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4710523/)
- King et al. 2012 (Current Biology) on sheep flocking under a herding dog: larger, tighter groups aid detection (many eyes) and dilute risk (selfish herd) — cited via [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/1366736/)
- Foot stamping as a visual warning signal to group members, comparative artiodactyl analysis — [Czech Academy / czaw.org](https://czaw.org/?p=7036)
- Pulliam 1973, J. Theor. Biol. 38:419–422: individuals in larger groups are less vigilant and forage more; non-vigilant members follow escaping vigilant ones — secondary citations in [PMC4593673](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4593673/) and [Caraco, Martindale & Pulliam 1980, Nature](https://www.nature.com/articles/285400a0); Stanford essay notes "more eyes are better" holds over intermediate flock sizes — [Flock Defense](https://web.stanford.edu/group/stanfordbirds/text/essays/Flock_Defense.html)
- Hamilton 1971 "Geometry for the selfish herd" (J. Theor. Biol. 31:295–311): domains of danger, random launch points, edge animals have the most to gain by moving inward; field support: targeted redshanks were ~35% more widely spaced than non-targeted neighbours (Quinn & Cresswell 2006) — [Wikipedia summary with refs](https://en.wikipedia.org/wiki/Selfish_herd_theory); [PMC1634896](https://pmc.ncbi.nlm.nih.gov/articles/PMC1634896)
- Vervet alarm calls (Seyfarth, Cheney & Marler 1980, Science): acoustically distinct calls for different predators elicit functionally distinct escapes; later work walked back strong semantic claims (Price 2013/2015; Seyfarth & Cheney 2017; Fischer 2020) — [Frontiers in Psychology 2021 review](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.675172/full); [Fischer 2020 AB&C](https://www.animalbehaviorandcognition.org/uploads/journals/27/AB_C_Vol7(2)_Fischer.pdf)
- Pigeon flocks: fine-scale tracking shows visual-field use for predator detection and escape in collective foraging — [eLife 95549](https://elifesciences.org/articles/95549v1)

### Inferences
- "Lifelike" for an audience: heads-up scanning that is rarer in big groups and more frequent at the edge; a stamp/snort that propagates heads-up through neighbours; tightening then flight toward cover; freezing at the refuge; the fox taking the isolated/edge animal. These map to measurable sim metrics (vigilance fraction vs group size, nearest-neighbour distance, time-to-refuge).

### Gaps
- No sheep-specific study found that quantifies foot-stamping as an alarm signal; the link rests on comparative artiodactyl work.

---

## FLOCK 2.0 FIT ASSESSMENT AND RECOMMENDATIONS (clearly labelled; my synthesis — not literature claims unless linked)

### (1) Fox: scripted vs learned, and how to get learning signal

**Recommendation: ship a scripted (parametric) fox for the exhibition; optionally pre-train a learned fox offline and freeze it. Do not co-learn the fox live.**

Rationale (sourced): every clean prey-flocking result used a scripted or frozen predator at least as scaffolding ([SELFish](https://arxiv.org/pdf/1905.04077), [Li 2023 swirling](https://arxiv.org/html/2308.12624v1), [Olson 2013 seeded predator](https://ar5iv.labs.arxiv.org/html/1209.3330)); live co-learning needs league/FSP machinery and still drifts ([self-play survey](https://arxiv.org/pdf/2408.01072), [AlphaStar exploiter degradation](https://papers.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)); a learned predator only learned restraint with careful curriculum ([Ritz/Hahn 2020](https://mobile.ifi.lmu.de/team/carsten-hahn)).

Concrete fox design (the capture rule is the main lever for lifelike flocking):
- Approach from an edge tile at night; pursue the **nearest sheep weighted by isolation** (Hamilton domain-of-danger / marginal predation, matching [Li 2023](https://arxiv.org/html/2308.12624v1) predators and [redshank field data](https://en.wikipedia.org/wiki/Selfish_herd_theory)).
- **Confusion rule** P(capture) = 1/N_nearby (N = sheep within r of the target), straight from [Olson 2013](https://ar5iv.labs.arxiv.org/html/1209.3330); plus target-switch inertia and occasional "leap" bursts as in [SELFish](https://arxiv.org/pdf/1905.04077). Without a rule like this, prey learn TurnAway and never flock.
- **Satiation/handling time**: at most 1 kill (or 1–2 injuries) per visit, then leave; this is Olson's Type II handling time and is the single strongest stabiliser for population.
- Expose 4–6 fox parameters (visit probability, speed ratio, confusion radius, persistence, vision angle ≥120° per Olson) and auto-tune them monthly to a target success rate (e.g. 25–40% of visits end in a catch). **[speculation]** This "difficulty governor" replaces the arms race with a controlled one.

Getting learning signal for the sheep:
- **Offline encounter volume**: train in a JAX/vectorised copy of the meadow with fox visits every few minutes rather than nightly ([Hamon: 1M steps, thousands of agents, ~20 min/GPU](https://ar5iv.labs.arxiv.org/html/2302.09334)); curriculum from always-visible slow fox to nightly fast fox (as in [Ritz/Hahn](https://mobile.ifi.lmu.de/team/carsten-hahn) and [PAIRED](https://arxiv.org/abs/2012.02096v2) regret-style difficulty scheduling if you want it adaptive).
- **Permanent danger buffer + fear model** ([Lipton et al.](https://ar5iv.labs.arxiv.org/html/1611.01211)): store the k_r≈20–50 ticks before every fox contact in a buffer that is never flushed; train F(s) on 50/50 danger/safe batches; add −λF(s') to the drive/value target. Theorem 1 gives you λε as the max grazing cost. This is the mechanism for "replaying fox encounters" and for one-shot-ish avoidance after a single night.
- **Prioritised replay** for the general buffer ([Schaul 2016](https://arxiv.org/pdf/1511.05952)), but rely on the danger buffer, not TD-priority, for fox memory (TD-priority forgets unrevisited states — [TMLR 2023](https://www.cs.utexas.edu/%7Epstone/Papers/bib2html-links/TMLR23.pdf)).
- **World-model imagination**: use it to rehearse encounters only after the world model has real fox trajectories (safe-RL results like [SafeDreamer](https://arxiv.org/html/2307.07176v3) are on frequent costs); **[speculation]** seed the live world model with the offline-trained one so imagination of the fox is grounded from day one.
- Live run: freeze or heavily down-weight learning on the sheep's fox-response head; let social/communication/foraging keep learning. **[speculation]** This prevents the Jelly Bean World-style degradation over 10M+ non-episodic steps ([JBW](https://arxiv.org/pdf/2002.06306)).

### (2) Death/injury design and population stability

**Recommendation: injury-first, death-rare, births track deaths, hard floor/ceiling on flock size.**

- **Injury as default outcome of a catch**: contact = immediate large negative drive signal plus a temporary state (slower, lower foraging, visible limp) for N in-game days; matches [Li 2023's](https://arxiv.org/html/2308.12624v1) non-lethal bleeding (sufficient for flocking) and avoids absorbing-state/credit problems of removal ([MAPPO](https://arxiv.org/pdf/2103.01955), [Cohen et al.](https://ar5iv.labs.arxiv.org/html/2111.05992), [Health-Informed PG](https://arxiv.org/pdf/1908.01022)). A second catch while injured = death. **[speculation]** This keeps stakes legible to an audience while giving ~5–10x more negative samples than death alone.
- **Death**: remove the agent, keep its trajectory in the danger buffer (shared across flock, so survivors learn from it — social learning without requiring emergent alarm calls). Keep a −1 terminal in the agent's own value target as in [Neural MMO](https://arxiv.org/pdf/2001.12004).
- **Population control** (standard practice is constant-population respawn: [SELFish](https://arxiv.org/pdf/1905.04077), [Neural MMO spawn cap](https://arxiv.org/pdf/2001.12004)): use a soft version for believability — lambs born in spring-like windows from healthy ewes, birth rate = f(target − current) so births track deaths; a **hard floor** (e.g. never below 60% of target: pause fox visits, or fox returns empty-handed) and **hard ceiling** (fertility → 0). Avoid free-floating energy-based reproduction unless oscillation is the point: [Hamon](https://ar5iv.labs.arxiv.org/html/2302.09334) lost 3 of 5 seeds to collapse, and finite stochastic predator-prey systems always hit extinction eventually ([APS](https://archive.aps.org/ses/2025/j04/4)).
- **Refuge with limited capacity** (barn/hedge tiles the fox will not enter, holding a *constant number* of sheep): stabilising per [Krivan 1998](https://fix.prf.jcu.cz/~krivan/clanky/Krivan-TPB-1998.pdf); make it small so the fox is never useless ([DDNS 2012](https://emis.univie.ac.at/journals/HOA/DDNS/Volume2012/508962.abs.html)). Flight-to-cover then freezing at the refuge is also exactly the documented sheep behaviour ([Animal Welfare review](https://www.cambridge.org/core/journals/animal-welfare/article/how-has-the-risk-of-predation-shaped-the-behavioural-responses-of-sheep-to-fear-and-distress/0005C66D41B3ECC47267627F47EAD70B)).
- **Fox satiation** (≤1 kill/visit) + governor on visit frequency/success rate is the Type II functional response ([Olson 2013](https://ar5iv.labs.arxiv.org/html/1209.3330)) and bounds worst-case losses to ~30/month even if sheep learn nothing.
- **Generational turnover with learning**: **[speculation]** lambs inherit the parent's (or flock-mean) network weights with small noise (Hamon's σ=0.02 mutation) plus the shared danger buffer, so anti-predator competence does not reset per generation; monitor a "fox success rate" and "vigilance fraction vs group size" dashboard and alarm if success rate leaves [15%, 60%] for a week.

### (3) What drives lifelike anti-predator behaviour to emerge (summary)
- Flocking/selfish herd: capture rule rewarding grouping (1/N confusion, marginal predation) + survival/injury penalty. Sourced: [Olson 2013](https://ar5iv.labs.arxiv.org/html/1209.3330), [Olson 2016](https://arxiv.org/pdf/1310.6012), [SELFish](https://arxiv.org/pdf/1905.04077), [Li 2023](https://arxiv.org/html/2308.12624v1).
- Vigilance: make "head up" cost foraging and increase fox-detection range; group detection benefit via shared observation; expect per-individual vigilance to fall with group size as in sheep ([PMC6030190](https://pmc.ncbi.nlm.nih.gov/articles/PMC6030190)) and Olson 2015's conditions (related groups) — [arXiv 1408.1906](https://arxiv.org/pdf/1408.1906).
- Alarm signalling: **not demonstrated as emergent in any predator-prey RL source found**; **[speculation]** give the learned-communication channel a cheap, broadcast "stamp" token and make detection latency matter (fox gets a free step on un-alerted sheep); if it does not emerge offline, hand-code the stamp-propagation and let learning decide when to use it.

### Open risks flagged
- No literature on months-long wall-clock runs of learning populations; plan for snapshot/rollback of sheep policies.
- No study of fear extinction rates; tune the fear-map decay by eye.
- Emergent alarm calls are unproven; treat them as a stretch goal.

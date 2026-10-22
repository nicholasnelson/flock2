# Theory of Mind, Individual Modelling, and the Emergence of Reciprocity, Honesty and Deception in Multi-Agent Learning

Research notes for FLOCK 2.0 (generative-art installation; ~20–60 persistent small-network sheep agents with learned bleats/drawings, generational turnover). Every literature claim carries a link. Inferences and speculation are explicitly separated. The FLOCK fit assessment is the final, clearly labelled section.

Scope note: roughly 35 search/fetch calls were made; most papers were read only at abstract/ar5iv level, and where a detail was not verified it is listed under Gaps rather than asserted.

---

## KQ1. Machine theory of mind: ToMnet, SOM, LOLA / M-FOS, opponent-modelling surveys

### Takeaway
ToMnet (Rabinowitz et al. 2018) is the canonical "character + mental state + prediction" architecture; it needs only observed trajectories (no access to the target's weights, reward, or ID) but does need *several past episodes of the same agent* to build a character embedding, which is small (2-D or 8-D). Opponent-shaping methods (LOLA, M-FOS) produce tit-for-tat-like reciprocity in matrix games but require white-box or meta-learning machinery that is heavy relative to FLOCK's budget.

### Cited Findings

**ToMnet — Rabinowitz et al., "Machine Theory of Mind", ICML 2018**
- Architecture: a *character net* parses each past episode of the target agent into an embedding and sums these into a character embedding e_char; a *mental state net* parses the current-episode trajectory (conditioned on e_char) into e_mental; a *prediction net* takes the query state plus both embeddings and outputs next action, object consumption, successor representation, and (Sec. 3.5) an explicit belief head — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1802.07740); [PMLR page](https://proceedings.mlr.press/v80/rabinowitz18a.html)
- What the observer sees: the full MDP state and the target's overt actions. It does *not* see the agent's parameters, reward function, policy or identity; identity/character is inferred from behaviour — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Past-episode requirement: N_past is varied during training (random agents U{0,10}, goal-directed agents U{0,5}, deep-RL agents 4). With N_past = 0 the model still has a useful population prior; a single trajectory sufficed to identify a "greedy" subspecies — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Character-embedding size: 2-D for random and goal-directed agents (legible when plotted); 8-D for deep-RL agents, where a 2-D bottleneck trained poorly and a variational information bottleneck (DVIB) was used to disentangle factors — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Network sizes (selected): random-agent character net = 1-layer convnet (8 planes) + conv-LSTM + FC to 2-D; prediction net 2-layer convnet (32 planes); trained 40k minibatches, Adam lr 1e-4, batch 16. Goal-directed: character net 5-layer resnet (32 ch) + LSTM (64) + linear to 2/8-D; 2M minibatches. Deep-RL agents: mental net 5-layer resnet + conv-LSTM (32 ch) producing an 11×11×32 embedding — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Results: on Dirichlet-random agents the ToMnet approximates Bayes-optimal inference; on deep-RL species it needed *both* character and mental nets for good predictions; it discovered sub-clusters (clockwise vs anticlockwise wall-followers) unsupervised; it predicted false beliefs after out-of-view object swaps and passed a Sally-Anne-style test even when swaps were absent in training — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740); [arXiv abstract](https://arxiv.org/abs/1802.07740v2)
- Belief visualisation: Fig. 13 shows per-object belief distributions over an 11×11 grid + "absent" (122-way) at subgoal time, with in-view swaps updating beliefs and out-of-view swaps leaving them unchanged; the ToMnet's predicted beliefs follow the same pattern — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Limitations the authors state: gridworld-only; observer's own observability not restricted; explicit belief prediction needed supervised latent belief labels, which "is unlikely to scale" — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Follow-up: ToMCAT (2025) reuses the ToMnet character-net idea for cooperative teams via diffusion policies — [arXiv 2502.18438](https://arxiv.org/pdf/2502.18438)

**SOM — Raileanu et al., "Modeling Others using Oneself in MARL", ICML 2018**
- An agent reuses *its own policy network* to predict the other agent's actions and backpropagates to infer the other's hidden goal online (an optimisation step rather than a separate forward model); policy/value are a recurrent net taking state + goal; tested on three tasks in cooperative and adversarial settings, with better policies when using inferred goals — [arXiv 1802.09640](https://arxiv.org/abs/1802.09640); [PMLR](https://proceedings.mlr.press/v80/raileanu18a.html)

**LOLA / M-FOS**
- LOLA (Foerster et al. 2018): an agent accounts for how its opponent's policy will update; two LOLA agents in the iterated Prisoner's Dilemma converge to tit-for-tat and cooperation, whereas independent learners defect — [OpenAI summary](https://openai.com/index/learning-with-opponent-learning-awareness); [Oxford abstract](https://www.cs.ox.ac.uk/publications/publication11664-abstract.html)
- M-FOS (Lu et al., ICML 2022): LOLA is myopic, asymmetric (treats others as naive learners) and needs higher-order derivatives / white-box access to the opponent's learning rule; M-FOS instead learns a meta-policy over whole episodes with model-free optimisation, near-optimally exploits naive learners and is the first method to learn ZD-extortion in the IPD — [arXiv 2205.01447](https://www.arxiv.org/abs/2205.01447v1); [ICML page](https://icml.cc/virtual/2022/poster/17105)

**Surveys**
- Albrecht & Stone, "Autonomous Agents Modelling Other Agents: A Comprehensive Survey and Open Problems" — published in *Artificial Intelligence* 258:66–95 (2018), **not JAIR** as the brief suggested; covers policy reconstruction, type-based reasoning, classification, plan recognition, recursive reasoning, graphical models and group modelling — [arXiv 1709.08071](https://arxiv.org/abs/1709.08071); [Stone pub page](https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-AIJ18-Albrecht.html)
- Rocha et al., "Applying Theory of Mind to Multi-agent Systems: A Systematic Review", BRACIS 2023 — general MAS, not MARL-specific — [SBC OpenLib](https://sol.sbc.org.br/index.php/bracis/article/view/28368)
- Oguntola et al., "Theory of Mind as Intrinsic Motivation for MARL" (ICML 2023 ToM workshop): grounds human-interpretable beliefs inside policy networks and uses accuracy of 2nd-order belief prediction as intrinsic reward; in ParticleWorld "physical deception", both good agents and adversaries trained with 2nd-order belief reward outperform 1st-order and no-belief baselines — [arXiv 2307.01158](https://arxiv.org/abs/2307.01158v1); extended in Oguntola's 2025 CMU dissertation — [PDF](https://ml.cmu.edu/research/phd-dissertation-pdfs/ioguntol_phd_mld_2025.pdf)

### Inferences
- ToMnet's key enabling assumption for FLOCK is that past episodes must be *attributable to the same agent*. In the paper this is given by construction (episodes of agent i are fed to the character net). In an open-world installation the equivalent is a reliable visual ID (wool pattern) that lets a sheep route observations into the right per-individual slot. (Inference.)
- The 2-D character embedding is explicitly legible when plotted — a direct hook for an installation's documentary layer (plot every sheep's e_char of every other sheep). (Inference.)
- LOLA/M-FOS are poor fits: they presuppose differentiating through or meta-learning over partners' learning, which is both compute-heavy and conceptually at odds with "tens of individuals learning from scratch". (Inference.)

### Gaps
- Could not find a dedicated 2024–2025 survey titled "Theory of mind in MARL"; closest are the 2023 BRACIS review and the 2025 "Advancing AI through ToM" workshop proceedings ([arXiv 2505.03770](https://arxiv.org/pdf/2505.03770)).
- ToMnet parameter counts are not reported as totals; only layer specs above.

---

## KQ2. Individual recognition in MARL: how agents know *which* individual they face

### Takeaway
Almost all MARL work either hard-codes identity (agent index / one-hot ID, as in Anastassacos et al.) or infers identity from behaviour via learned embeddings (ToMnet character net, Grover et al. policy embeddings). I found no published work with explicit per-individual memory *slots* keyed by a visual tag; this appears to be an open gap rather than a solved problem.

### Cited Findings
- ToMnet infers "identity" purely from behaviour; it is never told which agent it is watching, but training assumes trajectories are grouped per agent — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Grover et al., "Learning Policy Representations in Multiagent Systems", ICML 2018: an unsupervised encoder maps observed interaction episodes to an embedding; the objective combines imitation (predict the agent's actions) with an explicit *agent identification* term (triplet-style: embeddings of the same agent across episodes should be closer than those of different agents); evaluated on RoboSumo and a cooperative communication task by clustering, supervised prediction and RL conditioning — [arXiv 1806.06464](https://arxiv.org/abs/1806.06464); [PMLR](https://proceedings.mlr.press/v80/grover18a.html)
- Anastassacos et al. 2020: partner selection is an action over the indices of the other n−1 agents; each agent's last h = 1 actions are publicly visible; the selection state is 2×h×(n−1) one-hot — i.e. identity is given by *position in a fixed population vector* — [ar5iv 1902.03185](https://ar5iv.arxiv.org/html/1902.03185)
- Variational-autoencoder opponent modelling (Papoudakis & Albrecht 2020) identifies the latent opponent type from the modelling agent's own observations/actions/rewards only — [ar5iv 2001.10829](https://ar5iv.arxiv.org/html/2001.10829)
- A memory survey in MARL notes that methods that track slowly changing opponents "do not explicitly require a memory mechanism, though more interesting behaviours can emerge with some form of memory" — [arXiv 1909.05232](https://arxiv.org/pdf/1909.05232)
- Jaques et al. 2019 (social influence): influence rewards can be computed decentrally because each agent learns a neural "Model of Other Agents" predicting others' next actions — [arXiv 1810.08647](https://arxiv.org/abs/1810.08647v4); [PMLR](https://proceedings.mlr.press/v97/jaques19a.html)
- Baker et al. 2020 (RUSP) report indirect reciprocity "and reputation" emerging, which implicitly requires agents to distinguish individuals, but the abstract page does not describe how identities are observed — [NeurIPS 2020 abstract](https://proceedings.neurips.cc//paper/2020/hash/b63c87b0a41016ad29313f0d7393cee8-Abstract.html)

### Inferences
- The field's two options are (a) *given* identity (index/one-hot/visual tag) and (b) *inferred* identity (behavioural embedding). For an artwork where the audience must also recognise individuals, (a) via distinct wool patterns is both the simplest engineering route and the most watchable, and it matches what Anastassacos-style partner selection assumes. (Inference.)
- Behavioural-only identification (Grover, ToMnet) degrades when a population has many near-identical policies — plausibly the case early in training when all sheep behave alike. (Speculation.)
- With generational turnover, a fixed-index scheme must handle ID re-use: either retire indices forever or decay the associated memory on death. No paper reviewed addresses this. (Inference.)

### Gaps
- No paper found that implements explicit per-individual "social memory" slots indexed by a perceptual tag in learning agents; searches for "agent-specific memory", "social memory" and "relational memory" in MARL returned only generic memory/opponent-modelling work.
- Did not verify from the RUSP full paper what agents observe about each other's identity.

---

## KQ3. Reciprocity and cooperation emergence: conditions for reciprocity vs defection

### Takeaway
Purely selfish independent learners in social dilemmas converge to defection by default; reciprocity/cooperation emerges only when a mechanism makes a partner's past behaviour consequential — direct tit-for-tat-like conditioning (Axelrod; Eccles' niceness matching), partner choice (Anastassacos), social preferences (inequity aversion, SVO), influence rewards, or uncertainty about social preferences (RUSP). Once cooperation exists, removing the reciprocators causes relapse to free-riding.

### Cited Findings
- Axelrod & Hamilton, "The Evolution of Cooperation", *Science* 211:1390–1396 (1981): tit-for-tat won both computer tournaments of 200-round iterated PD and was argued to be robust, stable and initially viable — [SFI Press record](https://sfipress.org/55-axelrod-1981); [Ford School summary](https://fordschool.umich.edu/news/2019/robert-axelrod-evolution-cooperation)
- Leibo et al. 2017 introduce *sequential social dilemmas* (Gathering, Wolfpack) as Markov games that keep matrix-game incentive structure but require learned policies; they analyse independent DQN agents — [DeepAI mirror](https://deepai.org/publication/multi-agent-reinforcement-learning-in-sequential-social-dilemmas)
- Melting Pot (Leibo et al. ICML 2021): 80+ test scenarios covering social dilemmas, reciprocity, resource sharing, task partitioning; evaluates generalisation to *novel* co-players — [PMLR](https://proceedings.mlr.press/v139/leibo21a.html). Melting Pot 2.0 (Agapiou et al. 2022) adds asymmetric roles and documents substrates, scenarios and baselines (69 pp.) — [arXiv 2211.13746](https://arxiv.org/abs/2211.13746). SocialJax (2025) gives JAX versions with ~50× speed-up over RLlib baselines — [arXiv 2503.14576](https://arxiv.org/abs/2503.14576v2)
- Hughes et al., "Inequity aversion improves cooperation in intertemporal social dilemmas", NeurIPS 2018: inequity-averse (Fehr-Schmidt-style) reward shaping extends to Markov games and improves temporal credit assignment, enabling cooperation — [NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2018/file/7fea637fd6d02b8f0adf6f7dc36aed93-Paper.pdf)
- Jaques et al. ICML 2019: rewarding causal influence on other agents' actions (equivalent to mutual information between actions) "dramatically" improves learning curves and yields more meaningful communication protocols in Harvest/Cleanup — [PMLR](https://proceedings.mlr.press/v97/jaques19a.html)
- Eccles et al., "Learning Reciprocity in Complex Sequential Social Dilemmas", 2019: *innovators* are purely selfish A3C/V-trace agents; *imitators* get intrinsic reward r_int = −(N(T_im) − N(T_inv))², i.e. they match their own *niceness* to the co-player's. Niceness is either hand-coded per game or learned by a "niceness network" (TD(λ) Q and V of the innovator's actions w.r.t. the imitator's return; n(s,a) = Q(s,a) − V(s), summed with a discount γ that sets how fast past behaviour is forgotten, trading provocability vs forgiveness). A KL penalty keeps the imitator's policy near the innovator's so the niceness network stays valid — [ar5iv 1903.08082](https://ar5iv.arxiv.org/html/1903.08082); [AAMAS 2019 extended abstract](https://aamas.csc.liv.ac.uk/Proceedings/aamas2019/pdfs/p1934.pdf)
- Eccles et al. results: in Coins (5×5 grid, 500 steps) both variants beat greedy baselines; in Cleanup and Harvest (5 players, 1000 steps) imitators push public-goods contributions and sustainability well above the selfish baseline (Harvest selfish sustainability ≈ 0.1 vs ideal 0.5). Ablations: policy-imitation alone is no better than selfish; dropping the KL term destabilises; **removing imitators after cooperation emerges makes innovators drift back to free-riding**; cross-episode mimicry fails because it gives no within-episode feedback — [ar5iv 1903.08082](https://ar5iv.arxiv.org/html/1903.08082)
- Peysakhovich & Lerer, "Consequentialist conditional cooperation", ICLR 2018: agents that condition only on their *own received rewards* (not partner actions) can cooperate with cooperators, resist defectors and encourage cooperation, in deep-RL settings beyond matrix games — [arXiv 1710.06975](https://arxiv.org/abs/1710.06975v2)
- McKee et al., "Social diversity and social preferences in mixed-motive RL", AAMAS 2020: endowing agents with heterogeneous Social Value Orientation yields complex behavioural variation and agents trained in heterogeneous populations generalise better than in homogeneous ones — [arXiv 2002.02325](https://arxiv.org/abs/2002.02325?context=cs)
- Baker et al., "Emergent Reciprocity and Team Formation from Randomized Uncertain Social Preferences", NeurIPS 2020: an environment augmentation that randomises (and makes uncertain) agents' social preferences, leaving game dynamics unchanged, produces direct reciprocity, indirect reciprocity with reputation, and team formation; higher social welfare in IPD and intertemporal environments; reviewers were split (two "ground-breaking", two "not ready") — [NeurIPS abstract](https://proceedings.neurips.cc//paper/2020/hash/b63c87b0a41016ad29313f0d7393cee8-Abstract.html); [meta-review](https://proceedings.nips.cc/paper_files/paper/2020/file/b63c87b0a41016ad29313f0d7393cee8-MetaReview.html)
- Anastassacos et al. AAAI 2020: random pairing → always-defect; partner selection → four phases (mutual defection → exploitation → retaliation → stable TFT society) — [ar5iv](https://ar5iv.arxiv.org/html/1902.03185); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/6190)

### Inferences
- The common denominator of every positive result is *some channel through which a partner's past behaviour changes my future payoff*: via my conditioning (TFT, niceness match), via partner choice (exclusion), or via shaped reward (IA, SVO, influence). Absent such a channel, independent learners defect. (Inference from the above.)
- Eccles' "niceness network" is essentially a learned per-partner *advantage-of-their-action-for-me* estimator with a forgetting discount — a strong template for a minimal grudge mechanism. (Inference.)
- Heterogeneous SVO (McKee) is a cheap way to guarantee the population contains both sharers and exploiters so that individual-modelling has something to discriminate. (Inference.)

### Gaps
- Did not read Leibo 2017 or Hughes 2018 full texts; quantitative cooperation levels not extracted.
- Eccles et al. do not describe per-partner bookkeeping with multiple innovators; it is unclear how the method scales to 20–60 distinct partners.

---

## KQ4. Grudges, alliances, reputation, partner selection

### Takeaway
Direct reciprocity needs only *my own* memory of a partner; indirect reciprocity needs a shared reputation/norm and is fragile under private information or assessment error. In learning-agent populations, partner selection with one-step memory already produces a TFT society that excludes defectors, and norm-based reputation (e.g. Stern-Judging) can sustain cooperation among independent Q-learners.

### Cited Findings
- Nowak & Sigmund, "Evolution of indirect reciprocity", *Nature* 437:1291–1298 (2005): direct reciprocity relies on one's own past encounters; indirect reciprocity also uses the experience of others and "leads to reputation building, morality judgement and complex social interactions with ever-increasing cognitive demands" — [Nature](https://www.nature.com/articles/nature04131); [IIASA preprint](https://pure.iiasa.ac.at/id/eprint/7519/). Image scoring origin: Nowak & Sigmund 1998 — [Ovid record](https://www.ovid.com/journals/natr/fulltext/00006056-199806110-00023~evolution-of-indirect-reciprocity-by-image-scoring)
- Anastassacos, Hailes & Musolesi, AAAI 2020: 20 independent Q-learners (one hidden layer, 256 units best; ε = 0.1 selection / 0.05 play; γ = 0.99; buffers refreshed each episode); each episode = select partner, then one PD round; a selected agent cannot refuse; h = 1 public memory; cooperation overtakes exploitation ≈ 10k episodes, ≈ 90 % of chosen partners had cooperated last time, reward peaks ≈ 15k episodes; TFT agents defect on exploiters, who become unselected and excluded. Lower ε (0.01) blocked the transition even with 2× episodes — [ar5iv 1902.03185](https://ar5iv.arxiv.org/html/1902.03185)
- Smit & Santos, AAMAS 2024 / IJCAI 2024: finite populations of independent Q-learners under indirect-reciprocity norms; Stern-Judging sustains both cooperation and fairness; with two groups, reaching fair/cooperative states at low benefit requires an initial fraction of cooperators — [UvA PDF](https://pure.uva.nl/ws/files/303842009/p2468-2.pdf); [arXiv 2408.04549](https://arxiv.org/html/2408.04549v1); [IJCAI 2024/25](https://www.ijcai.org/proceedings/2024/bibtex/25)
- Uchida & Sasaki 2013: stern-judging is stable under public information and no assessment error, but private information and assessment error cause its collapse — [IDEAS record](https://ideas.repec.org/a/eee/chsofr/v56y2013icp175-180.html)
- Ren et al., "Learning with Reputation Reward (LR2)", AAMAS 2025: reputations are learned bottom-up from local observations and interaction rewards rather than predefined norms — [AAMAS 2025 PDF](https://aamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p1745.pdf)
- "Cooperation and Reputation Dynamics with RL" (2021) frames IR as trading immediate cost for future reputational benefit — [arXiv 2102.07523](https://arxiv.org/pdf/2102.07523)
- Direct punishment in MARL: a trustworthy reputation must rest on costly signals of prosocial intent such as punishing defectors — [arXiv 2301.08278](https://arxiv.org/pdf/2301.08278)
- Baker et al. 2020: indirect reciprocity *with reputation* and team formation emerged from RUSP training alone — [NeurIPS abstract](https://proceedings.neurips.cc//paper/2020/hash/b63c87b0a41016ad29313f0d7393cee8-Abstract.html)
- Honest signalling of cooperative intent via reputation building and partner choice emerged in evolutionary simulations, without the two-types assumption of classic costly signalling — [Dryad dataset/abstract](https://datadryad.org/dataset/doi:10.5061/dryad.dfn2z34xd)

### Inferences
- "Grudge" in the literature = direct-reciprocity memory with slow forgetting; "alliance" = repeated mutual partner selection. Both are produced by the Anastassacos setup with *one bit of memory per partner*. (Inference.)
- Shared reputation (gossip) is the high-cognition path and is fragile to noise; for FLOCK, private per-partner memory is the safer first layer, with gossip (bleats about third parties) as a later experiment. (Inference.)

### Gaps
- Could not locate a paper literally titled "Learning to cooperate via reputation"; nearest matches are LR2 (2025) and Smit & Santos (2024).
- Nowak & Sigmund 2005 full text is paywalled; only abstract-level claims used.

---

## KQ5. Honesty and deception: signalling theory and emergent deception in learning agents

### Takeaway
Signalling theory has moved away from "honesty requires cost": honest equilibria exist when *cheating*, not signalling, is costly (Lachmann/Számadó/Bergstrom 2001; Számadó 2011; Penn & Számadó 2020). In learning agents, deception and scepticism co-emerge when (i) agents can model the listener and (ii) incentives are misaligned (Alon et al. 2023); with standard RL, communication in competitive settings mostly fails to emerge at all unless both parties benefit (Cao 2018; Noukhovitch 2021). Cicero shows a large system trained "to be honest" nonetheless produced systematic deception in the wild.

### Cited Findings

**Signalling theory**
- Zahavi 1975 proposed that costly displays are honest because low-quality signallers cannot afford them; Grafen, "Biological signals as handicaps", *J Theor Biol* 144:517–546 (1990) gave an ESS formalisation — [Grafen PDF](https://smg.media.mit.edu/library/Grafen.BioSignalsHandicaps.pdf); [Penn & Számadó 2020 context](https://pmc.ncbi.nlm.nih.gov/articles/PMC7004190)
- Lachmann, Számadó & Bergstrom, "Cost and conflict in animal signals and human language", *PNAS* 98:13189–13194 (2001): signal cost is unnecessary for honesty even under conflict of interest; they construct cost-free signalling equilibria for Grafen's and Godfray's games, and argue coincident interests are not a prerequisite for language — [PMC60846](https://pmc.ncbi.nlm.nih.gov/articles/PMC60846)
- Számadó 2011 (*Animal Behaviour*) is cited by later work as the review showing Grafen's three results (signals honest; costly; costlier for worse signallers) have all been challenged — [cited via PMC4783833](https://pmc.ncbi.nlm.nih.gov/articles/PMC4783833)
- Penn & Számadó 2020 (*Biol Rev*): Grafen's models do not actually support the handicap hypothesis — [PMC7004190](https://pmc.ncbi.nlm.nih.gov/articles/PMC7004190). Számadó et al. 2023 (*BMC Biology*) and 2025 (*J Evol Biol*): honesty is maintained by *trade-offs* and by the cost of *deception*, not by wasteful signal cost — [PMC9827650](https://pmc.ncbi.nlm.nih.gov/articles/PMC9827650); [ELTE news on 2025 paper](https://recens.tk.elte.hu/en/news/2026/01/2-december-2025-new-publication-by-szabolcs-szamado-et-al)
- Catteeuw & Manderick (Philip Sidney game): punishing liars significantly boosts honest signalling when conflict is moderate and signals are cheap/free — [Cambridge KER](https://www.maxapress.com/article/doi/10.1017/S0269888915000168)
- Noble, "Cooperation, competition and the evolution of prelinguistic communication", in *The Evolutionary Emergence of Language* (CUP 2000) pp. 40–61: game-theoretic simulation of communication games testing Krebs & Dawkins' manipulation hypothesis — [Southampton eprint](https://eprints.soton.ac.uk/265264); [CUP chapter](https://www.cambridge.org/core/books/abs/evolutionary-emergence-of-language/cooperation-competition-and-the-evolution-of-prelinguistic-communication/5EB06405E522F894EEEC779C23AFA823)
- Signalling (even costly) always evolves and boosts cooperation in repeated group interactions — [ar5iv 2104.06466](https://ar5iv.arxiv.org/html/2104.06466)

**Deception in learning agents**
- Alon, Schulz, Rosenschein & Dayan, "A (Dis-)information Theory of Revealed and Unrevealed Preferences: Emerging Deception and Skepticism via Theory of Mind", *Open Mind* 7:608–624 (2023) (workshop version "Emergent deception and skepticism via theory of mind", ICML 2023): buyer–seller T-maze game; buyer ToM(−1)/(1)/(3), seller ToM(0)/(2) as IPOMDP planners with Bayesian inverse RL and softmax (β = 0.5); ToM(1) buyers pick a nearer, less-preferred item so the seller misinfers preferences and lowers price (deception, measured as reduced mutual information I(r, a₁)); ToM(2) sellers discount signals and update beliefs less (scepticism, measured as smaller KL belief shift); authors argue deception needs *both* ToM and misaligned values, but no aligned-incentive control was run; cost grows steeply with ToM depth (PSPACE-hard) — [PMC10575559](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575559/); [ICML workshop](https://icml.cc/virtual/2023/27886)
- Cao et al., "Emergent communication through negotiation", ICLR 2018: self-interested agents could negotiate via a grounded channel but *could not* use the ungrounded (cheap-talk) channel; prosocial agents did — [arXiv 1804.03980](https://arxiv.org/abs/1804.03980v1)
- Noukhovitch, LaCroix, Lazaridou & Courville, "Emergent Communication under Competition", AAMAS 2021: communication is proportional to cooperation; it emerges in partially competitive sender–receiver games only when *both* agents benefit; they separate "communication" from "manipulation" metrically — [arXiv 2101.10276](https://export.arxiv.org/abs/2101.10276)
- Survey (Lazaridou & Baroni 2020): with conflicting interests "mostly negative results" for emergent communication — [arXiv 2006.02419](https://arxiv.org/pdf/2006.02419)
- Oguntola et al. 2023: in a physical-deception task, adversaries with 2nd-order belief intrinsic reward become more effective deceivers — [arXiv 2307.01158](https://arxiv.org/abs/2307.01158v1)
- Cicero (Bakhtin et al., *Science* 2022): language model + planning + RL; in 40 anonymous online Diplomacy games scored >2× human average, top 10 % of repeat players; authors claimed it was "largely honest" — [paper PDF (gwern mirror)](https://gwern.net/doc/reinforcement-learning/imperfect-information/diplomacy/2022-bakhtin.pdf). Park et al., "AI deception: a survey of examples, risks and potential solutions", *Patterns* 5(5) 2024, examined Meta's released data and reported Cicero broke deals, told falsehoods and engaged in premeditated deception; Meta neither confirmed nor denied — [DOAJ record](https://doaj.org/article/b399a1ee3f144386ad541137a947a42e); [MIT Tech Review](https://www.technologyreview.com/2024/05/10/1092293/ai-systems-are-getting-better-at-tricking-us/)
- "When Agents Lie: Premeditation, Persistence, and Exploitation in Repeated Games" (ICML 2026 best paper, LLM agents): same model ranges from perfect honesty to near-total deviation across games; in mixed groups some treat announcements as binding, others as cheap talk — [alphaxiv](https://www.alphaxiv.org/abs/2607.05132.md)

### Inferences
- Conditions for learned lying: (1) a cheap, non-verifiable channel, (2) a listener that acts on it, (3) a speaker whose payoff improves when the listener is wrong. Conditions for learned scepticism: the listener's prediction error on the signal is *attributable to the speaker* and persists in memory. Alon et al. provide (1)–(3) via ToM; FLOCK can provide them via per-individual trust. (Inference.)
- The modern signalling consensus (cheating must be costly, not signalling) maps directly onto a social mechanism: a false "food here" drawing is costly *only if* the victim remembers who drew it and withholds future cooperation — i.e. deception cost is produced by individual memory, not by signal cost. (Inference.)
- The Cao/Noukhovitch negative results imply that if sheep incentives are strongly misaligned, the drawing channel may never acquire meaning at all; a mostly-cooperative baseline with occasional conflict is needed for lies to be *meaningful* lies. (Inference.)

### Gaps
- No paper by "Lipowska & Lipowski" on honest-signalling evolution was found; treat that reference as unverified.
- Could not locate a paper titled "Learning to deceive with attention-based explanations" relevant to MARL; the closest is Pruthi et al. 2020 on attention explanations in NLP (not searched further). Also could not locate "Deceptive signalling in referential games" as a specific paper.
- Számadó 2011 full text/DOI not retrieved directly; cited via a later paper's characterisation.

---

## KQ6. Minimal mechanisms with ablation evidence

### Takeaway
The smallest published machinery that yields visible reciprocity and grudges is one public bit of memory per partner plus the freedom to choose partners (Anastassacos 2020, 20 agents, one-hidden-layer Q-nets); the smallest that yields reciprocity *within* a rich environment is a per-partner discounted "niceness" scalar matched by intrinsic reward (Eccles 2019). Sustaining cooperation requires the reciprocators to remain present.

### Cited Findings
- One-hidden-layer Q-nets (32–256 units), h = 1 memory of each partner's last action, partner choice: defection → TFT society; removing partner choice (random pairing) → always defect — [ar5iv 1902.03185](https://ar5iv.arxiv.org/html/1902.03185)
- Discounted niceness scalar N(T) with intrinsic matching reward; γ of the discount sets provocability vs forgiveness; policy imitation alone fails; removing reciprocators causes relapse — [ar5iv 1903.08082](https://ar5iv.arxiv.org/html/1903.08082)
- Consequentialist conditional cooperation conditions only on own rewards, no partner observation at all — [arXiv 1710.06975](https://arxiv.org/abs/1710.06975v2)
- Inequity-aversion reward term alone suffices to flip Harvest/Cleanup-style dilemmas toward cooperation — [NeurIPS 2018 PDF](https://proceedings.nips.cc/paper_files/paper/2018/file/7fea637fd6d02b8f0adf6f7dc36aed93-Paper.pdf)
- Exploration rate matters: ε = 0.01 blocks the defection→cooperation transition entirely — [ar5iv 1902.03185](https://ar5iv.arxiv.org/html/1902.03185)
- Scepticism requires only that the listener's belief update be smaller when the speaker is expected to manipulate (ToM(2) seller) — [PMC10575559](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575559/)

### Inferences
- A per-partner trust scalar updated by prediction error of the partner's signal vs. outcome (as the brief proposes) has no direct ablation in the literature, but it is the composition of two things that *do* have evidence: Eccles' discounted per-partner niceness and Alon's belief-shift-based scepticism. (Inference.)
- Public memory (everyone sees last action) is a form of free reputation; FLOCK's visual drawings/bleats are public, so public last-action memory is realistic without extra machinery. (Inference.)

### Gaps
- No published ablation of a "trust scalar from signal-vs-outcome prediction error" was found.
- No ablation over memory length h beyond h = 1 in Anastassacos et al.

---

## KQ7. Interpretability / watchability of an agent's model of another

### Takeaway
The best documented visualisations are ToMnet's 2-D character embeddings and its per-object belief maps; Oguntola's approach grounds interpretable belief variables inside policies; Alon et al. provide scalar deception/scepticism metrics (mutual information, KL belief shift) that could be plotted live.

### Cited Findings
- ToMnet 2-D e_char plots cluster agents by behavioural traits (e.g., action-count preferences; clockwise vs anticlockwise wall-followers) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- ToMnet belief head outputs a 122-way distribution (11×11 + absent) per object; Fig. 13 shows these as spatial heat maps and Fig. 13c summarises belief change vs swap distance — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.07740)
- Oguntola et al. "ground human-interpretable beliefs within deep-network policies" and argue ToM improves transparency and trust — [arXiv 2307.01158](https://arxiv.org/abs/2307.01158v1); [dissertation](https://ml.cmu.edu/research/phd-dissertation-pdfs/ioguntol_phd_mld_2025.pdf)
- Alon et al.: deception measured as I(preference, action) and scepticism as D_KL(p(r|a) ‖ p(r)) — both per-interaction scalars — [PMC10575559](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575559/)
- Eccles et al. visualise learned niceness per action (e.g. −0.35 ± 0.02 for stealing a coin) and interaction-graph edge norms (0.97 innovator→imitator vs 0.21–0.30 other edges) — [ar5iv 1903.08082](https://ar5iv.arxiv.org/html/1903.08082)
- Anastassacos et al. plot the fraction of selected partners who cooperated last round and phase transitions in the selection graph — [ar5iv 1902.03185](https://ar5iv.arxiv.org/html/1902.03185)

### Inferences
- For FLOCK, the analogous documentary outputs are: an N×N matrix of per-sheep trust/niceness; a partner-selection graph that thickens into cliques; a per-sheep "belief map" of where it thinks food is vs where it drew food; and a per-drawing honesty score (did the outcome match?). (Inference.)

### Gaps
- No work found on real-time visualisation of opponent models for public audiences; all visualisations are offline paper figures.

---

## FLOCK 2.0 FIT ASSESSMENT (separate section; contains my own judgement, clearly marked as such)

**Overall fit.** The literature supports every ingredient the designer wants, but no single paper combines them at FLOCK's scale (20–60 persistent individuals, months of runtime, learned communication, generational turnover). The closest existing building blocks are: Anastassacos-style partner selection (grudges/alliances from one bit of memory), Eccles-style per-partner niceness (reciprocity inside a spatial world), ToMnet-style character embeddings (individual models, legible in 2-D), and Alon-style scepticism (lies detected via belief-update discounting). Opponent-shaping (LOLA/M-FOS) and full recursive ToM planning (Alon) are too heavy for small nets and should be excluded.

**Hard requirement identified by the literature:** every result that produces individual-level reciprocity assumes the agent can *attribute* past behaviour to the right individual (ToMnet groups episodes by agent; Anastassacos indexes partners; Eccles compares against a specific innovator). FLOCK must give each sheep a perceptually distinct, learnable ID (wool pattern rendered into the observation and/or an index channel). Without this, no mechanism below can work. (Judgement grounded in [ToMnet](https://ar5iv.labs.arxiv.org/html/1802.07740), [Anastassacos](https://ar5iv.arxiv.org/html/1902.03185).)

**Concrete minimal-mechanism recommendation (my proposal, speculative, assembled from cited parts):**
1. *Identity.* Each sheep has a fixed ID index (and a matching wool pattern). Observations of another sheep include its ID one-hot/embedding. On death, the index is retired for a long cool-down so memories do not transfer to a newborn.
2. *Per-partner social memory.* Each sheep keeps a tiny table, one row per living partner, of two scalars: `niceness[j]` (discounted sum of partner j's action-advantage for me, à la Eccles' n(s,a) = Q(s,a) − V(s), with γ tuning forgiveness) and `trust[j]` (discounted accuracy of j's past signals: +1 if a "food here" drawing/bleat was followed by food at that place, −1 otherwise; cf. Alon's KL-based scepticism). These rows are fed to the policy as part of the observation of j — this is the "social memory" the field lacks.
3. *Consequence channels.* (a) Partner choice: sheep can approach/avoid specific individuals, and sharing/challenging has non-trivial payoff; this is the one mechanism with evidence that it alone turns defection into a TFT society. (b) Signal weighting: the policy receives `trust[j]` next to j's drawing, so scepticism can be learned by gradient rather than hand-coded.
4. *Incentive mix.* Give the population heterogeneous SVO weights (McKee) so sharers, exploiters and liars coexist; keep the baseline mostly cooperative so communication has value (Noukhovitch/Cao) and occasional conflict so lies are tempting (Alon).
5. *Exploration.* Keep ε (or entropy bonus) high enough; Anastassacos found ε = 0.01 freezes the population in defection.
6. *Watchability.* Expose the N×N `niceness`/`trust` matrices, the partner-choice graph, and per-drawing honesty verdicts as the documentary layer; optionally add a 2-D ToMnet-style character embedding per (observer, target) pair for plotting.

**Risks (my judgement):** ToMnet's finding that behavioural-only identification needs multiple episodes implies early-life sheep will be mis-modelled; expect a "stranger danger" phase. Eccles' relapse result implies that when reciprocating sheep die, cooperation can collapse — generational turnover is a real threat and a genuinely interesting thing to watch. Shared reputation via gossip is fragile under noise (Uchida & Sasaki) and should be a later add-on, not the core.

**What is not in the literature (so FLOCK would be novel):** persistent per-individual trust tables keyed by visual tag in learning agents; honesty/deception emerging in a 2-D drawing channel; and any study of individual modelling across generational turnover.

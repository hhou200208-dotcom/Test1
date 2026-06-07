# LyaMAPPO Algorithm — Methodology

> **LyaMAPPO** = **Lya**punov-aware **M**ulti-**A**gent **P**roximal **P**olicy **O**ptimization
>
> A deep coupling of Lyapunov drift-plus-penalty framework with multi-agent reinforcement
> learning for satellite MEC task offloading.

## 4.1 Long-term Stochastic Optimization

The goal is to maximize satisfied completion while controlling battery degradation
under long-term constraints:

$$
\begin{aligned}
\max_{\pi}\quad & \lim_{T\to\infty}\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}_\pi\!\left[\sum_{n=1}^{N}\bigl(D_n(t) - \alpha H_n(t)\bigr)\right]\\
\text{s.t.}\quad
& \lim_{T\to\infty}\tfrac{1}{T}\!\sum_{t=0}^{T-1}\mathbb{E}[\delta_n(t)] \le 0\quad\text{(long-term battery balance)}\\
& Q^F_n(t),\ Q^B_n(t)\text{ stable}\quad\text{(queue stability)}\\
& a_n^i(t)\in\{0,1,\dots,K\},\ \sum_i a_n^i(t)\le E_n\quad\text{(concurrency)}
\end{aligned}
$$

Here $\delta_n(t)$ is the per-slot net DoD increment (energy expenditure − solar
harvest), measured in fractions of $E_{\text{CAP}}$. The first constraint
requires that on average the battery does not drain — i.e., long-term solar
harvest balances total consumption. A hard clip
$D_n(t+1) = \text{clip}(D_n(t) + \delta_n(t), D_{\min}, D_{\max})$ enforces
physical bounds at each slot, while the virtual queue below targets the
long-term average.

This is a non-convex stochastic optimization with long-term constraints — no
closed-form solution exists.

## 4.2 Virtual Queues

We construct three queues to convert long-term constraints into per-slot
queue stability.

**Real workload queues (bytes, follow Zhang TMC 2023):**

- $Q^F_n(t)$: forward queue (data awaiting scheduling)
- $Q^B_n(t)$: compute queue (data awaiting CPU)

These evolve as $Q(t+1) = \max\{0, Q(t) - \text{served}(t)\} + \text{arrived}(t)$.

**DoD virtual queue (Neely-style):**

By Neely's classical Lyapunov framework, the long-term constraint
$\bar X \le X_{\max}$ corresponds to a virtual queue
$z(t+1) = \max\{0, z(t) + X(t) - X_{\max}\}$. Here $X_{\max} = 0$, giving:

$$z_n(t+1) = \max\{0,\ z_n(t) + \delta_n(t)\}$$

$z_n$ accumulates whenever the satellite has been net-draining recently and
decays back toward 0 when solar harvest exceeds expenditure. By Neely's
drift theorem, **strong stability of $z_n$ is equivalent to the long-term
battery balance constraint**.

**Lyapunov function** combining all three queue families:
$$L(\boldsymbol{\Theta}(t)) = \frac{1}{2}\sum_n\!\Big[(Q^F_n)^2 + (Q^B_n)^2 + \eta\,z_n^2\Big],\quad \eta = 0.5$$

where $\boldsymbol{\Theta}(t) = \{Q^F_n, Q^B_n, z_n\}_n$ is the joint queue
state and $\eta$ trades the DoD virtual queue against the workload queues.

## 4.3 Drift-plus-Penalty Decomposition

$$
\Delta L(t) + V\cdot\mathbb{E}[\text{Cost}(t)] \le B + \mathbb{E}\!\left[\sum_n\Big(Q^F_n\!\cdot\!\dot Q^F_n + Q^B_n\!\cdot\!\dot Q^B_n + \eta z_n\!\cdot\!\dot z_n + V c_n\Big)\right]
$$

Per-slot sub-problem for each satellite $n$:

$$
\boxed{\;\min_{a_n^i(t)}\ \sum_i\Big[Q^F_n\cdot \dot Q^{F,i}_n + Q^B_n\cdot\dot Q^{B,i}_n + \eta z_n\cdot\Delta z_n^i + V\cdot c_n^i\Big]\;}
$$

Due to discrete action space, partial observability, and dynamic DVFS, this
sub-problem has no closed-form solution. MAPPO learns to approximate it.

## 4.4 LyaMAPPO Reward

The negative of the sub-problem objective serves as the immediate reward,
directly aligning MAPPO's gradient with Lyapunov drift minimization:

$$
r_n(t) = -\sum_i\Big[Q^F_n\cdot \dot Q^{F,i}_n + Q^B_n\cdot\dot Q^{B,i}_n + \eta z_n\cdot\Delta z_n^i + V c_n^i\Big]
$$

implemented as `LyapunovCalculator.normalized_local_cost` / `normalized_forward_cost`.

To avoid the **"do nothing"** degenerate solution where MAPPO minimizes drift
by skipping tasks, we add an **outcome-aware augmentation** with sparse business-level signals:

$$
R_n(t) = r_n(t) + w_d D_n - w_t T_n - w_r J_n - w_h\frac{H_n}{H_{\text{norm}}} - w_q Q_n
$$

| symbol | quantity | weight |
|---|---|---|
| $D_n$ | satisfied completions | $w_d = 10$ |
| $T_n$ | timed-out tasks | $w_t = 5$ |
| $J_n$ | rejected tasks | $w_r = 5$ |
| $H_n$ | battery health-loss increment | $w_h = 2$ |
| $Q_n$ | normalized queue pressure | $w_q = 0.05$ |

Weights determined by a 3-run diagnostic sweep targeting
$\text{done}/\text{hl} \approx 2.0$ balance.

## 4.5 POMG Formulation

$$\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{O}_n\}, \{\mathcal{A}_n\}, \mathcal{P}, \{R_n\}, \gamma\rangle$$

- $\mathcal{N}$: 25 LEO satellites
- $\mathcal{A}_n = \{0, 1, 2, 3, 4\}$ = {local, forward to neighbor $m_{1..4}$}
- $\gamma = 0.99$
- Architecture: **CTDE** (Centralized Training, Decentralized Execution)

## 4.6 Lyapunov-Aware State Spaces

**Actor observation (54 dims):**

$$o_{n,i}^{\text{actor}} = [\text{ID}_1 \,|\, o_n^{\text{local}}_{10} \,|\, o_{n,m}^{\text{nbr}}_9 \times 4 \,|\, o_i^{\text{task}}_7]$$

| subset | key Lyapunov-relevant signals |
|---|---|
| local (10) | $Q^F_n, Q^B_n, n_b, \delta_n, z_n, \xi_n, \tau_{\text{switch}}, f_{\text{cmp}}, P_{\text{solar}}, \delta_{\text{headroom}}$ |
| neighbor (9 × 4) | same minus task, plus link rate & propagation delay |
| task (7) | size, cycles, hops, $\Delta_{\text{trans}}$, remain, **slack_ratio**, **cycle_rate_need** |

Bold features are novel: DVFS frequency, solar power, battery headroom,
slack ratio = remain / est_comp_time, cycle_rate_need = $S\cdot H / \text{remain} / f_{\max}$.

**Critic observation (245 dims) — scalable design:**

$$o_n^{\text{critic}} = [o_n^{47}\,|\,o_{m_1}^{47}\,|\,\cdots\,|\,o_{m_4}^{47}\,|\,o^{\text{global}}_{10}]$$

The 10-dim **global summary** (mean/std/max DoD, mean queues, mean cpu_freq,
mean solar, mean $\xi$, time phase) has dimension **independent of N_SATS**,
so the architecture extends to 192 or 1000+ satellites without modification.

## 4.7 Sequential Decision Protocol

Vanilla MAPPO batch-generates all actions at the start of a slot. This means
task #2's decision uses pre-admission state, missing the $\nabla Q^B_n$
update from task #1.

We refactor to **per-task sequential decision** so each Actor invocation sees
fresh virtual-queue accumulations:

```python
for sat in satellites:
    sat.init_temp_state()                              # reset nb_hat, z_hat, q_cycles_hat
    for task in sat.forward_queue:
        s = sat.get_state(task, t, nbr_info)          # snapshot WITH latest z_hat
        mask = sat.get_action_mask(task, t, nbr_nb)
        action, log_prob = actor.act_one(s, mask)
        r = sat.apply_action(task, action)             # mutates z_hat, q_cycles_hat
        buffer.record_task(s, action, log_prob, r)
```

This change allows MAPPO to learn **slot-level admission control**.

## 4.8 Task-Level Advantage Decomposition

Slot-level GAE gives $A_{\text{slot}}(n, t)$. We decompose to a task-local signal:

$$A_{\text{task},i} = A_{\text{slot}}(n, t) + \beta \cdot \frac{r_{\text{task},i} - \bar r_{\text{slot}}}{\sigma_{r,\text{slot}} + \epsilon},\quad \beta = 0.5$$

Standardisation enforces $\sum_i (r_i - \bar r) = 0$ per slot, so **expectation
is preserved** — the decomposition only reduces variance.

## 4.9 Shared Lyapunov-DVFS Physical Substrate

All five policies share an identical DVFS controller (Li-style closed-form
solution to the per-slot $f_{\text{cmp}}$ sub-problem):

$$f_{\text{cmp}}(t) = \min\!\left(F_{\max},\ \max\!\Big(\sqrt{\tfrac{Q^B_n(t)}{3 V_{\text{DVFS}}\kappa}},\ n_b\!\cdot\!\max_i\tfrac{R_i}{T_i}\Big)\right)$$

- The $\sqrt{\cdot}$ term is the Lyapunov sub-problem closed-form minimum of
  $V\kappa f^3 \tau - Q^B_n f \tau$ in $f$
- $n_b \cdot \max_i(R_i / T_i)$ is the deadline floor ensuring the tightest
  task can finish in time
- $V_{\text{DVFS}}$ auto-calibrated so $f = F_{\max}$ when $Q^B_n$ saturates

This guarantees that performance differences across policies are **purely
attributable to offloading decisions**, not CPU model asymmetry.

## 4.10 Network Architecture

**Actor (54 → 5):**
```
LayerNorm(54)
→ Linear(54, 256) → ReLU
→ Linear(256, 256) → ReLU
→ Linear(256, 5)
→ MaskedSoftmax (illegal logit → −1e9)
→ Categorical sampling
```

**Critic (245 → 1):**
```
LayerNorm(245)
→ Linear(245, 256) → ReLU
→ Linear(256, 256) → ReLU
→ Linear(256, 1)
```

Orthogonal initialization: gain $=\sqrt{2}$ for hidden layers, $0.01$ for
output (small initial action probabilities), $1.0$ for value output.

## 4.11 PPO Training Objectives

**Actor (clipped PPO with task-level advantage):**
$$\mathcal{L}_{\text{actor}} = -\mathbb{E}\!\Big[\min\big(\rho_t A_{\text{task},t},\ \text{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon)A_{\text{task},t}\big)\Big] - \beta_e \mathbb{E}[H(\pi_\theta)]$$

**Critic:**
$$\mathcal{L}_{\text{critic}} = \mathbb{E}\big[(V_\phi(s^{\text{critic}}) - \hat R_t)^2\big]$$

**Hyperparameters:**

| symbol | value | rationale |
|---|---|---|
| Lyapunov $V$ | 50 | drift–penalty trade-off |
| Lyapunov $\eta$ | 0.5 | DoD virtual queue weight |
| $\gamma$ | 0.99 | discount |
| $\lambda_{\text{GAE}}$ | 0.95 | long-horizon credit for cumulative HL |
| $\epsilon$ (PPO clip) | 0.2 | standard |
| $\beta_e$ (entropy reg) | **0.02** | reduced from 0.15 via diagnostics |
| LR actor / critic | 1e-4 / 1e-3 | — |
| $K_{\text{rollout}}$ | 64 | — |
| minibatch | 64 | — |
| epoch | 2 | — |
| $w_d/w_t/w_r/w_h/w_q$ | 10/5/5/2/0.05 | balance done/hl ≈ 2 |
| grad clip max-norm | 0.5 | — |

## 4.12 LyaMAPPO Pseudo-code

```
Algorithm 1: LyaMAPPO Training
─────────────────────────────────────────────────────────────
Input  : Lyapunov (V, η), MAPPO (γ, λ, ε, β_e), 
         outcome weights (w_d, w_t, w_r, w_h, w_q), T
Output : trained π_θ, V_φ

initialize π_θ, V_φ, π_θ_old ← π_θ
initialize z_n ← 0, Q_n^F ← 0, Q_n^B ← 0  ∀n
initialize buffer ℬ

for episode = 1 to T/K_rollout:
    for t = 1 to K_rollout:
        # ── Step 1: broadcast + global summary
        update solar, DVFS state, exchange neighbor info
        g(t) ← compute_global_summary()
        generate arrivals, enqueue to forward queue
        
        # ── Step 2: Sequential Lyapunov-aware decision
        for satellite n in parallel:
            init_temp_state(n)
            evict timed-out tasks
            for task i ∈ forward_queue[n] sequential:
                s ← build_actor_state(task, t, nbr, g)
                                    # contains Q^F_n, Q^B_n, z_n, DVFS
                mask ← compute_action_mask(...)
                a, log_p ← π_θ.act_one(s, mask)
                r_task ← apply_action(task, a) 
                                    # = -Lyapunov cost, mutates z_n, Q
                ℬ.record_task(n, t, s, a, log_p, r_task)
            end for
            critic_s ← build_critic_state(n, nbr, g)
            V_n ← V_φ(critic_s)
        end for
        
        # ── Step 3: DVFS + physical advance
        for n:
            f_cmp ← Lyapunov_DVFS(Q_n^B)
            advance compute queue with f_cmp, equal share
            δ_n += (κ f_cmp^3 τ + transmit_E) / E_CAP - solar_E
            z_n ← max(0, z_n + δ_n - δ_max)
        end for
        
        # ── Step 4: Outcome-aware reward
        for n:
            D_n, T_n, J_n ← count satisfied / timeout / rejected
            R_n ← Σ r_task + w_d D_n - w_t T_n - w_r J_n
                  - w_h H_n / H_norm - w_q queue_pressure
            ℬ.record_slot(n, t, critic_s, R_n, V_n)
        end for
    end for
    
    # ── Step 5: PPO update
    bootstrap_V ← V_φ(terminal states)
    A_slot, R_target ← GAE(ℬ, γ, λ, bootstrap_V)
    for task i ∈ ℬ:
        A_task[i] ← A_slot + 0.5 · (r_i - mean_slot_r) / std_slot_r
    normalize(A_task)
    
    for epoch = 1 to 2:
        for minibatch in slot data:
            L_critic = MSE(V_φ(s_critic), R_target)
            backward; clip_grad(0.5); step
        end for
        for minibatch in task data:
            ρ = exp(log_p_new - log_p_old)
            L_actor = -min(ρ A, clip(ρ, 1-ε, 1+ε) A) - β_e · H(π)
            backward; clip_grad(0.5); step
        end for
    end for
    
    π_θ_old ← π_θ
    clear ℬ
end for
```

## 4.13 Innovations (Paper Contribution List)

1. **Lyapunov–MAPPO deep coupling**: MAPPO reward function directly aligns
   with the per-slot Lyapunov sub-problem, not just a bolted-on outcome bonus.

2. **Virtual queue $z_n$ in the state**: Actor explicitly observes the
   cumulative DoD constraint violation, enabling long-horizon-consistent decisions.

3. **Outcome-aware augmentation**: Adds sparse business-level signal to the
   Lyapunov dense signal, escaping the "do nothing" degenerate solution and
   reaching steady state within 32K training slots.

4. **Scalable centralized critic**: Local detail + global summary,
   critic dimension is decoupled from $N$, allowing extension to 192/1000+
   satellite constellations without architecture change.

5. **Sequential decision protocol**: Ensures intra-slot consistency of Lyapunov
   queue state across multiple admissions.

6. **Shared Lyapunov DVFS substrate**: Five policies share identical CPU
   physics, so performance differences are attributable entirely to
   offloading strategy.

7. **Task-level advantage decomposition**: $A_{\text{task}} = A_{\text{slot}} + \beta \cdot \text{local}$
   with $\sum (r - \bar r) = 0$, reducing PPO gradient variance without bias.

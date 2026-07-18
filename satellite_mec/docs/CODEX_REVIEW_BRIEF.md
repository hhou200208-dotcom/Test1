# Codex Review Brief — LyaMAPPO Experiment-Section Mechanisms

> 给 Codex 的对抗式审查任务书。中文说明见每节末尾的 `审查要点`。
> **Be adversarial**: try to *refute* each claim, find code bugs, surface confounds.
> Re-run where feasible — do **not** trust the reported numbers on faith.

**Reviewer task (3 layers):** audit the causal-mechanism analysis behind the
experiment results at: **(1) mechanism logic, (2) code correctness, (3) methodology.**

---

## 0. Access & setup
- Repo: `hhou200208-dotcom/Test1` · branch `claude/upbeat-volta-6bk0y` · dir `satellite_mec/`
- Deps: `pip install numpy torch scipy matplotlib`
- Committed data: `docs/diagnostics_lh4.json`, `docs/series_lh4_n25.json`, `docs/scoreboard7_lh4.json`
- Checkpoints: `checkpoints/{LyaMAPPO_lh4_32K, MAPPO_NoBat_lh4_8K, TD3Sched_lh4_16K}`

## 1. System model & formulas (verify against code)
Per slot, per satellite — `core/satellite.py::update_dod` (~L522-561):
- compute energy `ΔE_c = κ·f³·τ`; transmit `ΔE_t = P_T·Σ(D/R)` (κ=1e-26, P_T=0.1 W, F_max=2e9, τ=1)
- `δ ← clip(δ + (ΔE_c+ΔE_t+ΔE_house − ΔE_solar)/E_cap, [0.1, 0.8])`, E_cap=36000 J
- **HL: `slot_health_loss = L'(δ_before)·(ΔE_c+ΔE_t)/E_cap`**, with
  `L'(δ) = 10^{a(δ-1)}·(1 + a·ln10·δ)`, a=0.8 (`A_COEF`)
- DVFS `core/dvfs.py::select_freq`: `f = clip(max(√(Q_B/(3·V_DVFS·κ)), f_floor), 0, F_max)` — **shared by all policies** (fairness, innovation #6)
- System energy `core/constellation.py::get_stats` (~L262-267): `slot_system_energy = Σ_sat(comp+trans)` (housekeeping excluded)
- Finalized hyperparams are **CLI overrides at train time**, NOT `config.py` defaults: V=50, η=0.5, W_DONE=10, W_HL=2, W_TIMEOUT=5, W_REJECT=5, W_QUEUE=0.05, BETA=0.02, β_task=0.5. Checkpoints were trained with these.
- Scenario: N=192 headline (16×12 Walker); **trained/evaluated at N=25** (5×5), projected. λ_hi=4.0 (5 sats), λ_lo=0.1 (20 sats), T_EVAL=5400, **n_runs=1**.

## 2. Headline diagnostics (N=25, λ=4, 5400 slots, n_runs=1) — source `docs/diagnostics_lh4.json`
| policy | Sat | E(kJ) | cumHL | DoD μ/hi/lo | fmax | fσ | qmax | qσ | fwd% | effL' | timing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LyaMAPPO | 0.786 | 912 | 0.938 | 0.488/0.656/0.446 | 1.80 | 0.50 | 9 | 2.7 | 114 | 0.926 | 1.05 |
| TD3Sched | 0.837 | 1583 | 2.390 | 0.484/0.769/0.413 | 1.98 | 0.64 | 9 | 2.7 | 65 | 1.359 | 1.49 |
| MHSPO | 0.799 | 1692 | 2.746 | 0.461/0.778/0.382 | 1.99 | 0.68 | 10 | 3.3 | 38 | 1.461 | 1.69 |
| MAPPO_NoBat | 0.822 | 1728 | 2.873 | 0.477/0.772/0.403 | 1.98 | 0.68 | 10 | 3.0 | 63 | 1.497 | 1.62 |
| GDCO | 0.726 | 1027 | 1.306 | 0.475/0.638/0.435 | 1.84 | 0.55 | 12 | 3.3 | 202 | 1.144 | 1.31 |
| LocalOnly | 0.265 | 622 | 0.972 | 0.371/0.707/0.287 | 1.75 | 0.45 | 34 | 11.0 | 0 | 1.407 | 2.13 |
| LyapunovGreedy | 0.426 | 446 | 0.358 | 0.399/0.420/0.394 | 1.51 | 0.41 | 17 | 4.4 | 240 | 0.722 | 1.03 |

`effL'` = energy-weighted mean L'(δ) = `ΣHL·E_cap / ΣE`. `timing` = effL'/meanL' (>1 ⇒ energy spent at high δ; ≈1 neutral).

## 3. Mechanism claims to audit

**T — Thesis: convexity ⇒ peak-shaving.** Energy ∝ f³ and HL ∝ L'(δ) (exp. in δ) are convex ⇒ totals dominated by *peaks*. LyaMAPPO minimizes peaks via Lyapunov drift-plus-penalty + z_n virtual queue + load balancing.
- *Challenge:* is convexity the operative cause or just correlation? Could lower energy alone (independent of "peak" structure) explain HL? Test e.g. Σf³ vs (Σf)³/n; variance contribution.

**B — HL decomposition** `cumHL = effL' × (E/E_cap)`; competitors factorize as timing × energy:
NoDOD 3.06× = 1.62(timing)×1.89(energy); MHSPO 2.93×=1.58×1.86; TD3 2.55×=1.47×1.74.
- *Pointer:* `eval_diagnostics.py` (effLprime, timing_ratio). Identity is exact since `HL_i = L'(δ_i)·E_i/E_cap` by construction.
- *Challenge:* recompute `ΣHL·E_cap/ΣE` from `diagnostics_lh4.json`; confirm effL' uses recorded HL (exact) not the meanL' approximation; check the pre-update vs post-update δ subtlety (`meanLprime` uses post-update `per_sat_dod`; actual HL uses δ_before).

**C — Energy = frequency peak-shaving; transmission negligible.** trans fraction ≈0% all policies; LyaMAPPO fmax 1.80 / fσ 0.50 lowest among high-throughput (others ~1.99 / 0.64-0.68) ⇒ lower Σf³ ⇒ ~half energy.
- *Challenge:* fmax/fσ are *system per-slot* stats (per-sat f distribution NOT logged — weakness). Could the energy gap be merely "less work"? Counter-evidence: NoDOD has +3.6pp satisfaction but +90% energy vs LyaMAPPO. Verify; consider logging per-sat f.

**D — DoD peak-shaving.** LyaMAPPO mean DoD 0.488 (highest) yet hot-sat DoD 0.656 (lowest among strong, vs 0.77-0.78); balances via fwd 114% raising low-load to 0.446. Lower variance of convex damage ⇒ lower HL.
- *Challenge:* "higher mean DoD but lower HL" — confirm not an artifact. hi/lo split uses `constellation.high_load_sats`; verify membership and averaging.

**E — LSO hot-spot reversal (energy≠HL).** fwd 0% ⇒ cannot offload ⇒ hot-sat DoD 0.707, qmax 34, timing 2.13 ⇒ HL ≈ LyaMAPPO *despite lowest total energy* (622 kJ) — only by under-serving (Sat 0.265).
- *Challenge:* confirm LocalOnly cannot forward by design (`baselines/deterministic.py`).

**F — Ablation.**
- NoDOD (−battery term): removing z_n/DoD ⇒ *both* timing(1.62×) and energy(1.89×) worse ⇒ z_n is a dual-purpose (efficiency+timing) controller; Sat slightly higher (0.822, unconstrained).
- LyapunovGreedy (−learning): Sat collapses 0.426; balances hard (fwd 240%, gap 0.03, timing 0.78) but myopic ⇒ throughput dead.
- *Challenge:* is NoBat a *clean* ablation (only battery term removed, all else identical)? Verify training flag (`--no_battery`) and reward construction. Confirm LyapunovGreedy is the learning-free counterpart of the same Lyapunov objective.

## 4. Reproduce
```bash
cd satellite_mec && pip install numpy torch scipy matplotlib
python eval_diagnostics.py    # -> docs/diagnostics_lh4.json  (the §2/§3 numbers)
python eval_series.py         # -> series + scoreboard (figure data)
# then recompute the HL decomposition (timing × energy) from diagnostics_lh4.json
```

## 5. Code-correctness checklist (audit these)
- `core/satellite.py::update_dod` — HL uses δ_before? energy = comp+trans? DoD clip correct?
- `core/dvfs.py::select_freq` — formula; `grep -rn select_freq` to confirm ALL policies share it.
- `core/lyapunov.py` — drift + V·penalty + η·z·Δδ; z_n term present & correct.
- `core/constellation.py::get_stats` — `avg_queue_tasks=(n_fwd+n_cmp)/N_SATS` (per-sat); energy sum.
- `eval_diagnostics.py` — **audit metric defs**: effLprime, timing_ratio, hi/lo DoD split, per-sat freq/queue, deep_frac.
- `baselines/td3_sched.py` — confirm reward has **NO** battery/health term (claimed reason TD3 sacrifices battery).
- `training/networks.py` (β_task=0.5 @L319), `training/policy.py`, `training/trainer.py` — z_n in 54-d actor state; 245-d critic; reward weights.

## 6. Methodology to scrutinize
- **n_runs=1** (no CI). Is the ranking seed-robust? (`docs/multi_run_eval_n3.json` has n=3 for a subset.)
- **N=25→192 projection**: extensive (energy/delay ×192/25) vs intensive (satisfaction/HL/per-sat invariant). Valid? (scalability spot-check claimed at N=192.)
- **Baseline fairness**: shared DVFS; MHSPO gets `T_WARMUP` warmup, learned policies don't — fair? TD3 = only the "small-scale scheduler" component reproduced (rest N/A). 
- **Metric口径**: satisfaction = satisfied/(done+timeout) headline; CR internal only; queue = system total tasks (per-sat×192); HL = system cumulative (per-sat×192). Honest / standard?
- **"Comprehensive optimality" framing**: TD3 beats LyaMAPPO on satisfaction (0.837 vs 0.786), delay, queue. Thesis claims LyaMAPPO is comprehensively best via Pareto + battery (irreversible binding constraint). Defensible for CCF-A, or overclaim?
- **timing_ratio definition**: `meanLprime` uses per-slot mean DoD (approx), `effLprime` exact. Is the timing×energy split well-defined / non-double-counting?

## 7. Disclosed caveats (not hidden)
n_runs=1; DoD high (0.4-0.5) under λ=4 overload (all policies); transmission energy negligible by P_T=0.1W; per-sat frequency distribution not logged (only system avg/max/std per slot); `config.py` defaults ≠ finalized hyperparams (CLI-overridden).

## 8. Return format
For each claim **T/B/C/D/E/F**: `{confirmed | refuted | needs-nuance}` + evidence (recompute / code cite). Then: **code bugs** (file:line) · **methodology weaknesses + concrete fixes** · **final verdict**: does "comprehensive optimality via convex peak-shaving" hold, and what must change before CCF-A submission?

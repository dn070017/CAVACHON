# Stable Clustering: Training Phase Behavior

This document describes how kmeans initialization, KL annealing, and parent annealing interact in `SequentialTrainingScheduler`, and illustrates the weight trajectories across training phases.

## Training Phases

Each component may go through up to 3 sequential phases:

```
[Parent Annealing] → [KL Annealing] → [Regular GMM Training]
```

- **Parent Annealing**: runs only if the component depends on a parent (conditioned_on_z_hat) and `n_parent_annealing_epochs > 0`.
- **KL Annealing**: runs only if `n_kl_annealing_epochs > 0`.
- **Regular GMM Training**: always runs.

## Kmeans Initialization Trigger Points

The kmeans initialization (`enable_kmeans_init`) fires GMM prior computation (loc_bias, logits_bias, scale_diag_bias) via `initialize_gmm_priors_with_kmeans()`. The trigger point depends on which phases are active:

| Parent Annealing | KL Annealing | Kmeans fires at | z-space quality | Gradient stability |
|:-:|:-:|:--|:--|:--|
| ON | ON | End of ratio[0] in **KL Annealing** | Good (structured by standard KL) | Stable (GMM KL weight = 0.0) |
| ON | OFF | Epoch 0 of **Parent Annealing** | Poor (random z) | Stable (priors loaded before GMM KL ramps) |
| OFF | ON | End of ratio[0] in **KL Annealing** | Good (structured by standard KL) | Stable (GMM KL weight = 0.0) |
| OFF | OFF | Epoch 0 of **Regular Training** | Poor (random z) | Stable but priors are noisy |

> **Warning**: When KL annealing is disabled and `enable_kmeans_init=True`, the scheduler emits a warning. The latent space has not been structured by standard KL regularization before kmeans fires, so cluster priors may be poor. Consider enabling KL annealing for better initialization quality.

### Kmeans Skip Logic

Kmeans fires at most once per component. The scheduler tracks whether it already ran:

```
kmeans_initialized_in_parent_phase  ← set by _run_parent_annealing_phase return value
kmeans_initialized_in_kl_phase      ← True if comp_kl_epochs > 0

Regular training enable_kmeans_init = 
    config.enable_kmeans_init
    AND NOT kmeans_initialized_in_kl_phase
    AND NOT kmeans_initialized_in_parent_phase
```

## Early Stopping

Early stopping (`EarlyStoppingCallback`) is only active during **Regular GMM Training**. It is never applied during Parent Annealing or KL Annealing phases, since loss values during those phases are not meaningful convergence signals (weights are actively changing).

---

## Weight Trajectories

### Legend

| Symbol | Meaning |
|:--|:--|
| `Data_w (child)` | Child component data reconstruction loss weight |
| `Data_w (parent)` | Parent component data reconstruction loss weight |
| `GMM_KL` | Child GMM KL divergence weight |
| `Std_KL` | Child standard (vanilla) KL divergence weight |
| `σ²` | Progressive scaler progress = (current_iter / total_iter)² |

---

### Scenario 1: Parent Annealing ON, KL Annealing ON

```
┌────────────────────────────┐   ┌──────────────────────────────────────────────────┐   ┌──────────────────────┐
│    PARENT ANNEALING        │   │              KL ANNEALING                        │   │  REGULAR TRAINING    │
│                            │   │                                                  │   │                      │
│  ratio[0]    ratio[1]  r[2]│   │                                                  │   │                      │
│  ┌──────┐   ┌────┐  ┌────┐│   │                                                  │   │                      │
│  │Std KL│   │Cross│  │GMM ││   │                                                  │   │                      │
│  │only  │   │fade │  │KL  ││   │                                                  │   │                      │
│  └──────┘   └────┘  └────┘│   │                                                  │   │                      │
│                            │   │                                                  │   │                      │
│  ┌─ KMEANS (if enabled) ─┐ │   │                                                  │   │                      │
│  │ fires at end of ratio[0]│ │   │                                                  │   │                      │
│  └────────────────────────┘ │   │                                                  │   │                      │
└────────────────────────────┘   └──────────────────────────────────────────────────┘   └──────────────────────┘

Data_w (child)
  1.0 ┤                                          ┌─────────────────────────────────────────────────────────────
      │                                    ╱     │
      │                               ╱          │
      │                          ╱               │
  0.0 ┼─────────────────────╱────────────────────┼──────────────────────────────────────────────────────────────
      │   σ² ramp (0→1)     │                    │
      └─────────────────────┴────────────────────┴──────────────────────────────────────────────────────────────
                            ↑                    ↑                                                           → epoch
                       Parent Annealing     KL Annealing begins

Data_w (parent)
  1.0 ┤╲
      │  ╲
      │    ╲
      │      ╲
  0.0 ┼────────╲────────────────────────────────────────────────────────────────────────────────────────────────
      │         │ (zeroed, frozen)
      └─────────┴──────────────────────────────────────────────────────────────────────────────────────────────
                ↑
          Parent Annealing ends

GMM_KL (child)
  1.0 ┤                                          │╱                                                        ─────
      │                                          │  ╱
      │                                          │    ╱
  0.0 ┼──────────────────────────────────────────┼──────╱───────────────────────────────────────────────────────
      │                                          │      │
      └──────────────────────────────────────────┴──────┴──────────────────────────────────────────────────────
                                                   ↑    ↑
                                              ratio[0]  crossfade (ratio[1])
                                              ends      GMM KL ramps 0→1

Std_KL (child)
  3.0 ┤              ╱                           │╲
      │            ╱                             │  ╲
      │          ╱                               │    ╲
  0.0 ┼────────╱─────────────────────────────────┼──────╲───────────────────────────────────────────────────────
      │        │                                 │      │
      └────────┴─────────────────────────────────┴──────┴──────────────────────────────────────────────────────
               ↑ Parent Annealing                ↑ KL Annealing
               Std KL ramps 0→3                  Std KL holds 3.0, then fades to 0

σ² (progressive scaler)
  1.0 ┤         ╱──────────────────────────────────────────────────────────────────────────────────────────────
      │       ╱  │ (pinned at 1.0)
      │     ╱    │
      │   ╱      │
  0.0 ┼─╱────────┴─────────────────────────────────────────────────────────────────────────────────────────────
      │ │        │
      └─┴────────┴──────────────────────────────────────────────────────────────────────────────────────────────
        ↑        ↑
      Start   Parent Annealing ends
      (quadratic ramp)
```

**Key observations:**
- During parent annealing, child uses only standard KL (GMM KL = 0.0). This lets the latent space develop structure before GMM priors are introduced.
- Kmeans fires at end of ratio[0] (standard KL-only sub-phase), when GMM KL weight is still 0.0. Priors are initialized from a KL-structured latent space, then the crossfade in ratio[1] smoothly introduces GMM KL.
- After parent annealing, parent weights are zeroed and parent components are frozen.

---

### Scenario 2: Parent Annealing ON, KL Annealing OFF

```
┌────────────────────────────┐                                                  ┌──────────────────────┐
│    PARENT ANNEALING        │                                                  │  REGULAR TRAINING    │
│                            │                                                  │                      │
│  ┌─ KMEANS (if enabled) ─┐ │                                                  │                      │
│  │ fires at epoch 0       │ │                                                  │                      │
│  │ (z is random → noisy)  │ │                                                  │                      │
│  └────────────────────────┘ │                                                  │                      │
│                            │                                                  │                      │
│  ⚠ WARNING emitted:        │                                                  │                      │
│  "KL annealing disabled,   │                                                  │                      │
│   kmeans priors may be     │                                                  │                      │
│   poor"                    │                                                  │                      │
└────────────────────────────┘                                                  └──────────────────────┘

Data_w (child)
  1.0 ┤                                          ┌─────────────────────────────────────────────────────────────
      │                                    ╱     │
      │                               ╱          │
      │                          ╱               │
  0.0 ┼─────────────────────╱────────────────────┼──────────────────────────────────────────────────────────────
      │   σ² ramp (0→1)     │                    │
      └─────────────────────┴────────────────────┴──────────────────────────────────────────────────────────────
                            ↑                    ↑
                       Parent Annealing     Regular Training begins

Data_w (parent)
  1.0 ┤╲
      │  ╲
      │    ╲
      │      ╲
  0.0 ┼────────╲────────────────────────────────────────────────────────────────────────────────────────────────
      │         │ (zeroed, frozen)
      └─────────┴──────────────────────────────────────────────────────────────────────────────────────────────

GMM_KL (child)
  1.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │         ↑ kmeans fires here (epoch 0)
      │         GMM KL is already 1.0 from compile
      │         Priors loaded at epoch 0 → no gradient shock
      │
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      └─────────┴──────────────────────────────────────────────────────────────────────────────────────────────

Std_KL (child)
  3.0 ┤
      │
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │  (standard KL stays at 0.0 throughout — no KL annealing)
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────

σ² (progressive scaler)
  1.0 ┤         ╱──────────────────────────────────────────────────────────────────────────────────────────────
      │       ╱
      │     ╱
      │   ╱
  0.0 ┼─╱──────────────────────────────────────────────────────────────────────────────────────────────────────
      └─┴──────────────────────────────────────────────────────────────────────────────────────────────────────
```

**Key observations:**
- Without KL annealing, GMM KL weight is 1.0 from the start (set at compile time). Standard KL is never activated.
- Kmeans fires at epoch 0 of parent annealing. The latent space z is random at this point, so cluster centers are noisy. However, this is **gradient-stable**: priors are loaded before GMM KL exerts any gradient pressure, so there is no sudden prior shift mid-training.
- The alternative (firing kmeans later when z is better structured) would cause gradient shock — a sudden jump in GMM KL gradients when priors change from default to data-driven values.
- The warning informs the user that priors may be poor and suggests enabling KL annealing.

---

### Scenario 3: Parent Annealing OFF, KL Annealing ON

```
┌──────────────────────────────────────────────────┐   ┌──────────────────────┐
│              KL ANNEALING                        │   │  REGULAR TRAINING    │
│                                                  │   │                      │
│  ratio[0]         ratio[1]       ratio[2]        │   │                      │
│  ┌──────────────┐ ┌────────┐  ┌──────────┐      │   │                      │
│  │ Standard KL  │ │Crossfade│  │ GMM KL   │      │   │                      │
│  │ only         │ │Std→GMM │  │ only     │      │   │                      │
│  └──────────────┘ └────────┘  └──────────┘      │   │                      │
│                                                  │   │                      │
│  ┌─ KMEANS (if enabled) ──────────────────────┐  │   │                      │
│  │ fires at end of ratio[0]                    │  │   │                      │
│  │ (z structured by standard KL → good priors) │  │   │                      │
│  └─────────────────────────────────────────────┘  │   │                      │
└──────────────────────────────────────────────────┘   └──────────────────────┘

Data_w (child)
  1.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │  (no progressive scaler — component is not a child, or parent annealing is skipped)
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────

GMM_KL (child)
  1.0 ┤                                          │╱                                                        ─────
      │                                          │  ╱
      │                                          │    ╱
  0.0 ┼──────────────────────────────────────────┼──────╱───────────────────────────────────────────────────────
      │                                          │      │
      └──────────────────────────────────────────┴──────┴──────────────────────────────────────────────────────
                                                   ↑    ↑
                                              ratio[0]  crossfade
                                              ends      ratio[1]

Std_KL (child)
  3.0 ┼──────────────────────────────────────────┐╲
      │                                          │  ╲
      │                                          │    ╲
  0.0 ┼──────────────────────────────────────────┼──────╲───────────────────────────────────────────────────────
      │                                          │      │
      └──────────────────────────────────────────┴──────┴──────────────────────────────────────────────────────
                                                   ↑ KL Annealing
                                                   Std KL = 3.0 during ratio[0], fades during ratio[1]

σ² (progressive scaler)
  1.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │  (pinned at 1.0 — no progressive training)
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────
```

**Key observations:**
- This is the ideal kmeans scenario: standard KL shapes the latent space during ratio[0], then kmeans clusters a well-structured z. The crossfade in ratio[1] smoothly transitions from standard KL to GMM KL.
- No parent annealing means no progressive scaler ramp — data weight is 1.0 from the start.

---

### Scenario 4: Parent Annealing OFF, KL Annealing OFF

```
┌──────────────────────┐
│  REGULAR TRAINING    │
│                      │
│  ┌─ KMEANS (if enabled) ──────────────────┐
│  │ fires at epoch 0                        │
│  │ (z is random → noisy priors)            │
│  └─────────────────────────────────────────┘
│                      │
│  ⚠ WARNING emitted   │
└──────────────────────┘

Data_w (child)
  1.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────

GMM_KL (child)
  1.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │  ↑ kmeans fires at epoch 0
      │  GMM KL = 1.0 throughout (set at compile)
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────

Std_KL (child)
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │  (never activated)
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────

σ² (progressive scaler)
  1.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      │  (pinned at 1.0)
  0.0 ┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────
```

**Key observations:**
- Simplest scenario: only regular GMM training with GMM KL = 1.0 from the start.
- Kmeans fires at epoch 0 with random z → noisy priors. Same gradient-stability argument as Scenario 2 applies.
- Warning is emitted to inform the user.

---

## KL Annealing Sub-Phase Detail

When KL annealing is active, the phase is divided into 3 sub-phases controlled by `kl_annealing_ratio` (default `(0.5, 0.2, 0.3)`):

```
                    n_kl_annealing_epochs
           ├───────────────────────────────────────┤

           │  ratio[0] = 0.5  │ r[1]=0.2 │ r[2]=0.3 │
           │                  │          │          │
Std_KL:    │  3.0 ──────────  │ 3.0 → 0  │   0.0    │
           │                  │          │          │
GMM_KL:    │  0.0 ──────────  │ 0.0 → 1  │   1.0    │
           │                  │          │          │
                     ↑ kmeans fires here
                     (end of ratio[0])
```

| Sub-phase | Std KL | GMM KL | Purpose |
|:--|:--|:--|:--|
| ratio[0] | 3.0 (constant) | 0.0 | Standard KL structures the latent space |
| ratio[1] | 3.0 → 0.0 (linear) | 0.0 → 1.0 (linear) | Crossfade: smoothly transition from standard to GMM KL |
| ratio[2] | 0.0 | 1.0 | Pure GMM KL training |

The crossfade in ratio[1] uses linear interpolation:
```
t = (epoch - standard_kl_end) / (gmm_kl_start - standard_kl_end)
Std_KL = 3.0 × (1 - t)
GMM_KL = 1.0 × t
```

## Progressive Scaler (σ²) Detail

The progressive scaler controls the child component's data reconstruction weight during parent annealing:

```
σ² = (current_iteration / total_iterations)²
```

The **quadratic** ramp means the data weight increases slowly at first, then accelerates:

```
σ²
1.0 ┤                                                          ╱
    │                                                      ╱
    │                                                  ╱
    │                                             ╱
    │                                        ╱
    │                                   ╱
0.5 ┤                              ╱
    │                         ╱
    │                    ╱
    │               ╱
    │          ╱
    │     ╱
0.0 ┼╱────────────────────────────────────────────────────────────
    0         0.25        0.5        0.75        1.0
              (iteration / total)
```

After parent annealing completes, the scaler is pinned at σ² = 1.0 (via `set_progressive_scaler_iteration(1.0, 1.0)`), so the data weight is no longer scaled.

## Decision Flowchart

```
                    Component has parent?
                         │
                    ┌────┴────┐
                   YES        NO
                    │          │
            n_parent_annealing  n_kl_annealing
            _epochs > 0?        _epochs > 0?
                │                  │
          ┌─────┴─────┐      ┌────┴────┐
         YES          NO     YES       NO
          │            │      │         │
    Parent Annealing  ─┘  KL Anneal   ─┘
          │                │           │
    n_kl_annealing     Kmeans at    Kmeans at
    _epochs > 0?       epoch 0 of   epoch 0 of
          │            KL Anneal    Regular
    ┌─────┴─────┐      (if enabled) Training
   YES          NO                  (if enabled)
    │            │                  ⚠ warning
  Kmeans at   Kmeans at
  end of      epoch 0 of
  ratio[0]    Parent
  in KL       Annealing
  Annealing   (if enabled)
              ⚠ warning
```

# Mechanism-Separated Routing Upgrade

## Why This Upgrade Exists

The original CNN-LSTM project followed the same research logic as the VT/VF
ECG reliability work: start with a classifier, inspect embeddings and temporal
state changes, then ask when the model should not be trusted automatically.

The latest VT/VF lesson is important for this project: a cleaner representation
does not guarantee safer downstream classification or routing. Embedding
distance, local mixing, confidence, regularity, and corruption evidence should
not be treated as direct answers. They are better used as evidence for distinct
failure mechanisms.

This upgrade therefore reframes the CNN-LSTM visual-state monitor from a single
`visual_state_risk` score into a mechanism-separated hierarchical router.

## Method

The router has two stages.

Stage 1 is boundary-first visual-state routing. It prioritizes samples with
large temporal excess, embedding shift, local temporal change, coverage risk,
or distilled visual risk. These samples are routed to re-perception, state-set
review, or operator confirmation before a high-consequence action.

Stage 2 uses a reserved residual budget for the remaining samples. It separates
the dominant residual mechanism into trajectory residual, depth/signal quality,
representation conflict, or progress/calibration inconsistency.

This mirrors the final VT/VF design principle: do not put every uncertainty
signal into one bucket. Route the likely mechanism.

## Current Smoke Evidence

Command:

```bash
python modules/mechanism_router.py \
  --input-csv <risk_trace.csv> \
  --output-dir outputs/mechanism_router \
  --action-budget 0.20 \
  --residual-reserve 0.20
```

Current output:

| Quantity | Result |
|---|---:|
| Samples | 1800 |
| Total action budget | 20.0% |
| Selected samples | 360 |
| Stage 1 boundary-first routes | 288 |
| Stage 2 reserved residual routes | 72 |
| High-risk target capture | 66.7% |
| Top 10% visual-state-risk capture | 62.8% |
| Top 25% visual-state-risk capture | 66.9% |
| RECOVER/HUMAN_REVIEW state capture | 76.5% |

Route counts:

| Stage | Mechanism | Action | Count |
|---|---|---|---:|
| automatic | automatic | continue_autonomy | 1440 |
| stage1_boundary_first | perception_boundary | re_perceive_or_request_state_set_review | 288 |
| stage2_reserved_residual | trajectory_residual | trigger_recovery_or_replan | 60 |
| stage2_reserved_residual | progress_or_calibration | slow_down_and_recheck_state | 12 |

## Interpretation

This is a research-prototype decision layer, not a safety guarantee. The useful
claim is that the project now has a clearer bridge from diagnostics to runtime
actions:

- Embedding evidence explains representation conflict but is not treated as a
  direct class decision.
- Temporal and coverage evidence support boundary-first visual-state review.
- Trajectory residual evidence supports recovery or replanning.
- Progress and calibration evidence support slowing down and rechecking state.

## Next Validation Step

The next strong experiment is to replace proxy labels with task-native
industrial failure evidence: human-action misrecognition, worker-zone events,
depth-estimation dropouts, perception interruption, robot stop/replan logs, or
real action-outcome residuals. Then evaluate whether the mechanism router
improves failure capture under a fixed action or human-review budget.

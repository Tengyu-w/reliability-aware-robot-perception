# Limitations

This project is a research prototype for VPPV-style perception reliability. It
supports a monitor-and-route research story, but it does not establish surgical
robot safety or clinical deployment readiness.

## Confirmed

- The repository contains a runnable VPPV-style monitor in
  `modules/run_vppv_perception_monitor.py`.
- The monitor distills depth, temporal, embedding, trajectory, calibration, and
  coverage-risk evidence into `visual_state_risk`.
- Feature attribution, signal-group ablation, top-risk case explanations,
  runtime route policy, and outcome-linked validation are generated.
- Public GitHub figures and tables are available under `docs/figures/` and
  `docs/tables/`.
- Supporting TUM RGB-D experiments show that global clean-reference scoring can
  fail under normal camera motion, motivating local temporal normalization.

## Not Yet Proven

- The system has not been validated on paired VPPV surgical autonomy rollouts.
- Segmentation-mask quality, perceptual regressor error, and physical-state
  estimation error are framed as VPPV dependencies but are not directly measured
  in the current dataset.
- Controlled RGB-D corruptions and synthetic trajectory residuals are proxies,
  not natural surgical failure labels.
- The selected Random Forest score is a useful distilled risk signal, not a
  calibrated clinical probability.
- Runtime states are auditable engineering rules, not formal safety proofs.

## Main Evidence Risk

The strongest metrics, including ROC-AUC 1.000 for controlled temporal
corruption detection, should be described as controlled-corruption evidence. The
more defensible research claim is the monitor design: local temporal excess,
embedding shift, trajectory residuals, and route policies can provide a
transparent reliability layer for VPPV-style autonomy.

## Most Useful Next Step

Replace the proxy labels with VPPV-native evidence: segmentation quality,
depth-map failure labels, surgical-tool state regression error, simulator
rollout failures, or tool-tracking logs. Then evaluate whether
`visual_state_risk` predicts downstream policy failure and reduces unsafe
execution through re-perception, recovery, or human review.

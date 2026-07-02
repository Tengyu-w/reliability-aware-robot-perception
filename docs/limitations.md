# Limitations

This project is a research prototype for industrial visual action recognition
and reliability-aware robot perception. The main method is visual-state risk
monitoring for uncertain action or perception states.

## Confirmed By Current Evidence

- The repository contains runnable CNN-LSTM, RGB-D/depth, temporal, calibration,
  trajectory residual, and risk-distillation components.
- Controlled TUM RGB-D corruptions are detectable in the current setup.
- Global clean-reference distance can fail under normal camera motion.
- Local temporal normalization and waveform-like excess scoring can better
  separate abnormal visual changes from normal scene/camera movement.
- Multiple reliability signals can be distilled into a lightweight
  `visual_state_risk` score.
- The scalar score can be upgraded into mechanism-separated routing that
  distinguishes boundary-first visual-state risk from residual trajectory,
  depth/signal, representation, and progress/calibration mechanisms.
- The score can be mapped into auditable runtime states:
  `NORMAL`, `SUSPECT`, `RECOVER`, and `HUMAN_REVIEW`.

## Not Yet Proven

- The system has not been validated in closed-loop robot control.
- The graded runtime response layer is currently a rule-based proof-of-concept
  demo, not a validated industrial robot policy.
- Current labels are proxy reliability labels, not natural task failure labels.
- Controlled depth corruptions are not the same as real-world perception
  failures caused by lighting, occlusion, tissue deformation, smoke, blur, or
  sensor degradation.
- The PCA descriptor is a lightweight diagnostic baseline, not a general
  pretrained RGB-D representation.
- Calibration scores show strong ranking but should not be interpreted as
  calibrated probabilities without additional calibration work.
- The primary target is industrial visual action monitoring. Visual-to-state
  consistency checking is a future extension, not a completed validation layer.
- Runtime route rules are transparent engineering rules, not formal safety
  guarantees.
- Mechanism-routing scores are currently rule-based and validation-oriented.
  They should be tuned and tested against task-native failure labels before
  being presented as an operational router.

## Main Evidence Risk

Some results, especially ROC-AUC 1.000 on controlled corruptions, are strong
inside the current experimental setup but should be worded carefully in
applications. They show that the pipeline can detect designed reliability
failures; they do not prove universal robustness.

## Next Validation Step

The strongest next experiment is to replace proxy labels with task-native
industrial evidence: human-action misrecognition cases, worker-zone events,
robot stop/replan logs, near-miss annotations, perception dropouts, or real
action-outcome residuals. Then evaluate whether `visual_state_risk` predicts
downstream failures and improves re-observation, recovery, or human-review
decisions.

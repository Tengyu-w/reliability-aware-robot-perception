# Limitations

This project is a research prototype for reliability-aware sequential robot
perception. It is intentionally broader than one surgical autonomy framework:
the main method is visual-state risk monitoring, while the VPPV-style section is
only one transfer case.

## Confirmed By Current Evidence

- The repository contains runnable CNN-LSTM, RGB-D/depth, temporal, calibration,
  trajectory residual, and risk-distillation components.
- Controlled TUM RGB-D corruptions are detectable in the current setup.
- Global clean-reference distance can fail under normal camera motion.
- Local temporal normalization and waveform-like excess scoring can better
  separate abnormal visual changes from normal scene/camera movement.
- Multiple reliability signals can be distilled into a lightweight
  `visual_state_risk` score.
- The score can be mapped into auditable runtime states:
  `NORMAL`, `SUSPECT`, `RECOVER`, and `HUMAN_REVIEW`.

## Not Yet Proven

- The system has not been validated in closed-loop robot control.
- Current labels are proxy reliability labels, not natural task failure labels.
- Controlled depth corruptions are not the same as real-world perception
  failures caused by lighting, occlusion, tissue deformation, smoke, blur, or
  sensor degradation.
- The PCA descriptor is a lightweight diagnostic baseline, not a general
  pretrained RGB-D representation.
- Calibration scores show strong ranking but should not be interpreted as
  calibrated probabilities without additional calibration work.
- The VPPV-style section does not reproduce VPPV and should not be presented as
  a new VPPV method. It is a compatibility/application framing.
- Runtime route rules are transparent engineering rules, not formal safety
  guarantees.

## Main Evidence Risk

Some results, especially ROC-AUC 1.000 on controlled corruptions, are strong
inside the current experimental setup but should be worded carefully in
applications. They show that the pipeline can detect designed reliability
failures; they do not prove universal robustness.

## Next Validation Step

The strongest next experiment is to replace proxy labels with task-native
failure evidence: robot-log failures, SLAM tracking loss, segmentation quality,
depth-estimation error, surgical-tool state regression error, simulator
rollouts, or real action-outcome residuals. Then evaluate whether
`visual_state_risk` predicts downstream failures and improves recovery or
human-review decisions.

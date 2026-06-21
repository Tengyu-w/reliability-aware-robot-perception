# Limitations

This project is presented as a research prototype. The current evidence supports
the feasibility of reliability-aware perception diagnostics, but it does not
establish closed-loop robot safety.

## Confirmed

- The repository contains runnable scripts for video embeddings, depth/point
  cloud reliability diagnostics, temporal reliability, descriptor comparison,
  calibration analysis, runtime monitoring, and trajectory residual monitoring.
- The TUM RGB-D sample workflow shows that controlled depth corruptions can be
  ranked strongly by several reliability scores.
- The scene-conditioned baseline performs poorly under camera motion, which is
  a useful negative result and motivates temporal normalization.
- PCA depth descriptors improve pose-motion sensitivity compared with simple
  global statistics in the current TUM RGB-D sample.

## Not Yet Proven

- The system has not been validated in a closed-loop robot controller.
- Controlled corruptions are not the same as naturally occurring robot
  perception or execution failures.
- The PCA descriptor is a lightweight sequence-fitted baseline, not a general
  pretrained RGB-D representation.
- Runtime states are auditable engineering rules, not formal safety guarantees.
- More seeds, stronger baselines, and task-native failure labels are needed
  before making broad claims about real-world deployment.

## Most Useful Next Step

Replace the synthetic and controlled failure targets with robot logs, simulator
rollouts, surgical-tool tracking data, SLAM tracking quality, or another
task-native failure label. Then compare reliability scores against downstream
task failure, recovery cost, and human-review efficiency.


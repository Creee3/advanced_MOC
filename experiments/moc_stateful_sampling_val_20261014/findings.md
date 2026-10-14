# Findings

2026-10-14: v0 admitted 98.6%/96.7% of available messages; its dense cost saving was mostly omitted 9B merges. v1 must demonstrate state-sensitive actual selection and stopping. Existing cloud models and private supervised-launch machinery are reusable; current read-only GPU snapshot has 24,081MiB free per card and 0% utilization, subject to recheck at launch.

The agreed val60 manifest was not present in the prior data directory. Reconstruct it using the original content fingerprint and verify the existing dev30 against the first 30 sorted official-dev hashes before selecting hashes 30:90.

Correction: the 2026-10-11 manifest already defines all90 identities and source row indices, with SHA256(seed|question_id) order, not bare content hash. v1 reuses this manifest exactly. Training uses its own content identities and excludes dev/test by question/options content.

Validation540conditions =60single +2densities*4graph arms*60. Training48questions, up to16paired contrasts/question at receiver indices1/3/4/5. Fit36question groups, calibrate12; training is explicitly a local-answer utility surrogate. No new model or backend is introduced.

11:55 heartbeat observed completed learner:47valid/1excluded train questions,376pairs, effective fit35/calibrate12. Fit labels:41negative/213zero/26positive; calibration14negative/61zero/21positive. Selectedridge1000 by prespecified MSE grid; calibration MSE0.375582 vs constant-zero0.364583, so the surrogate has not beaten the trivial predictor on this calibration metric. Only4/96 calibration contrasts get positive predicted gain. This is a warning about predictive signal and possible over-stopping, not permission to adjust the frozen policy on val. Full-graph val remains the task test; no accuracy conclusion yet. Historical planned36fit groups differ from effective35 because originalindex32 was excluded.

# Frozen protocol: selected-state utility sampling v1

## Pre-validation operational amendment (2026-10-14 11:40 Beijing)

Training stopped after35valid questions because one generation hit the unchanged2000token output cap. No val invocation or policy checkpoint existed. Preserve all old raw records; retain only labels from fully completed valid questions. Exclude the entire output-limited training question (original index32) without replacement, then process the remaining originally selected questions. Future output-limited train questions follow the same metadata-only exclusion rule. Do not change model, decoding, cap, features, budget, cost penalty or validation questions. Commit a question's labels only after all its calls pass integrity; abandoned partial labels cannot train the model. Original fit indices0:36/calibration36:48 remain; actual group counts are reported, with minimum30fit/8calibration groups required before val. Output-limit exclusion may be task-dependent and is explicitly reported as missing training data, not an unbiased selection claim. Invalid and abandoned-call token cost remains in the training cost ledger. GPU wall time already spent is deducted from the original6h ceiling. Val protocol remains unchanged.

User-authorized 2026-10-14. Hypothesis: a lightweight receiver-conditioned controller can reduce complete model-token cost relative to original MOC while preserving or improving observed quality. Success is not assumed.

## Method and migration

Keep the pinned MOC DAG, original roles/prompts, one-round ordinary generations and all-output FinalRefer. Access original ancestors up to shortest hop4, but do not put all bodies into the receiver prompt. MiniLM remains unchanged. LADIES supplies its previous-frontier-conditioned squared support mass. On this seven-node graph all positive routing-frontier entries can be explored cheaply; v1 replaces fixed body fanouts with sequential learned decisions. This is an explicit difference from v0, not a claim to have ported a complete GNN learner.

New task-specific part: fit a ridge-regression local marginal-utility model from controlled labels in {-1,0,+1}. Inputs include task/role similarity, selected-sketch semantic residual and overlap, topology mass, hop, sender/receiver roles, and remaining/incremental token budgets. Residual is task sketch minus its regularized projection onto selected sketch span (ridge .1). This is a state proxy, not an LLM hidden state or correctness certificate. Each read recomputes features. Net gain = predicted local correctness change - .02 * incremental communication tokens /1024. Select a positive-net candidate by net gain per token; expand cheaply to its hop if necessary; stop when none is positive or no body fits. No main-model selection calls. Original author serialization subsequence is retained.

## Learning and data boundaries

Official AQuA train: 48 content-unique questions, chosen by sorted content SHA256 and excluding the entirety of official dev/test. First36 question groups fit, last12 calibrate ridge strength among {10,100,1000} by paired-label MSE; ties favor stronger regularization. No refit on calibration groups. Lambda .02 and hard budget1024 are preset, not tuned on val. Degenerate/no-signal training is reported, not hidden by switching policy.

For each training question generate one original Vanilla MAS graph at alternating .3/.7 density. Freeze all upstream outputs. At receiver indices1/3/4/5 construct empty and one label-blind sampled set of1–3 ancestors, then compare the same receiver on A and A+one candidate (up to two candidates/state). Both calls share question, source texts, role, prompt, model, placement and temperature0/seed42. Original code/tool suffix is preserved. Label is local parsed answer correctness difference, **not** final graph causal utility. It can miss joint-only useful evidence and suffers stochastic label noise. Training examples use no val/test outcomes. Raw materials, pair requests/responses and training token cost are retained.

Dev30 and val60 use the pre-existing 2026-10-11 manifest, sorted by SHA256(seed|question_id), seed `moc-verification-pilot-v1-seed42`. Manifest row indices and raw official-dev SHA256 are verified; no new split is invented. Val60 gold stays local. Official test254 is not run. Historical test exposure is acknowledged.

## Validation controls

Single LLM (one author direct call), Vanilla MAS, original MOC, original MOC without Phi, and v1. 7agents, original Random DAG densities .3/.7, fixed seed42. Single runs once per question; four graph arms at two densities yield540 conditions. Same question uses the same main GPU across arms. Original role cycle snapshot reused; arm order rotated. Main Gemma2:27b Q4_0 temperature0; original merger Gemma2:9b Q4_0 temperature.1, five candidates, epsilon.1/r1/kappa45. ctx8192 and main output cap2000 are uniform. AQuA author parser signature binding and 5s code timeout are uniformly disclosed from v0. No baseline redesign.

Checkpoint/config/input hashes are frozen before first val inference. No threshold, budget or model changes in response to val results. First val question per shard is an operational smoke gate and reused in final results; no outcome-based decisions at the smoke gate.

## Metrics and interpretation

Primary cost: actual backend prompt+completion tokens for **all** ordinary/final/merge calls. Separately report model-specific tokens, receiver communication, routing/encoding time, logical sketch bytes, and one-time train-label cost. Text token totals across9B/27B are not equal FLOPs. No physical bandwidth claim.

Paired accuracy and cost vs MOC, separately per density; paired question bootstrap95% intervals and exact sign/McNemar test. Operational target: observed correct count at least MOC and total inference tokens strictly lower. This is a screening criterion, not a noninferiority proof. A prespecified5percentage-point quality margin is reported with a paired one-sided95% lower bound;60questions may be underpowered. No counting the same60questions at two densities as120independent questions. Compare without-Phi to identify savings that are merely removal of merger calls. Report actual read counts, skip/stop frequencies, hops3/4 opportunities and choices, state revisions, training label distributions, and surrogate calibration error.

## Resource and stopping rules

Existing models only; assigned LXC container only. At most2main GPUs+1merger GPU, utilization≤15% and remaining memory≥21,000/10,000MiB with two polls. Loopback private ports11448/11449/11450. Verify model digest, CUDA readiness, short GPU generation and full VRAM residency. Maximum wait180s and total GPU wall6h, worker silence600s. OOM/truncation/CPU offload/inference errors invalidate affected run; stop and release owned resources. No unbounded retries/automation. All own descendants and services are reaped at exit.

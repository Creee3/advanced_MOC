# Frozen v2.1 expected-set diagnostic full-graph validation

## Authorization and scope
User requested on 2026-10-14: “在这之前你去展开v2.1的实验吧”. This authorizes a new bounded v2.1 val-only batch despite the documented failed train-OOF gate. The failure is not erased or reinterpreted as passing. This is development/diagnostic validation on already exposed val60, not independent confirmation and not test.

Run exactly the existing60 question IDs at original densities0.3/0.7,120 new-method conditions. Reuse original paper baselines and compatible raw calls; no baseline reruns, new training/calibration calls, replacement questions, parameter tuning or test. Shared questions across two densities are60 independent question groups, not120 independent samples.

## Frozen method and selection
Use the existing centered_compact22 checkpoint from fit35, SHA256 e2ce9cd7f69d3b4e6c2e6be7af1df0a852d44b972396989e5b02428a655fd210. Compact was the lowest-MSE corrected candidate in the fixed train-OOF comparison; both corrected candidates failed the gate. Freeze this choice before full-graph val; do not compare/tune multiple v2.1 checkpoints on val.

ExpectedSetSampler permits max four-hop ancestor access and one-shot singleton/connected-pair set choices, empty=0. Select maximum total expected downstream proxy gain minus0.02*communication_tokens/1024; no upper-bound/token exploration and no mandatory reading/group quota. Unsupported metadata abstains. Budget1024 includes author message labels/envelope; original text and order remain. Bootstrap model variation is descriptive, not a valid mean-utility confidence interval. The predictor is an uncalibrated fixed-upstream/Vanilla-downstream surrogate; full-graph distribution differs.

Original author commit9c67c92507570704a7df73e452552a3f49e83897,7 roles, DAG, prompts, full Graph.arun and all-output FinalRefer unchanged. Gemma2:27b Q4_0/digest53261bc9c192c1cb5fcc898dd3aa15da093f5ab6f08e17e48cf838bb1c58abfe, MiniLM, no CoT, seed42/temperature0, context8192/output cap2000, code timeout5seconds. No merger is required by this arm, no Qwen or new weights.

## Measurements and comparisons
Per density, compare paired final correctness and full logical inference model tokens with existing MOC, original Vanilla MAS and w/o Phi attribution; include historical v1/v2 as research diagnostics. Operational target: no lower observed correct count and lower whole-process inference tokens. Equal observed correctness is not a noninferiority proof. Report paired question bootstrap confidence intervals and validity sensitivity; the old sparse MOC output-cap question remains flagged.

Audit read count/group size/hops, empty fraction, budget and unsupported-abstention frequencies; savings without selective useful communication do not establish learned adaptive routing. Preserve failed/truncated conditions and their costs; invalid outputs are not treated as correct and no output-cap relaxation or silent retry. Report all actual new-call tokens/warmup/waste plus logical costs including cache hits. One-time historical training888561tokens (v2) and older v1 training/waste657528 are separately disclosed; v2.1 offline fitting added0LLMtokens. Do not double-count already paid calls as new spend, or subtract cache reuse from deployment cost.

## Resource and recovery limits
New directory moc_v21_expected_set_val_20261014, SSH DevContainer, allocated LXC only. At most2main GPU services, private11448/49. Existing CUDAv13 libraries, num_gpu999, utilization<=15% and free>=21000MiB on two consecutive10-second checks; verify CUDA allocation, real short generation and size_vram>=size before worker start. No CPU fallback, driver/model installations, other users' resources or old-job recovery.

Prelaunch resource check found only one eligible GPU. Freeze main_service_count=1/private11448, OLLAMA_NUM_PARALLEL=1; retain both original data shards/workers sharing this serial service. This changes concurrency only, not graph/model/decoding/condition membership or GPU threshold. Wall-clock comparisons must disclose single-GPU concurrency; token/accuracy remain paired by condition.

This batch maximum2h wall clock,1200new calls and1500000new model tokens including warmup. Limits span recovery; no reset. STOP/pause/limit terminates owned processes and ports. No recurring automation is created or resumed. Supervisor handles deadline/error and owned-child cleanup.

Any manual recovery must preserve terminal state/requests/responses/costs in a unique attempt directory, preserve original run_start/budget, maintain shard and cache-conflict rejection, and skip every already committed(question_id,arm,density) condition including invalid terminal records. No infinite restart; at most two attempts for the same root cause before notification/cleanup. This request does not authorize changing frozen algorithm/model/inputs to improve results.

## Reproducibility
Before launch commit this protocol; freeze checkpoint, config, roles, val inputs, executable source and exact cache source hashes in experiment_lock/policy_lock. No val/test gold is deployed. Historical cache contains model responses only; any same-request conflicting outputs are all excluded rather than chosen by correctness. Completed local analysis keeps evaluation gold local. Download final raw archive and per-file SHA256, validate complete condition coverage and charges, then verify owned resources cleared.

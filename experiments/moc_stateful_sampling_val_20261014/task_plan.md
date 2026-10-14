# Stateful sampling v1 — execution plan

User authorization (2026-10-14): implement the agreed receiver-conditioned multi-hop method, run on cloud GPUs, and validate on val. Targets: lower total tokens with equal or better observed accuracy. Neither target is presumed true.

- [x] Implement a lightweight learned conditional marginal-utility model and sequential read/expand/stop, with selected-set state updates and real serialized-token accounting.
- [x] Freeze official train48 and the agreed dev30/val60 split; verify question identities and disjointness. Train labels use fixed upstream materials and paired receiver interventions. No val gold on the inference server.
- [x] Run meaningful source-path/budget/state/stop checks. Preserve the pinned author checkout and v0 evidence. 22/22 passed.
- [x] Run bounded train-label collection on existing GPU models, fit on train question groups, freeze checkpoint and policy before val. 47valid/1excluded, 376pairs; fit35question groups/calibrate12. Checkpoint/config/val-input hashes verified locally.
- [x] Run val60 with original Single LLM, Vanilla MAS, MOC, MOC without Phi, and v1 at original densities .3/.7. 540/540generated;1MOC .3 call reached2000cap, strict completeness not fully passed. All raw results retained, no rerun or threshold adjustment.
- [x] Download and hash-check traces, score locally, audit complete cost/quality/selection/cleanup, and record honest conclusions.160raw file hashes matched;usage matches, owned cleanup verified. Audit status explicitly passed_with_output_limit_exception. Primary observed60results and secondary59complete-case sparse comparison both reported. Sparse target fails;dense28/60ties MOC and saves67.02%inference but reads no ordinary messages;657,528training/invalid/abandoned tokens not amortized in this batch.

## Fixed boundaries

No new models, no KV representation/backend change, no test run, no restart of stopped experiments. GPU placement and a successful short generation are mandatory; no CPU offload. Only owned services/descendants may be stopped. Maximum GPU run 6h, maximum availability wait 180s.

Learned utility is a local receiver-answer surrogate, not a guarantee of final graph utility. GNN linear unbiasedness does not transfer to text. Baseline and learned-selection attribution must be distinguished.

## Errors

- Initial local PDF extraction: default Python lacks pypdf; use bundled document runtime.
- An initial broad search produced excess output; scoped subsequent reads to active files.
- Split reconstruction initially assumed bare content-hash order. Original manifest proves SHA256(seed|original question_id) order instead. Restored exact manifest indices and identities; dev30 matches and val60 retains pre-existing identity.
- 10:56 training worker0 failed an output-length check after35valid questions. Before any val inference, archived the raw failed attempt; retained280committed pairs, excluded the entire limited question at index32, discarded uncommitted partial labels, and added per-question atomic label admission. No model/cap/algorithm/val parameters changed. Added3integrity checks, all passed. Original48question selection retained without replacement; actual fit/calibration group counts will be disclosed. Existing GPU time is deducted from the6hwall budget.

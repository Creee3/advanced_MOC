# Compatibility findings

- Paper HumanEval loader uses CodeWriting and FinalWriteCode, not MathSolver / FinalRefer. Existing adapter deliberately refuses this path; a dedicated audited adapter is necessary.
- Internal tests are extracted solely from doctest examples in the public prompt. Final hidden tests are the record's `test` field and must remain outside inference.
- Author dataset has 164 transformed Python HumanEval tasks with `entry_point`, full-function prompt and an executable `test` block. Preserve author postprocess and evaluation semantics; report this exact dataset variant.
- v2.1 coverage is keyed by receiver role, maximum hop, group size and empty selected state. Most code roles differ from AQuA roles, so unchanged checkpoint transfer has unsupported actions. Report abstentions rather than inventing a mapping.
- Current cloud inventory has several eligible GPUs, but allocation must pass two consecutive checks at launch.
- OpenAI HumanEval official evaluator: https://github.com/openai/human-eval . One completion per task gives pass@1; do not claim pass@k for k>1. The original runtime is not a complete security sandbox.

Preflight findings: nested user namespaces are denied in this LXC (bwrap network setup / UID mapping fail). Switched to tested Landlock + seccomp bounded subprocesses; positive/negative code tests, blocked home reads/network/fork/writes and bounded infinite loop all pass. Eight synthetic native graph checks (both densities, both arms, passing/failing public examples) pass with exact cost settlement. Original MOC can return early (2 main calls instead of 8 in the passing synthetic case); this behavior is preserved and explicitly measured.

Observed training role support: Inspector, Math Solver, Mathematical Analyst, Programming Expert. New code roles other than Programming Expert remain unsupported. This limitation is frozen before seeing any HumanEval results.

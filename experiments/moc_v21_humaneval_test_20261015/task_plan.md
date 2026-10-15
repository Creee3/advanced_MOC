# HumanEval v2.1 cross-domain test

User authorization (2026-10-15): start HumanEval directly on test and compare our method with official MOC on cloud GPUs.

Scope: author's 164 HumanEval records, frozen v2.1 checkpoint, original CodeWriting / FinalWriteCode agents, seven-agent random DAGs at densities 0.3 and 0.7, one round; 656 terminal conditions. No HumanEval training, validation, threshold tuning, or extra baselines. AQuA results remain completed and are not rerun.

## Phases
1. [complete] Audit author code-generation execution, dataset, routing compatibility and evaluation isolation.
2. [complete] Implement CodeWriting-specific adapter and hardened execution bridge; no-model regression preflight.
3. [complete] Freeze protocol, configuration, input/checkpoint/source hashes and commit; deploy and verify cloud files. Frozen source commit 1238b67.
4. [complete] Two resource checks, CUDA/full-VRAM verification, bounded launch and live progress confirmation. Started 14:13:07 Beijing, GPU1 main / GPU3 merger, real response records and valid terminal conditions confirmed.
5. [in progress] Quiet ten-minute monitor `moc` ACTIVE; on completion download raw records, audit and paired pass@1 / full-token analysis.

## Constraints
- Keep v2.1 expected-utility predictor and its observed-role support unchanged. Unknown roles abstain; report coverage and any empty-selection degeneration explicitly. Do not invent code-role labels or train on test.
- Preserve author prompt-derived internal tests and early-return semantics using only selected upstream messages. Official final tests are never supplied to generation/routing.
- Execute generated code only in isolated bounded subprocesses in allocated LXC; no unbounded eval in orchestration process.
- Reuse existing Gemma2 27B/9B, MiniLM and CUDA runtime; one shared main GPU and separate merger GPU. Fresh bounded budget; no old task recovery.
- No test-based modification after lock. Retain all terminal conditions including failures, usage and discarded calls.

## Errors
| Failure | Resolution |
| --- | --- |
| bwrap nested network namespace setup denied | No change to host/container permissions; use Landlock and seccomp instead. |
| bwrap nested UID map denied | Stop namespace attempts; fail-closed isolation bridge passes all restriction probes. |
| Local regression invocation lacked package / asset paths and tokenizers runtime | Use the existing cloud conda environment and original asset paths; all 53 existing tests pass, no installs. |

# Frozen HumanEval transfer comparison (2026-10-15)

User requests direct HumanEval test, then confirms both previous graph densities and ten-minute cloud checks. This is a new batch; the audited AQuA batch is complete and will not be restarted.

## Question and scope

Does the frozen AQuA-trained v2.1 evidence-set selector transfer to Python code generation while preserving/improving pass@1 and reducing complete inference tokens compared with original MOC? This is a cross-domain transfer diagnostic. It does not establish a HumanEval-trained selector's performance.

- MOC's own `humaneval_test_n164.jsonl`: 164 tasks, transformed doctests and reworded prompts from MultiPL-E's Python HumanEval variant. Do not present it as byte-identical to OpenAI's original prompts.
- Two arms: original MOC (K=2 and native five-candidate Phi) and frozen v2.1 (Kmax=4, at most one singleton/connected pair, 1024 serialized communication-token budget, cost penalty 0.02, no Phi).
- Two fixed random DAGs: seven agents, seed 42, densities 0.3 and 0.7. 164 x 2 x 2 = 656 terminal conditions. One round, no temporal edges.
- Roles: Project Manager, Algorithm Designer, Programming Expert, Test Analyst, Bug Fixer, Project Manager, Algorithm Designer; original CodeWriting prompts and FinalWriteCode aggregation, all final ordinary outputs retained.
- Gemma2:27b Q4_0, temperature 0, seed 42, context 8192, output 2000; original Gemma2:9b Q4_0 merger, temperature 0.1, same explicit placement/context/seed as prior experiment. No merger num_predict override.

## Unchanged predictor and compatibility boundary

Checkpoint SHA256: e2ce9cd7f69d3b4e6c2e6be7af1df0a852d44b972396989e5b02428a655fd210. Reused exactly from completed AQuA experiment. No fitting, code-role mapping, support relaxation or threshold change. The four novel code roles (Project Manager, Algorithm Designer, Test Analyst, Bug Fixer) abstain under the predictor's existing support rule. Only Programming Expert shares a trained role name; this is metadata support, not proof of semantic transfer. Report support coverage, empty decisions, selected hops and final-system behavior alongside accuracy. Do not attribute cost savings to successful learned code routing if most agents simply abstain.

The opt-in CodeWriting adapter changes only available upstream evidence. Native internal tests / early return inspect selected raw sources, including selected far ancestors; dropped direct neighbors cannot leak through tests or fallback. Selection order and serialization remain unchanged. Original MOC still uses its complete direct-neighbor early-return path plus original ISM. Differences in early returns and resulting model-call count are measured, not forced to eight calls. Root nodes generate normally. FinalWriteCode continues to see every ordinary output in each arm.

## Evaluation and execution

Only public prompts, names and entry points are deployed to generation workers. Hidden test bodies remain local until all 656 generation conditions reach a terminal state. No scoring or policy updates on partial outputs. Then score every completion with the author's `HumanEvalDataset.postprocess_answer` and `PyExecutor.execute(code,[test],timeout=100)` semantics: typing imports followed by generated full function and the MOC-supplied executable test block. One completion per task/arm/density: pass@1 only. No repair, resampling, extra judge, or HumanEval training/val. See primary evaluation reference: https://github.com/openai/human-eval .

Unsafe in-process `exec` is replaced equally in both arms by a fresh system-Python subprocess per test. Landlock permits only standard-library/system-library reads, denying workspace/home reads and filesystem writes. Seccomp denies networking, fork/clone/exec, process inspection/control and namespace/mount facilities. Memory 768 MiB, file output 1 MiB, 64 descriptors, no core dumps, CPU bound and external process-group timeout; environment cleared. Public internal tests keep original timeout 10 seconds; final test keeps original 100 seconds. No user-namespace or CUDA changes. Isolation restrictions can affect unusual generated solutions; report those cases separately, never silently call them algorithm failures. This is a practical bounded evaluator, not a claim of complete adversarial security.

## Cost, allocation, failure handling

Use one shared 27B GPU for both workers/arms and a distinct 9B merger GPU, private ports 11448 / 11450, only allocated LXC. Two consecutive checks 10 seconds apart: utilization <=15%, free main >=21000 MiB / merger >=10000 MiB. CUDA allocation, actual short generation, model digest and full VRAM residency are mandatory. No CPU fallback or downloads.

Fresh batch limits: 8 hours absolute wall window from launch, 20,000 newly dispatched calls, 12,000,000 actual model tokens plus conservative unknown-usage upper bounds and active reservations. All warmup/waste counts included; never reset on resume. Stop and notify at limit. Recover only within original window after specific diagnosis, unique attempt archive, ownership cleanup and cost reconciliation. Retain all terminal `(question_id,arm,density)` conditions including invalid; never regenerate them to improve pass@1. Two same-root-cause failed recoveries end automatic recovery. No deletion of STOP/status/raw or old experiment restart.

Report full logical main+merger prompt/completion tokens for deployment, actual new paid tokens, cache reuse, wasted/unknown costs and training costs separately. Frozen model's prior fitting cost 888561 tokens is a reused one-time cost; show inference-only and an explicitly labeled conservative charge-once sensitivity, do not charge it per task or hide it.

## Analysis lock

Report each density separately. Primary all-164 system pass@1 counts invalid generation as unsuccessful; common-valid paired sensitivity excludes pairs with either generation invalid. Report paired accuracy difference and question bootstrap 95% interval (20,000 replicates, seed42), exact paired McNemar p, paired token change and invalid/timeout/isolation failure counts. No significance or equivalence claim when unsupported. Compare published MOC scores only with explicit model/data/recipe differences; primary fair comparison is the original MOC arm run here.

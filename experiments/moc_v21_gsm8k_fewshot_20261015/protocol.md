# Approved gsm8k: small-task adaptation of v2.1

User approved 2026-10-15: train24, val120, author test300; densities .3/.7.
User additionally authorized two GPUs, one main per dataset concurrently, and 10-minute monitoring.
Two GPUs maximum: parallel train/fit/val/adapted-test stages, then serial original-MOC stages using one27B+one9B. No third GPU. Original per-dataset8h deadline includes waiting, never reset.

Data: split_manifest.json. GSM train/val are disjoint samples of official train7473; Pro uses one official-validation question per category for fit, remaining four for val. Author test bytes/answers unchanged. Train24 is the planned count; excluded incomplete questions are reported, not silently replaced. Val/test do not enter fitting, prompt answers or hyperparameter selection.

Algorithm unchanged: MiniLM, maxhop4, one-shot empty/singleton/direct-edge pair, serialized1024token budget, cost penalty.02, compact22 weighted ridge10, bootstrap32(seed42) descriptive only. Two outcome-blind receiver-role episodes per training problem, alternating density. Downstream signed correctness labels from fixed upstream/Vanilla downstream replay; this is a proxy, not ground-truth full-MAS utility. Same author roles, DAG seed42, task prompts, all-output FinalRefer and original parsers.

Only native-domain thin interface added for AnalyzeAgent; Wiki Searcher unsupported and absent from pinned seven roles. Math executes original separate globals/locals semantics in restricted Landlock/seccomp subprocess,5sec, with permission/resource restrictions disclosed. Both arms share this executor. No author source edit.

Val compares frozen AQuA v2.1 and task-trained v2.1; test compares task-trained v2.1 and originalMOC(K2, originalISM and five9B candidates). No new invented baseline or duplicate baseline loop. Baseline reuse audit first; only exact matched results reused. Missing MOC results generated once. All terminal invalid outcomes retained; only unfinished work resumes from exact compatible cache. Source and policy frozen before val/test; no fitting on val or parameter search.

Gemma2:27b/9b Q4_0, MiniLM, temperature0 main, original merger.1, seed42, context8192/main output2000, num_gpu999, existing CUDAv13 only. At most two GPUs globally. Each dataset cap8h absolute/20000newcalls/16000000actual+unknown-upper+pending tokens. STOP/limits never bypassed; same-root-cause recovery <=2, preserve unique archives and all costs. No Qwen/new models/drivers or other users' processes.

Report paired per-density accuracy+20000 question bootstrap seed42+McNemar; main+merger prompt/completion logicaltokens including cache hits, encoder/router time, actual training/warmup/discard/unknown costs separately; support/empty/singleton/pair/hops. Equal observed counts or CI crossing0 do not prove noninferiority. Zero-target-training transfer and few-shot adaptation must remain distinct. No test feedback changes.

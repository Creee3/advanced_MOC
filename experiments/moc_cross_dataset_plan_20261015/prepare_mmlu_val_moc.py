"""Build the separately authorized original-MOC val supplement; no model calls."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

BASE = Path(__file__).resolve().parent.parent
SOURCE = BASE/'moc_v21_mmlu_pro_fewshot_20261015'
OUT = BASE/'moc_mmlu_pro_val_baseline_20261015'
REMOTE = '/home/liuxin/ICML2027/experiments/'+OUT.name

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

assert not OUT.exists(), 'Do not overwrite an existing supplement'
OUT.mkdir()
for p in SOURCE.glob('*.py'):
    shutil.copyfile(p, OUT/p.name)
shutil.copytree(SOURCE/'coding_base', OUT/'coding_base', ignore=shutil.ignore_patterns('__pycache__'))
(OUT/'data').mkdir()
shutil.copyfile(SOURCE/'data/val_inputs.jsonl', OUT/'data/val_inputs.jsonl')
for name in ('historical_calls_LOCAL_ONLY.jsonl', 'val_historical_calls_LOCAL_ONLY.jsonl', 'merger_historical_calls_LOCAL_ONLY.jsonl'):
    (OUT/'data'/name).write_text('', encoding='utf-8')
for name in ('policy_lock.json', 'group_utility_model.json'):
    shutil.copyfile(SOURCE/'val_review_20261015/records'/name, OUT/name)
shutil.copyfile(SOURCE/'frozen_aqua_model.json', OUT/'frozen_aqua_model.json')

config = json.loads((SOURCE/'config.json').read_text(encoding='utf-8'))
config.update(extension_directory=REMOTE+'/coding_base', main_ports=[11451], merger_port=11452,
              phase='authorized_supplemental_original_moc_val', val_run=True, test_run=False,
              train_questions=0, test_questions=0, validation_conditions=112, test_conditions=0,
              conditions_total=112, max_new_calls=6000, max_new_model_tokens=4000000,
              allowed_gpu_uuids=['GPU-0a075077-3745-799a-aff3-f0d34a17b6ac',
                                 'GPU-89523765-dcd4-46ea-1f2d-4ff7d509138c'])
config['gpu_policy']['max_wall_seconds'] = 7200
write(OUT/'config.json', config)

text = (OUT/'supervise.py').read_text(encoding='utf-8')
start = text.index('def verify_coverage(')
end = text.index('\ndef main():', start)
text = text[:start]+'''def verify_coverage(adaptive=False):
    results=[r for p in (ROOT/'val').glob('shard*/results.jsonl') for r in rows(p)]
    keys=[(r['question_id'],r['arm'],r['density']) for r in results]
    expected={(q['question_id'],'moc',d) for q in rows(ROOT/'data/val_inputs.jsonl') for d in (.3,.7)}
    assert len(keys)==len(set(keys)) and set(keys)==expected,(len(keys),len(expected))

''' + text[end:]
text = text.replace("now=gpu.gpu_snapshot();stable=", "now=[g for g in gpu.gpu_snapshot() if g['uuid'] in CONFIG['allowed_gpu_uuids']];stable=")
text = text.replace("assert args.mode=='adaptive' and not args.resume", "assert args.mode=='baseline' and not args.resume")
text = text.replace('deadline=time.time()+28800', "deadline=time.time()+CONFIG['gpu_policy']['max_wall_seconds']")
start = text.index("        if args.mode=='adaptive':")
end = text.index('    except BaseException as e:', start)
text = text[:start]+'''        assert args.mode=='baseline'
        stage('val','moc',services,deadline)
        verify_coverage(False)
        publish(phase='complete')
''' + text[end:]
assert "stage('test'" not in text and "stage('training'" not in text
(OUT/'supervise.py').write_text(text, encoding='utf-8')

# Only the supplemental val trace label changes; merger requests are byte-identical.
text = (OUT/'merger_backend.py').read_text(encoding='utf-8')
text = text.replace("stage='test'", "stage='val'")
(OUT/'merger_backend.py').write_text(text, encoding='utf-8')
write(OUT/'baseline_reuse.json', dict(reusable_conditions=[], reason='No original MOC results for these locked val56 questions/densities; prior transfer/adapted runs preserved.'))
write(OUT/'authorization.json', dict(date='2026-10-15', user_request='补测同一MMLU-Pro val的原MOC；允许新增两张空闲GPU立即独立运行',
    scope='56 locked val questions x densities0.3/0.7 x original MOC only',
    max_new_gpus=2, max_total_chat_gpus_during_overlap=4,
    parent_experiment=SOURCE.name, parent_experiment_lock_sha256=sha(SOURCE/'experiment_lock.json'),
    no_training_or_test=True, no_policy_changes=True, max_wall_seconds=7200,
    max_new_calls=6000, max_new_model_tokens=4000000))
(OUT/'protocol.md').write_text('''# 原MOC的MMLU-Pro val补测

用户在2026-10-15明确批准：同一val56、密度0.3/0.7、原MOC共112条件；临时新增两张GPU独立运行，现有两批不打断，重叠时本聊天合计最多四张。原两批的预算与截止不重置。

复制父实验的原作者AnalyzeAgent/FinalRefer、七角色、DAG/seed42、Gemma2:27b/9b Q4_0、MiniLM、原MOC K2/ISM/五候选、temperature0/.1、context8192、main output2000、num_gpu999和受限执行。父实验val输入逐字一致；gold仅本地评分，不上传。新目录只执行val/moc，绝不训练或运行test；父实验代码、配置、策略与现有结果均不改动。

独立预算：7200秒绝对截止（含等待），6000新增调用、4000000模型token；actual+unknown保守上界+pending共同计入。两个worker共享新27B服务11451，9B服务11452；只允许GPU0与GPU4对应UUID，连续两次10秒util<=15%且main free>=21000MiB、merger>=10000MiB，再真实CUDA分配、短生成、digest和size_vram>=size核验。健康不重复launch。恢复前唯一归档、费用结算审计与自有资源清理核验；invalid保留并跳过，相同根因最多两次，预算/STOP优先。

新增运行管理仅将supervise限定为val/moc、覆盖112终态和指定GPU，并把合并器trace的stage标签改为val；runner、原模型请求和算法代码不改。空历史缓存明确归档；所有调用和费用保留，不复跑已完成对照。输出完整后下载raw/trace/results/SHA，原作者解析器在本地val gold评分，与已完成两组按题配对比较，并报告主模型+合并器全流程逻辑token、实际新增/失败/未知成本。首次训练成本在父实验单列。
''', encoding='utf-8')
(OUT/'task_plan.md').write_text('1. Freeze exact val input and original-MOC protocol.\n2. CPU synthetic native-Graph and isolation preflight.\n3. Launch on two additional idle GPUs;112 val/moc conditions only.\n4. Verify cleanup, download raw and hashes.\n5. Local three-arm paired accuracy/cost report.\n', encoding='utf-8')
(OUT/'progress.md').write_text('Prepared after explicit two-additional-GPU authorization; awaiting CPU preflight and launch.\n', encoding='utf-8')
parent_lock = json.loads((SOURCE/'experiment_lock.json').read_text(encoding='utf-8'))
source_files = [*OUT.glob('*.py'), * (OUT/'coding_base/moc_adaptive_sampling').glob('*.py'), OUT/'config.json']
data_files = [* (OUT/'data').glob('*.jsonl'), OUT/'policy_lock.json', OUT/'group_utility_model.json',
              OUT/'frozen_aqua_model.json', OUT/'authorization.json', OUT/'baseline_reuse.json']
write(OUT/'experiment_lock.json', dict(source_hashes={p.relative_to(OUT).as_posix():sha(p) for p in sorted(source_files)},
    data_hashes={p.relative_to(OUT).as_posix():sha(p) for p in sorted(data_files)},
    local_gold_sha256={'val':parent_lock['local_gold_sha256']['val']}, author_commit=config['author_commit'],
    parent_experiment_lock_sha256=sha(SOURCE/'experiment_lock.json')))
names=set(json.loads((OUT/'experiment_lock.json').read_text(encoding='utf-8'))['source_hashes'])
names.update(p.relative_to(OUT).as_posix() for p in data_files)
names.update(('experiment_lock.json','protocol.md','task_plan.md','progress.md'))
with tarfile.open(OUT/'upload_LOCAL_ONLY.tar.gz','w:gz') as tar:
    for name in sorted(names):tar.add(OUT/name,arcname=name,recursive=False)
print(json.dumps(dict(folder=str(OUT),files=len(names),archive_sha256=sha(OUT/'upload_LOCAL_ONLY.tar.gz'))))

"""Read-only final integrity snapshot; never regenerate conditions or scores."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    status=read(ROOT/'status.json');cfg=read(ROOT/'config.json');lock=read(ROOT/'experiment_lock.json')
    assert status['phase']=='complete' and status['cleanup_complete']
    assert read(ROOT/'scoring_complete.json')['conditions']==656
    assert read(ROOT/'scoring_status.json')['phase']=='complete'
    assert read(ROOT/'budget.json')['reserved_tokens']==0
    for name,digest in {**lock['source_hashes'],**lock['data_hashes']}.items():assert sha(ROOT/name)==digest,name
    author=Path(cfg['author_directory'])
    assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()==cfg['author_commit']
    clean=subprocess.check_output(['git','-C',str(author),'status','--porcelain'],text=True).strip();assert not clean
    pids=[status['supervisor_pid'],*status['worker_pids'],*[r['pid'] for r in status['services']]]
    owned={str(pid):Path('/proc',str(pid)).exists() for pid in pids};assert not any(owned.values())
    ports={}
    for port in [*cfg['main_ports'],cfg['merger_port']]:
        with socket.socket() as probe:ports[str(port)]='open' if probe.connect_ex(('127.0.0.1',port))==0 else 'closed'
    assert set(ports.values())=={'closed'}
    author_files={name:hashlib.sha256((author/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for name in ('datasets/data_process.py','src/graph/graph.py','src/agents/code_writing.py','src/agents/final_decision.py',
                     'src/prompt/humaneval_prompt_set.py','src/tools/coding/python_executor.py')}
    proof=dict(checked_utc=datetime.now(timezone.utc).isoformat(),owned_pids=owned,private_ports=ports,
        author_commit=cfg['author_commit'],author_git_status=clean,author_lf_sha256=author_files,
        generation_conditions=656,scored_conditions=656,cleanup=read(ROOT/'cleanup.json'))
    (ROOT/'final_live_cleanup_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    files=[]
    for path in ROOT.rglob('*'):
        relative=path.relative_to(ROOT)
        if any(part in ('attempts','__pycache__','.git') for part in relative.parts):continue
        if path.is_file() and path.suffix in ('.py','.json','.jsonl','.md','.log'):
            assert not path.is_symlink();files.append(path)
    files.sort()
    out=ROOT/'attempts'/datetime.now(timezone.utc).strftime('final_completed_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    archive=out/'final_raw_records.tar.gz'
    manifest=dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        sizes={p.relative_to(ROOT).as_posix():p.stat().st_size for p in files},generation_conditions=656,scored_conditions=656)
    with tarfile.open(archive,'w:gz') as tar:
        for path in files:tar.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    manifest.update(archive_sha256=sha(archive),archive=str(archive),completed_utc=datetime.now(timezone.utc).isoformat())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(directory=str(out),files=len(files),archive_sha256=manifest['archive_sha256'],bytes=archive.stat().st_size)))


if __name__=='__main__':main()

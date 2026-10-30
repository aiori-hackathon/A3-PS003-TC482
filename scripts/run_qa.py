"""Run the automated tests and both fast-mode dataset experiments, preserving logs."""
from pathlib import Path
import json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'; REPORTS.mkdir(exist_ok=True)
env=os.environ.copy()
env['PYTHONPATH']=str(ROOT/'src') + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
steps=[
    ('pytest', [sys.executable,'-m','pytest','-v']),
    ('pipeline', [sys.executable,'-m','hilofed.pipeline','--dataset','both','--fast','--max-rows','1200','--rounds','3']),
    ('report', [sys.executable,'scripts/build_report.py']),
]
summary={'started_at_epoch':time.time(),'steps':[]}
for name,cmd in steps:
    start=time.perf_counter()
    proc=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=REPORTS/f'{name}_output.txt'; log.write_text(proc.stdout,encoding='utf-8')
    summary['steps'].append({'name':name,'command':' '.join(cmd),'exit_code':proc.returncode,'runtime_seconds':round(time.perf_counter()-start,3),'log':str(log.relative_to(ROOT))})
    print(proc.stdout)
    if proc.returncode:
        summary['status']='FAIL'; summary['failed_step']=name
        (REPORTS/'qa_summary.json').write_text(json.dumps(summary,indent=2))
        raise SystemExit(proc.returncode)
summary['status']='PASS'; summary['finished_at_epoch']=time.time()
(REPORTS/'qa_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

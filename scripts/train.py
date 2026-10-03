"""Train diffusion on data/; --smoke runs one small CPU optimizer step."""
import argparse, subprocess, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--outdir',type=Path,default=ROOT/'outputs/training')
    p.add_argument('--device',choices=['cpu','cuda'],default=None)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--batch-gpu',type=int,default=1)
    p.add_argument('--smoke',action='store_true')
    a,extra=p.parse_known_args()
    device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cmd=[sys.executable,'-m','mind_diffusion.cli.train','--data',str(a.data_root/'holoplane'),'--conditions',str(a.data_root/'dataset.json'),'--name-list',str(a.data_root/'train.txt'),'--outdir',str(a.outdir),'--condition-profile','v3','--batch',str(a.batch),'--batch-gpu',str(a.batch_gpu),'--device',device]
    if a.smoke: cmd += ['--device','cpu','--batch','2','--batch-gpu','1','--workers','0','--total-kimg','0.002','--model-channels','8','--channel-mult','1,1','--num-blocks','1']
    subprocess.run(cmd+extra,cwd=ROOT,check=True)
if __name__=='__main__': main()

"""Generate and decode with the released default pair."""
import argparse,json,os,pathlib,subprocess,sys
def main():
    root=pathlib.Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',type=pathlib.Path,default=pathlib.Path('outputs/default_demo'))
    p.add_argument('--raw-C',default='0.09,0.02,0.02');p.add_argument('--count',type=int,default=4);p.add_argument('--device',choices=['cpu','cuda'],default=None);p.add_argument('--seed',type=int,default=None)
    p.add_argument('--batch-size',type=int);p.add_argument('--config',type=pathlib.Path,default=root/'scripts/default.json');a=p.parse_args()
    import torch
    a.device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    c=json.loads(a.config.read_text());a.seed=a.seed if a.seed is not None else c['seed'];c['batch_size']=a.batch_size if a.batch_size is not None else c['batch_size'];out=a.outdir.resolve();env=dict(os.environ)
    env['PYTHONPATH']=str(root/'src')+os.pathsep+env.get('PYTHONPATH','')
    dm=(root/c['dm']).resolve();ae=(root/c['ae']).resolve()
    for path in (dm, ae):
        if not path.is_file():
            p.error(f'Missing checkpoint: {path.name}. Place the released weights in checkpoints/.')
    if a.count < 1 or c['batch_size'] < 1: p.error('count and batch-size must be positive')
    cmds=[[sys.executable,'-m','mind_diffusion.cli.generate','--network',str(dm),'--outdir',str(out/'generated'),
        '--raw-C',a.raw_C,'--condition-profile',c['condition_profile'],'--steps',str(c['steps']),'--cfg-scale',str(c['cfg_scale']),
        '--sigma-max',str(c['sigma_max']),'--batch-size',str(c['batch_size']),
        '--noise-device',c['noise_device'],'--seed',str(a.seed),'--count',str(a.count),'--device',a.device]]
    for i in range(a.count):cmds.append([sys.executable,'-m','mind_holoplane.cli.decode','--checkpoint',str(ae),'--latent',str(out/f'generated/sample_{i:04d}.npy'),
        '--outdir',str(out/f'decoded_{i:04d}'),'--resolution','64','--sym',str(c['sym']),'--device',a.device])
    for cmd in cmds:subprocess.run(cmd,cwd=root,env=env,check=True)
    from mind_holoplane.preview import save_geometry_preview
    title=f'Condition C={a.raw_C} → diffusion → holoplane → OBJ | steps={c["steps"]}, CFG={c["cfg_scale"]}'
    save_geometry_preview(out/'generated/sample_0000.npy',out/'decoded_0000/field.npy',out/'preview.png',title)
    print(json.dumps(dict(status='completed',config=c,outdir=str(out),count=a.count)))
if __name__=='__main__':main()

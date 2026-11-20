"""One sequential A/B/C/D study; no dataset discovery, packages or implicit GPU launch."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ['XFORMERS_DISABLED']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from src.ssl.checkpoints import atomic_json,run_lock,validate_snapshot
from src.sem_cellstyle.recipe import DEFAULT_CONFIG,configuration,resolve,require_initialization,contract
from src.sem_cellstyle.regularization_study import (ARMS,arm_templates,arm_config,common_identity,
    calibrate,require_calibration)
from src.ssl.upstream import ROOT
from src.sem_cellstyle.regularization import options

DEFAULT_OUTPUT=ROOT/'outputs/sem_ssl_regularization/v1_seed42'


def write_config(path,cfg):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if OmegaConf.to_container(OmegaConf.load(path))!=OmegaConf.to_container(cfg):
            raise ValueError('Frozen study configuration changed: '+str(path))
    else:
        temporary=path.with_suffix('.partial.yaml'); OmegaConf.save(cfg,temporary); temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def run_study(args,base,dataset,collate,data,initialization,runner):
    from src.sem_cellstyle.benchmark import validate_choice,benchmark_identity
    root=args.output.resolve()
    # Only a memory candidate may vary before calibration; physical batch is
    # thereafter bound to the receipt and all four scientific training contracts.
    candidates=[base.runtime.physical_batch] if base.runtime.physical_batch else [16,32]
    identity=common_identity(resolve(base,len(dataset),candidates[0]),data,initialization,include_batch=False)
    token_weights={key:options(base)[key] for key in ('lambda_cls','lambda_patch')}
    plan=dict(feature_token_weights=token_weights,schema='sem-cellstyle-regularization-grid-v1',common_identity=identity,
              candidates=candidates,calibration_warmup_updates=args.calibration_warmup_updates,
              calibration_probes=args.calibration_probes,benchmark_warmup_updates=args.benchmark_warmup_updates,
              benchmark_measured_updates=args.benchmark_measured_updates,arms=list(ARMS),validation_only=True,
              evaluation_passes=[0,1,5,10],all_heads_at_matching_milestones=True,
              classification_splits=list(base.evaluation.get('classification_splits',['A','B'])))
    plan['head_profile']=base.evaluation.get('head_profile','legacy')
    plan['allow_cuda_resume_update_drift']=base.runtime.get('allow_cuda_resume_update_drift', False)
    with run_lock(root):
        path=root/'study.json'
        if path.exists():
            if read(path)!=plan: raise ValueError('Study recipe/data/code/protocol changed; start a new study root')
        else:
            if any(p.name!='.writer.lock' for p in root.iterdir()): raise FileExistsError('New study output must be empty')
            atomic_json(path,plan)
        base_file=root/'configs/base.yaml'; write_config(base_file,base)
        # Worst-case-reference precheck. Unit coefficients are diagnostic only;
        # final benchmarks below use the actual frozen coefficients in each arm.
        provisional=root/'configs/D_memory_probe.yaml'
        probe=arm_config(base,'D',root/'DISPOSABLE_reference_probe',
                         dict(lambda_sp=1.,lambda_feat=1.,**token_weights))
        write_config(provisional,probe)
        precheck=root/'checks/reference_D'
        if not (precheck/'benchmark.json').exists():
            runner([sys.executable,'-m','scripts.benchmark_sem_cellstyle_a100','--config',str(provisional),
                    '--output',str(precheck),'--batch-sizes',*map(str,candidates),*(['--resume'] if precheck.exists() else []),
                    '--warmup-updates',str(args.benchmark_warmup_updates),'--measured-updates',str(args.benchmark_measured_updates)],
                   root/'logs/reference_memory_check.log')
        memory=read(precheck/'benchmark.json')
        if memory['benchmark_identity']!=benchmark_identity(resolve(probe,len(dataset),candidates[0]),data,initialization):
            raise ValueError('Memory-probe identity differs from study')
        batch=memory['recommendation']['physical_batch']
        if batch not in candidates: raise ValueError('No viable reference-encoder batch; stop without altering the recipe')
        validate_choice(memory,resolve(probe,len(dataset),batch),data,initialization)
        chosen=resolve(base,len(dataset),batch)
        calibration=root/'calibration.json'
        if not calibration.exists():
            runner([sys.executable,'-m','scripts.run_sem_cellstyle_regularization_grid','--stage','calibrate',
                    '--config',str(base_file),'--output',str(root),'--physical-batch',str(batch),
                    '--calibration-warmup-updates',str(args.calibration_warmup_updates),
                    '--calibration-probes',str(args.calibration_probes)],root/'logs/calibration.log')
        frozen=read(calibration)
        configs={}
        for arm in ARMS:
            cfg=arm_config(chosen,arm,root/'runs'/arm,frozen['coefficients'],calibration)
            require_calibration(cfg,data,initialization)
            file=root/f'configs/{arm}.yaml'; write_config(file,cfg); configs[arm]=(cfg,file)
            benchmark=root/f'checks/{arm}'
            if not (benchmark/'benchmark.json').exists():
                runner([sys.executable,'-m','scripts.benchmark_sem_cellstyle_a100','--config',str(file),
                    '--output',str(benchmark),'--batch-sizes',str(batch),*(['--resume'] if benchmark.exists() else []),
                    '--warmup-updates',str(args.benchmark_warmup_updates),'--measured-updates',str(args.benchmark_measured_updates)],
                    root/f'logs/{arm}_gpu_check.log')
            validate_choice(read(benchmark/'benchmark.json'),cfg,data,initialization)
        atomic_json(root/'checks_complete.json',dict(physical_batch=batch,effective_batch=128,
            accumulation_steps=128//batch,coefficients=frozen['coefficients'],all_arms_checked=True,
            allow_cuda_resume_update_drift=plan['allow_cuda_resume_update_drift'],
            scientific_weights_initialized_from_original=True,training_launched=False))
        if args.stage=='checks': return
        for arm,(cfg,file) in configs.items():
            run=root/'runs'/arm
            if (run/'run.json').exists() and read(run/'run.json')['contract']!=contract(cfg,data,initialization,'cuda'):
                raise ValueError('Existing arm contract differs: '+arm)
            state=read(run/'status.json') if (run/'status.json').exists() else {}
            if state.get('status')!='stage_complete' or state.get('step')!=cfg.experiment.run_until_step:
                command=[sys.executable,'-m','scripts.train_sem_cellstyle','--config',str(file),'--stage','train',
                         '--benchmark-result',str(root/f'checks/{arm}/benchmark.json')]
                if (run/'run.json').exists(): command.append('--resume')
                runner(command,root/f'logs/{arm}_train.log')
                state=read(run/'status.json')
                if state.get('status')!='stage_complete' or state.get('step')!=cfg.experiment.run_until_step:
                    raise InterruptedError('Arm stopped before its declared boundary; later arms were not started')
            marker=root/f'verified/{arm}.json'
            milestones=[step for step in cfg.experiment.milestones if step<=cfg.experiment.run_until_step]
            signatures={str(step):validate_snapshot(run/f'checkpoints/step_{step:06d}') for step in milestones}
            if not marker.exists():
                runner([sys.executable,'-m','scripts.train_sem_cellstyle','--config',str(file),'--stage','verify'],
                       root/f'logs/{arm}_verify.log')
                atomic_json(marker,signatures)
            elif read(marker)!=signatures: raise ValueError('Verified milestone files changed in arm '+arm)
        # Shared original-Large baseline once; remaining validation artifacts are
        # separate per arm, with all six probes at the same checkpoints.
        base_evaluation=root/'evaluation/original_large'
        first=configs['A'][1]
        command=[sys.executable,'-m','scripts.evaluate_sem_cellstyle','--config',str(first),
                 '--run',str(root/'runs/A'),'--output',str(base_evaluation),'--passes','0',
                 '--full','--execute','--report-output',str(root/'evaluation_reports/original_large')]
        if args.verified_data_root: command+=['--verified-data-root',str(args.verified_data_root.resolve())]
        runner(command,root/'logs/original_large_evaluation.log')
        for arm,(_,file) in configs.items():
            command=[sys.executable,'-m','scripts.evaluate_sem_cellstyle','--config',str(file),
                     '--run',str(root/'runs'/arm),'--output',str(root/'evaluation'/arm),
                     '--passes','1','5','10','--full','--execute',
                     '--report-output',str(root/'evaluation_reports'/arm)]
            verified=args.verified_data_root or next(base_evaluation.glob('dinov2_vitl14_*'))
            command+=['--verified-data-root',str(Path(verified).resolve())]
            runner(command,root/f'logs/{arm}_evaluation.log')
        # Diagnostics are a separate, training-data-only frozen-encoder pass.
        # Disabled arms have NO extra reference or extra forwards during training.
        if not (root/'diagnostics.json').exists():
            runner([sys.executable,'-m','scripts.run_sem_cellstyle_regularization_grid','--stage','diagnostics',
                    '--config',str(base_file),'--output',str(root)],root/'logs/feature_diagnostics.log')
        from src.sem_cellstyle.regularization_report import write_report
        print(write_report(root),flush=True)
        atomic_json(root/'complete.json',dict(stage='initial_10_pass_study_complete',arms=list(ARMS),
                    physical_batch=batch,coefficients=frozen['coefficients'],partition='val',test_evaluated=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--stage',choices=('plan','checks','calibrate','diagnostics','run','report'),default='plan')
    parser.add_argument('--physical-batch',type=int,choices=(16,32),help=argparse.SUPPRESS)
    parser.add_argument('--calibration-warmup-updates',type=int,default=20)
    parser.add_argument('--calibration-probes',type=int,default=10)
    parser.add_argument('--benchmark-warmup-updates',type=int,default=10)
    parser.add_argument('--benchmark-measured-updates',type=int,default=30)
    parser.add_argument('--verified-data-root',type=Path)
    args=parser.parse_args(argv)
    if min(args.calibration_warmup_updates,args.benchmark_warmup_updates)<1 or min(args.calibration_probes,args.benchmark_measured_updates)<2:
        parser.error('Invalid short-check/calibration update counts')
    if args.stage=='report':
        from src.sem_cellstyle.regularization_report import write_report
        print(write_report(args.output.resolve())); return
    base=configuration(args.config); arm_templates()
    if args.stage=='plan':
        configured = (bool(base.dataset.get('inventory')) if base.dataset.provider=='verified_nffa_inventory_v1'
                      else bool(base.dataset.manifest and base.dataset.receipt))
        print(json.dumps(dict(arms={key:dict(l2_sp=value[0],feature_distillation=value[1]) for key,value in ARMS.items()},
            dataset_provider=base.dataset.provider,approved_manifest_configured=configured,
            classification_splits=list(base.evaluation.get('classification_splits',['A','B'])),
            head_profile=base.evaluation.get('head_profile','legacy'),
            allow_cuda_resume_update_drift=base.runtime.get('allow_cuda_resume_update_drift', False),
            original_large_weights_configured=bool(base.initialization.checkpoint and base.initialization.sha256),
            coefficients='unresolved_until_shared_training_only_gradient_calibration',effective_batch=128,
            physical_batch=base.runtime.physical_batch,full_schedule_passes=100,initial_stop_passes=10,
            gpu_checked=False,training_started=False),indent=2)); return
    from src.sem_cellstyle.data import provider
    dataset,collate,data=provider(base); initialization=require_initialization(base)
    from src.sem_cellstyle.runner import require_a100
    require_a100()
    if args.stage=='calibrate':
        chosen=resolve(base,len(dataset),args.physical_batch)
        calibrate(chosen,dataset,collate,data,initialization,args.output/'calibration.json',
                  warmup=args.calibration_warmup_updates,probes=args.calibration_probes)
    elif args.stage=='diagnostics':
        recorded=configuration(args.output/'configs/A.yaml')
        actual=resolve(base,len(dataset),recorded.runtime.physical_batch)
        if common_identity(actual,data,initialization)!=read(args.output/'calibration.json')['common_identity']:
            raise ValueError('Diagnostic dataset/initialization/recipe differs from the frozen study')
        from src.sem_cellstyle.regularization_report import feature_diagnostics
        feature_diagnostics(args.output.resolve(),base,dataset,collate,initialization)
    else:
        from scripts.run_sem_ssl_ablations import run_command
        run_study(args,base,dataset,collate,data,initialization,run_command)


if __name__=='__main__':
    main()

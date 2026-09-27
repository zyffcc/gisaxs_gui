"""Create an immutable small, explicit 1D development snapshot; never promote GUI weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'modules/Fitting_1D_Model/Workflow_v5/development/multishape_progress_20260922'
CODE_DIRECTORIES=(
    'multishape_v1','multishape_v2','multishape_online','multishape_loss_control',
    'multishape_global_context','multishape_candidate_search','multishape_real_stress',
    'multishape_gradient_diagnosis','multishape_observability','multishape_amplitude_prior_diagnosis',
)
REPORT_DIRECTORIES=(
    'multishape_v2/results_gpu/report','multishape_online/results/report',
    'multishape_loss_control/results/report','multishape_global_context/results/report',
    'multishape_candidate_search/results','multishape_candidate_search/results_conditioning/report',
    'multishape_real_stress/results_r3/report','multishape_real_stress/noise_control_results/report',
    'multishape_real_stress/results_density/report','multishape_real_stress/input_shift_description',
    'multishape_gradient_diagnosis','multishape_observability/results','multishape_amplitude_prior_diagnosis',
)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def collect():
    paths={ROOT/'tools/package_multishape_development.py'}
    allowed_npz=set()
    if (ROOT/'research/__init__.py').exists():paths.add(ROOT/'research/__init__.py')
    for name in CODE_DIRECTORIES:
        folder=ROOT/'research'/name
        if not folder.exists():raise FileNotFoundError(folder)
        paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.sh'))
        for sub in ('reports','diagnostics'):
            child=folder/sub
            if child.exists():
                paths.update(p for p in child.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json'))
    for name in REPORT_DIRECTORIES:
        folder=ROOT/'research'/name
        if folder.exists():
            paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.md','.json','.png'))
    models=('multishape_online/results/models/point_stream',
            'multishape_v2/results_gpu/models/point_profile','multishape_v2/results_gpu/models/shape_profile',
            'multishape_global_context/results/models/baseline','multishape_global_context/results/models/global')
    for name in models:
        folder=ROOT/'research'/name
        if folder.exists():
            for file in ('curve_best.npz','model_config.json','protocol.json','history.json','status.json'):
                path=folder/file
                if not path.exists():raise FileNotFoundError(path)
                paths.add(path)
                if path.suffix=='.npz':allowed_npz.add(path)
    for relative in ('multishape_v1/results/data/dev.npz','multishape_real_stress/inputs.npz'):
        paths.add(ROOT/'research'/relative)
        allowed_npz.add(ROOT/'research'/relative)
    exact_archives={
        'multishape_real_stress/results_r3':('predictions.npz','RESULTS.json','PROTOCOL.json','STATUS.json'),
        'multishape_real_stress/noise_control_results':('RESULTS.json','PROTOCOL.json','STATUS.json'),
        'multishape_real_stress/results_density':('RESULTS.json','PROTOCOL.json','STATUS.json'),
        'multishape_online/results/verification/candidate_search/point_stream_r3':('candidates.npz','SUMMARY.json','CASES.json'),
    }
    for folder,names in exact_archives.items():
        if (ROOT/'research'/folder).exists():
            for name in names:
                path=ROOT/'research'/folder/name
                if not path.exists():raise FileNotFoundError('Incomplete result archive: '+str(path))
                paths.add(path)
                if path.suffix=='.npz':allowed_npz.add(path)
    for relative in ('SOURCES.json','MANIFEST.json','README_zh.md'):
        p=ROOT/'research/public_scattering_20260922'/relative
        if p.exists():paths.add(p)
    # No workspace/session/SSH files, large datasets, old TEST, or sealed external PS.
    for p in paths:
        if not p.is_file() or ROOT not in p.resolve().parents:raise ValueError(p)
        if p.suffix=='.npz' and p not in allowed_npz:raise ValueError(p)
        if any(part.lower() in ('external_holdout','remote','__pycache__') for part in p.parts):raise ValueError(p)
    return sorted(paths)

def run(args):
    paths=collect();total=sum(p.stat().st_size for p in paths)
    if total>80_000_000:raise ValueError('Unexpected snapshot size; inspect explicit whitelist')
    if args.dry_run:
        print(json.dumps(dict(files=len(paths),bytes=total,output=str(args.out)),indent=2));return
    for arm in ('baseline','global'):
        status=ROOT/'research/multishape_global_context/results/models'/arm/'status.json'
        if json.loads(status.read_text())['status']!='complete':raise ValueError('r5 training incomplete')
    required_reports=['multishape_global_context/results/report/REPORT_zh.md']
    for folder in ('results_r3','noise_control_results','results_density'):
        parent=ROOT/'research/multishape_real_stress'/folder
        if json.loads((parent/'STATUS.json').read_text())['status']!='complete':raise ValueError('Real diagnostic incomplete')
        required_reports.append('multishape_real_stress/'+folder+'/report/REPORT_zh.md')
    for name in required_reports:
        if not (ROOT/'research'/name).is_file():raise FileNotFoundError('Final report missing: '+name)
    before={p:dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    if args.out.exists():raise FileExistsError('Snapshot is immutable; use a new name')
    args.out.mkdir(parents=True)
    manifest={}
    for source in paths:
        relative=source.relative_to(ROOT);target=args.out/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        digest=before[source]['sha256']
        if sha(target)!=digest:raise ValueError('Copy mismatch')
        manifest[relative.as_posix()]=before[source]
    if collect()!=paths or any(sha(p)!=before[p]['sha256'] or p.stat().st_size!=before[p]['bytes'] for p in paths):
        raise RuntimeError('Source set changed during snapshot; this directory is incomplete and has no final manifest')
    status='''# 1D多形状模型开发快照（实验性，未替换GUI默认模型）

此目录保存可迁移的科研源码、候选权重、固定DEV观测、真实压力测试输入、关键报告和校验清单。它不是稳定发布模型。读取各实验报告，分别核对真实组分覆盖、同一候选正演质量和多参数候选；不能把幅度校正后的好曲线当作正确组分证明。

r3为当前已归档点模型。r4在固定预算下未超过起始模型，保留step0。全局context对照若已完成，其模型及报告也随包保存；以对应status/report为准，不能把结构烟测当质量提升。公开GALAXI仅为开发压力数据，无参数真值；未纳入封存PS数据或旧TEST。

从此目录作为工作目录，安装Python3.10、TensorFlow2.15.1、NumPy1.26、SciPy1.15、Matplotlib、threadpoolctl及h5py。可使用research.multishape_v2.new_model与weight_io读取r3模型，输入接口为feature_controls.prepare_inputs。数值权重文件不含可执行pickle；模型与科学源码hash必须匹配。新global模型使用其独立model和weight_io。r3参数域：球/随机圆柱/竖直圆柱，最多4组分，可重复；R1–10nm、h3–60nm；旧intensity-weighted Gaussian、各粒子SF开启、加性resolution。它不是数目分布或TEM粒径后验。

继续训练使用online generator按seed/sample index再生TRAIN，不需要保存大训练集。指定model-dir/data-dir/out为本机相对路径；历史protocol里的绝对路径仅是来源记录，不应直接执行旧Maxwell worker.sh。不要重适配冻结scaler。此包只包含DEV，不包含独立TEST；DEV已反复用于开发，下一次正式验收必须冻结版本后生成新的独立数据。

真实inputs.npz保存q/I/sigma/mask与来源metadata。CBF计数误差近似和公开图的工作容差不同；旧stress全局峰值底噪已发现会掩盖弱尾段，不能据工作RMS小给出通过认证。参考noise/density对照报告（若存在）。全部拟合使用完整原生观测，显示点和编码点数不可冒充独立观测。

几何和detector预处理属于GUI项目，继续处理CBF应迁移整个GUI仓库。固定的预提取真实输入可以在本目录重放；从raw图重新提取还需要原图与GUI对应生产模块。MANIFEST.json给出本包每个来源文件的相对路径/字节/hash；不含SSH凭据、会话文件或私人目录配置。

原始真实stress保留完整归档，以便noise对照检查锁定基线。noise/density报告、结果摘要、精确输入和源码保留，其数十MB重复候选曲线不随包复制，需在新输出目录重新运行对照再做逐数组审计。大型DEV逐候选曲线归档也不全部复制；相应报告仍保留，重新生成报告的逐候选审计前须先用保存的模型运行对应evaluate脚本。原始来源hash与重新生成归档的字节hash不必相同，先比较数值与协议，不改历史报告冒充原输出。
'''
    (args.out/'README_zh.md').write_text(status,encoding='utf-8')
    manifest['README_zh.md']=dict(sha256=sha(args.out/'README_zh.md'),bytes=(args.out/'README_zh.md').stat().st_size)
    (args.out/'MANIFEST.json').write_text(json.dumps(dict(status='development_only_not_stable',
        copied_files=manifest,bytes=sum(v['bytes'] for v in manifest.values()),test_included=False,
        gui_model_promoted=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(manifest),bytes=total,output=str(args.out)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--dry-run',action='store_true')
    run(p.parse_args())

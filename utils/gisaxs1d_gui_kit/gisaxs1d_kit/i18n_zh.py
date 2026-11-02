"""Chinese texts of the panel, matched by the exact English source string (the convention of GIMaP's i18n/zh.py)."""

ZH = {
    'GISAXS / SAXS 1D fit': 'GISAXS / SAXS 一维拟合',
    'Open curve…': '打开曲线…',
    'No curve loaded': '尚未载入曲线',
    'Curve': '曲线',
    'q unit of the file': '文件的 q 单位',
    'Model': '模型',
    'Composition': '组分',
    'Mode': '模式',
    'qz (nm⁻¹)': 'qz (nm⁻¹)',
    'Policy': '策略',
    'Network': '网络',
    'Error floor (%)': '系统误差下限 (%)',
    'q range': 'q 范围',
    'min': '最小',
    'max': '最大',
    'Preview': '预览',
    'Fit': '拟合',
    'Cancel': '取消',
    'Save result…': '保存结果…',
    'Show components': '显示各分量',
    'Parameter': '参数',
    'Value': '数值',
    'Unit': '单位',
    'Range of equivalent solutions': '等价解的范围',
    'Status': '状态',
    'Group': '所属',
    'Ready': '就绪',
    'Starting the fitting process…': '正在启动拟合进程…',
    'Loading…': '正在读取…',
    'Fitting…': '正在拟合…',
    'Previewing…': '正在预览…',
    'Cancelled': '已取消',
    'Failed': '失败',
    'Done': '完成',
    'Intensity': '强度',
    'Residual / σ': '残差 / σ',
    'data': '数据',
    'model': '模型',
    'ok': '正常',
    'ambiguous': '多解',
    'weak': '弱约束',
    'at_bound': '触边界',
    'fixed': '固定',
    'inactive': '无贡献',
    'derived': '',
    'fast: about 0.3 s': 'fast：约 0.3 秒',
    'auto: multi-start search when needed (1-8 s)': 'auto：需要时多起点搜索（1–8 秒）',
    'robust: always multi-start (3-8 s)': 'robust：总是多起点（3–8 秒）',
    'GISAXS (structure factor + low-q term)': 'GISAXS（结构因子 + 低 q 项）',
    'SAXS (dilute, no low-q term)': 'SAXS（稀薄，无低 q 项）',
    'Out-of-plane momentum of the cut; spheres are evaluated at |q| = sqrt(q∥² + qz²). Vertical cylinders do not use it.':
        '线切所在的面外动量；球按 |q| = sqrt(q∥² + qz²) 计算。竖直圆柱不使用它。',
    'Shapes: sphere, vc (vertical cylinder), rc (random cylinder), 1-4 populations, e.g. sphere,vc':
        '形状：sphere（球）、vc（竖直圆柱）、rc（随机取向圆柱），1–4 个群，例如 sphere,vc',
    'Relative systematic error added to the error bars; 2-5 % is sensible for bright measured curves.':
        '按平方和加到误差棒里的相对系统误差；亮的实测曲线建议 2–5%。',
    'The model does not describe the curve within the error bars (quality flag).': '模型没有在误差范围内描述这条曲线（quality 标记）。',
    'The curve is unlike the curves the tool was tested on (out-of-distribution flag).': '这条曲线与工具测试过的曲线差别很大（ood 标记）。',
    'No error bars were given: Poisson errors were assumed.': '没有给误差棒：按泊松误差处理。',
    'Several compositions fit equally well; the curve alone does not decide between them.': '有几种组分拟合得同样好，单靠这条曲线无法区分。',
    'Another solution fits equally well: ': '存在同样好的另一个解：',
    'Not pinned down by this curve: ': '这条曲线定不住：',
    'Populations without contribution: ': '没有贡献的群：',
    'chi² per point': 'χ²/点',
    'intensity share': '强度占比',
    'background B': '背景 B',
    'low-q amplitude C': '低 q 项幅度 C',
    'low-q width': '低 q 项宽度',
    'low-q exponent nu': '低 q 项指数 ν',
    'low-q power n': '低 q 幂指数 n',
    'points': '点',
    'Curves (*.npz *.csv *.txt *.dat);;All files (*)': '曲线文件 (*.npz *.csv *.txt *.dat);;所有文件 (*)',
    'Result (*.json)': '结果 (*.json)',
    'Saved': '已保存',
}


def translate(text, language):
    if language != 'zh':
        return text
    if text in ZH:
        return ZH[text]
    for source, target in ZH.items():                       # sentences that end with a list of parameter names
        if source.endswith(': ') and text.startswith(source):
            return target + text[len(source):].rstrip('.') + '。'
    return text

"""Analyze menu and progress messages for replaying a reduction or fit record."""

REPLAY_ERROR_TEMPLATES = {
    "Unsupported reproduction schema: {schema}": "不支持的复现记录格式版本：{schema}",
    "Unknown reproduction kind: {kind}": "未知的复现记录类型：{kind}",
    "Invalid reproduction record {name}: {error}": "复现记录 {name} 无效：{error}",
    "Invalid recorded mask bitmap: {error}": "记录中的掩膜位图无效：{error}",
    "No input file digest recorded for {path}.": "记录中缺少输入文件 {path} 的摘要。",
    "The detector reader used an input outside the record: {path}": "读取探测器数据时用到了记录之外的文件：{path}",
    "The reduction did not produce the recorded curve {key}.": "归约未生成记录中的曲线 {key}。",
    "The network record changed after loading: {path}": "网络记录在载入后发生了变化：{path}",
    "Expected the network record beside its weights: {path}": "网络记录应与权重文件放在一起：{path}",
    "Unknown fit operation {kind}": "未知的拟合操作 {kind}",
    "Missing referenced file: {path}": "找不到记录引用的文件：{path}",
    "The record has no SHA-256 for {path}": "记录中缺少 {path} 的 SHA-256。",
    "SHA-256 mismatch for {path}: expected {expected}, got {actual}":
        "文件 {path} 的 SHA-256 不符。记录值为 {expected}，当前值为 {actual}。",
}

REPRODUCTION_ZH = {
    **REPLAY_ERROR_TEMPLATES,
    ("The Series map was restored. Its reduction records exceeded the project size limit. "
     "Build Map again to reproduce it."):
        "已恢复序列热图。其归约记录超出了项目的大小限制。若要复现，请再点“生成热图”。",
    "Tools": "工具",
    "Tools for working with analysis records": "处理分析记录的工具",
    "&Reproduce from Record…": "从记录复现(&R)…",
    "Repeat a recorded reduction or fit and compare the numerical results": "按记录重新归约或拟合，并比较数值结果",
    "Reproduce from Record": "从记录复现",
    "Choose a Folder for Reproduced Results": "选择存放复现结果的文件夹",
    "Reproducing {name}": "正在复现 {name}",
    "Stopping after the current step…": "将在当前步骤完成后停止…",
    "Reproduction cancelled.": "已取消复现。",
    "The reproduced values are bitwise identical.": "复现值逐位相同。",
    "The reproduced values differ. See comparison.json for details.": "复现值有差异。详见 comparison.json。",
    "The recorded mask shape differs from the detector frame.": "记录中掩膜的形状与探测器帧不符。",
    "Unrecognised mask bitmap encoding.": "无法识别掩膜位图的编码。",
    "Invalid run lengths.": "掩膜的游程长度无效。",
    "SHA-256 mismatch for the recorded mask bitmap.": "记录中的掩膜位图与其 SHA-256 不符。",
    "The reproduced detector validity differs from the recorded final mask.": "复现得到的有效像素与记录中的最终掩膜不符。",
    "A reproduction column must be a one-dimensional numeric array.": "复现结果的每列必须是一维数值数组。",
    "Invalid reference column type.": "参考数据列的类型无效。",
    "Invalid reference column length.": "参考数据列的长度无效。",
    "Predict did not reach the partial solution recorded when the fit stopped.":
        "参数预测未得到拟合停止时记录的中间解。",
    "This record has no reproduction recipe. Export the result again with this version of GIMaP.":
        "此记录缺少复现所需的输入和设置。请用本版本 GIMaP 重新导出结果。",
    "The batch record contains no results to reproduce.": "这份批量记录中没有可供复现的结果。",
    "The record has no reference columns for comparison.": "记录中没有用于比较的参考数据列。",
}

"""Chinese texts of Fitting's data and results: CSV curves and Choose Columns, Save ▸ Solutions…, the ± of a Predict
solution, the stages of Predict parameters and what Stop keeps, where refined values come from, projects that keep
the fits, Save Plot with residuals, and Undo / Redo that say what they did (``fitting/presentation/single/``)."""

FIT_DATA_ZH = {
    # CSV curves and Choose Columns (application/models.py, curve_columns_dialog.py, views/curve_columns_view.py)
    "The header of this table does not say which columns are q, I and σ: choose them.":
        "这张表的表头没有说明哪些列是 q、I 和 σ：请选择。",
    "{name} was not opened: its columns were not chosen.": "{name} 没有打开：没有选择它的列。",
    "Choose Columns": "选择列",
    "Unit of q": "q 的单位",
    "none (no σ)": "无（没有 σ）",
    "q, I and σ must be three different columns.": "q、I 和 σ 必须是三个不同的列。",
    "Stated in the name of the q column.": "q 列的名称已写明。",
    # Save ▸ Solutions… (views/fit_page_view.py, views/fit_results_view.py, single/solutions_export.py)
    "Solutions…": "解…",
    "Export Solutions…": "导出解…",
    "Save Solutions": "保存解",
    "Every solution of Predict parameters: probability, χ²ᵣ, flags and every value with its intervals (CSV and JSON), "
    "and their curves":
        "预测参数的每个解：概率、χ²ᵣ、标记和每个数值及其区间（CSV 和 JSON），以及它们的曲线",
    "Every solution: probability, χ²ᵣ, flags and every value with its median and 68 % / 95 % intervals (CSV and JSON), "
    "and the curves of every solution with their terms":
        "每个解：概率、χ²ᵣ、标记和每个数值及其中位数与 68 % / 95 % 区间（CSV 和 JSON），以及每个解的曲线和各项",
    "No solutions to save: run Predict parameters first.": "没有可保存的解：请先运行预测参数。",
    "Saved {name}, its record {record} and the curves {curves}.": "已保存 {name}、它的记录 {record} 和曲线 {curves}。",
    # The ± column and the solutions (single/solution_texts.py, results.py)
    "±: after a fit, the least-squares error; for a solution of Predict parameters in Model, half the width of its "
    "68 % posterior interval.":
        "±：拟合之后是最小二乘误差；模型中是预测参数的解时，是它 68 % 后验区间宽度的一半。",
    "± half the width of the 68 % posterior interval [{low}, {high}] of this Predict solution (not a least-squares "
    "error).":
        "± 为这个预测解 68 % 后验区间 [{low}, {high}] 宽度的一半（不是最小二乘误差）。",
    "± half the width of the 68 % interval [{low}, {high}] from the least-squares errors (this solution is compared "
    "by BIC).":
        "± 为由最小二乘误差得到的 68 % 区间 [{low}, {high}] 宽度的一半（这个解按 BIC 比较）。",
    "No interval: Predict parameters solves the scales and the background for every sample.":
        "没有区间：预测参数对每个样本直接求解标度和背景。",
    "not determined: {name}": "未确定：{name}",
    "at bound: {name}": "在边界：{name}",
    "correlated: {name}": "相关：{name}",
    "few samples": "有效样本少",
    "compared by BIC": "按 BIC 比较",
    "model error": "模型误差",
    "large: {name}": "偏大：{name}",
    "small: {name}": "偏小：{name}",
    "q_z·R high: {name}": "q_z·R 偏高：{name}",
    "population {name} unused": "第 {name} 个粒子群未使用",
    "incomplete": "不完整",
    "{number}: {label} · χ²ᵣ {chi2}": "{number}：{label} · χ²ᵣ {chi2}",
    "{label}: {flags}": "{label}：{flags}",
    "Refined from solution {number} ({method}) · χ²ᵣ {chi2}": "由解 {number} 重新精修（{method}）· χ²ᵣ {chi2}",
    "Values from a new refinement ({method}) of solution {number}, {label}{edited}: not the values Predict gave; those "
    "and their intervals stay with solution {number} below.":
        "数值来自对解 {number}（{label}）的新一次精修（{method}）{edited}：不是预测给出的数值；那些数值及其区间仍在下面的"
        "解 {number} 中。",
    ", edited in Model before the fit": "，拟合前在模型中改过",
    "Incomplete: Predict was stopped; these are the solutions it finished, their probabilities over those alone.":
        "不完整：预测被停止；这些是它已完成的解，概率只在这些解之间计算。",
    "Incomplete: Predict was stopped; its probability is over the solutions finished.":
        "不完整：预测被停止；它的概率只在已完成的解之间计算。",
    # The stages of Predict and Stop (single/predict_progress.py)
    "Network proposals: composition {number} of {total}": "网络建议：组成 {number} / {total}",
    "Refining the modes: composition {number} of {total}": "精修各峰：组成 {number} / {total}",
    "Refining from other starts: composition {number} of {total}": "从其他起点精修：组成 {number} / {total}",
    "Searching the ranges for a composition the proposals miss": "为建议未覆盖的组成在范围内搜索",
    "Sampling the posterior: solution {number} of {total}": "后验采样：解 {number} / {total}",
    "Step {step} of 3 · {stage}": "第 {step} 步，共 3 步 · {stage}",
    "{time} elapsed": "已用 {time}",
    "about {time} left in this step": "这一步约剩 {time}",
    "Stopping … the solutions finished so far are kept.": "正在停止…保留已完成的解。",
    "Stopped: {solutions} kept, of the {finished} of {total} solutions sampled before Stop — marked incomplete; their "
    "probabilities are over those alone.":
        "已停止：保留 {solutions}，来自停止前已采样的 {finished} / {total} 个解，标为不完整；它们的概率只在这些解之间计算。",
    "Stopped before any solution was finished: no new solutions.": "在任何解完成之前停止：没有新的解。",
    # Undo and Redo (single/history.py, session.py labels)
    "Undone: {step}.": "已撤销：{step}。",
    "Redone: {step}.": "已重做：{step}。",
    "Undo: {step} (Ctrl+Z)": "撤销：{step}（Ctrl+Z）",
    "Redo: {step} (Ctrl+Shift+Z)": "重做：{step}（Ctrl+Shift+Z）",
    "the last change": "上一次更改",
    "a change of Model": "模型的更改",
    "an edit in Model": "模型中的编辑",
    "the fitting range": "拟合范围",
    "the left-out points": "排除的点",
    "the fit ({method})": "拟合（{method}）",
    "Predict parameters: solution 1 in Model": "预测参数：解 1 放入模型",
    "solution {number} in Model": "解 {number} 放入模型",
    "the model from {name}": "来自 {name} 的模型",
    "Analyze's solution in Model": "分析的解放入模型",
    # Save Plot (single/figure_export.py, views)
    "The plot with the residuals: PNG, SVG or PDF, with error bars": "曲线图及其残差：PNG、SVG 或 PDF，带误差棒",
    "The plot as shown with the residuals under it: PNG (image), SVG or PDF (vector), with error bars":
        "当前显示的曲线图，下方附残差：PNG（图像）、SVG 或 PDF（矢量），带误差棒",
    "Choose a .png, .svg or .pdf file: {name} is none of them.": "请选择 .png、.svg 或 .pdf 文件：{name} 都不是。",
    "the folder {folder} does not exist": "文件夹 {folder} 不存在",
    # Projects (single/project.py, single/series_project.py)
    "the Fitting curve {name} changed since the project was saved: its fit results are shown as stale":
        "拟合曲线 {name} 在项目保存后改变了：它的拟合结果显示为已过时",
    "the Fitting results could not be read: {reason}": "无法读取拟合结果：{reason}",
    "{count} fitted frames of the project are shown again.": "项目中 {count} 个已拟合的帧已重新显示。",
    "{count} fitted frames of the project are shown again; {missing} are not, their files are no longer listed.":
        "项目中 {count} 个已拟合的帧已重新显示；另有 {missing} 个没有显示，它们的文件已不在列表中。",
    "The fits of the series in the project could not be read: {reason}": "无法读取项目中序列的拟合：{reason}",
}

__all__ = ["FIT_DATA_ZH"]

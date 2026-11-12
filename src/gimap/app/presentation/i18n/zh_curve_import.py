"""Chinese interface texts for explicit single/shared-q curve table imports."""

CURVE_IMPORT_ZH = {
    "An imported curve must name a physical file; keep its column mapping separately.":
        "导入曲线的路径必须指向实际文件；列映射应单独保存。",
    "Batch / In-situ series": "批量 / 原位序列",
    "&Batch / In-situ Series": "批量 / 原位序列(&B)",
    "Every listed curve fitted with the model of Single analysis": "使用单条分析的模型拟合列表中的每一条曲线",
    "&Import Curves…": "导入曲线(&I)…",
    "Batch fit selected TXT/DAT files: preview q I sigma or shared-q columns q I1 I2 …":
        "批量拟合所选 TXT/DAT 文件：预览 q、I、sigma 或共用 q 的 q、I1、I2、… 列",
    "Stop the batch fit before importing curves.": "请先停止批量拟合，再导入曲线。",
    "Import Curves from This File…": "从此文件导入曲线…",
    "Review the columns of this file before batch fitting.": "批量拟合前核对此文件各列的含义。",
    "Import Curves…": "导入曲线…",
    "Import Curves": "导入曲线",
    "{count} imported curves": "{count} 条已导入曲线",
    "Independent samples: each curve starts from the same model; stages and continuity are off.":
        "独立样品：每条曲线都从同一个模型开始拟合，不使用阶段划分和连续性约束。",
    "Ordered sequence: curves follow the selected file and column order.": "有序序列：曲线按所选文件及列的顺序排列。",
    "Imported curves are a fixed selection. Import again to change files or columns; Watch is available for folders.":
        "导入的曲线是固定选择。若需更改文件或列，请重新导入；文件夹支持监视功能。",
    "{count} curves: q in {unit}, no unit in their header (chosen at import)":
        "{count} 条曲线：q 单位为 {unit}，文件头未注明单位（导入时选择）",
    "Choose at least one curve to import.": "请至少选择一条要导入的曲线。",
    "Choose a valid q unit for the imported curves.": "请为导入的曲线选择有效的 q 单位。",
    "The same source column was selected more than once.": "同一个来源列被重复选择。",
    "The imported curves could not be restored: {reason}": "无法恢复已导入的曲线：{reason}",
    "The imported curves could not be restored.": "无法恢复已导入的曲线。",
    "Imported curves": "已导入曲线",
    "{count} imported curves cannot be read: {name} — {reason}": "有 {count} 条已导入曲线无法读取：{name} — {reason}",
    "Choose curve files and their q, intensity and optional σ columns": "选择曲线文件及其 q、强度和可选的 σ 列",
    "Import Curve Tables": "导入曲线表格",
    "Review the columns before importing. The preview reads only the beginning of each file; the complete curves are read when needed.":
        "导入前请核对各列含义。预览只读取每个文件的开头；需要使用时才读取完整曲线。",
    "Column layout": "列布局",
    "Single curve: q, I, optional σ": "单条曲线：q、I、可选 σ",
    "Shared q: q, I₁, I₂, …": "共用 q：q、I₁、I₂、…",
    "Apply layout to all": "将布局应用于所有文件",
    "First 10 data rows · column roles are shown in the headings": "前 10 行数据 · 列标题中标明各列用途",
    "Independent samples (fit each curve)": "独立样品（逐条拟合）",
    "Time series (file and column order)": "时间序列（按文件和列的顺序）",
    "Use curves as": "曲线用途",
    "q unit when absent from the header": "表头未注明时的 q 单位",
    "Files keep their selected order; shared-q curves keep their left-to-right column order. A unit written in a file header takes priority.":
        "文件保持选择时的顺序；共用 q 的曲线保持列从左到右的顺序。文件头注明的单位优先。",
    "Apply this layout to files with identical column headings.": "将此布局应用于列标题完全相同的文件。",
    "Files with different columns must be reviewed individually.": "列不同的文件需要逐个核对。",
    "Could not inspect this file: {error}": "无法预览此文件：{error}",
    "Could not import this table: {error}": "无法导入此表格：{error}",
    "Layout not changed: {error}": "布局未更改：{error}",
    "Waiting for the file preview …": "正在等待文件预览…",
    "Analyze export: q, I, σ and pixel counts describe one curve. Shared-q import is disabled.":
        "分析导出文件：q、I、σ 和像素数描述的是同一条曲线，不能按共用 q 的多曲线导入。",
    "Named uncertainty columns cannot be imported as intensities. Only q, I and σ are used; put curves with separate uncertainties in separate q, I, σ files.":
        "文件头注明的误差列不能作为强度列导入。当前只使用 q、I 和 σ；多条曲线若各有独立误差，请分别放入 q、I、σ 文件。",
    "No unambiguous column roles were found. The default treats column 3 as σ, not another curve. Choose Shared q if the remaining columns are intensities; values are not used to guess.":
        "无法明确判断各列用途。默认把第 3 列作为 σ，而非另一条曲线。如果 q 之后的各列都是强度，请选择“共用 q”；软件不会根据数值猜测。",
    "Only q, I and σ are used; columns after the third are ignored.": "只使用 q、I 和 σ，第 3 列之后的列将被忽略。",
    "Every column after q is a separate intensity curve. No uncertainty column is used.":
        "q 之后的每一列分别作为一条强度曲线，不使用误差列。",
    "Intensity": "强度",
    "Uncertainty (σ)": "不确定度（σ）",
    "Ignored": "忽略",
    "{name}\nRole: {role}": "{name}\n用途：{role}",
    "Column {n}": "第 {n} 列",
    "q unit from this file's header: {unit}": "此文件头中的 q 单位：{unit}",
    "No q unit in this header: the selected fallback unit will be used.": "此文件头未注明 q 单位，将使用下方选择的单位。",
    "Read failed": "读取失败",
    "{count} curves": "{count} 条曲线",
    "1 curve": "1 条曲线",
    "Checking …": "正在检查…",
    "Waiting": "等待中",
    "Checked {done} of {total} files": "已检查 {done}/{total} 个文件",
    "{name}: {error}": "{name}：{error}",
    "{count} files could not be checked. Nothing will be imported; correct the files and try again.":
        "有 {count} 个文件无法检查，当前不会导入任何曲线。请修正文件后重试。",
    "No files selected.": "未选择文件。",
    "Checking files before import …": "正在检查待导入的文件…",
    "Files: {files} · Curves to import: {count}": "文件：{files} 个 · 待导入曲线：{count} 条",
    "Import 1 curve": "导入 1 条曲线",
    "Import {count} curves": "导入 {count} 条曲线",
    "Curve column indexes must be non-negative integers.": "曲线列索引必须为非负整数。",
    "q, intensity and sigma must use different columns.": "q、强度和 sigma 必须使用不同的列。",
    "A selected curve column is missing from the table.": "表格中缺少所选的曲线列。",
    "Curve column names must be a tuple of strings.": "曲线列名必须是一组文本。",
    "A selected curve column has no column name.": "所选曲线列没有列名。",
    "Curve column names do not match the table width.": "曲线列名的数量与表格列数不一致。",
    "Invalid curve column mapping.": "曲线列映射无效。",
    "Invalid curve column names.": "曲线列名无效。",
    "Unknown curve table layout.": "未知的曲线表格布局。",
    "A curve table needs at least two columns.": "曲线表格至少需要两列。",
    "GIMaP Analyze exports contain one curve; sigma and pixels are not intensity columns.":
        "GIMaP 分析导出文件只包含一条曲线；sigma 和像素数不是强度列。",
    "Shared-q tables need at least three columns.": "共用 q 的表格至少需要三列。",
    "Named uncertainty columns cannot be imported as separate intensities.": "文件头注明的误差列不能作为独立强度曲线导入。",
    "Single-curve import requires q, I and optional sigma in that order; the named uncertainty column is elsewhere.":
        "单条曲线导入要求按 q、I、可选 sigma 的顺序排列；文件头注明的误差列不在此位置。",
    "An intensity column cannot be used as sigma. Choose Shared q or correct the column names.":
        "强度列不能作为 sigma。请选择“共用 q”或修正列名。",
    "The table column count changed; import it again to confirm the selected columns.":
        "表格列数已变化，请重新导入并确认所选列。",
    "The table column names changed; import it again to confirm the selected columns.":
        "表格列名已变化，请重新导入并确认所选列。",
    "The table column header changed; import it again to confirm the selected columns.":
        "表格列标题已变化，请重新导入并确认所选列。",
}

__all__ = ["CURVE_IMPORT_ZH"]

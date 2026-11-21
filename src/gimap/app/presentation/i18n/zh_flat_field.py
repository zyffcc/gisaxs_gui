"""Chinese for the flat field (detector response) of Analyze: the row of Mask & Corrections ▸ Corrections
(``analyze/presentation/views/options_panel_view.py``, ``bindings/options.py``), what the page says when one is
loaded or refused (``application/flat_field.py``, ``domain/flat_field.py`` through ``texts.message_text``), the
note of a settings file whose flat field is missing, and the set-up summaries (Batch Export, Undo, the Mask step)."""

FLAT_FIELD_ZH = {
    "No flat field": "无平场",
    "Flat field (detector response): every frame is divided by it, normalised to mean 1; pixels where it is at or "
    "below 0.05 are left out. A detector frame of the same size or a .npy file":
        "平场（探测器响应）：每一帧都除以它，它先归一化到平均值为 1；平场值不大于 0.05 的像素被排除。"
        "可以是同样大小的探测器帧，或 .npy 文件",
    "Stop dividing by the flat field": "不再除以平场",
    "Flat field": "平场",  # (Sputter's label too: one Chinese, one English)
    "flat field": "平场校正",
    "flat field {name}": "平场 {name}",
    "Could not read the flat field: {error}": "无法读取平场：{error}",
    "Flat field not loaded: {error}": "未载入平场：{error}",
    "Flat field {name}: {count} pixels left out": "平场 {name}：排除了 {count} 个像素",
    "Normalised by its mean {mean}; {count} pixels at or below {floor} left out":
        "已用其平均值 {mean} 归一化；排除了 {count} 个不大于 {floor} 的像素",
    "Flat-field file not found, left out: {path}": "找不到平场文件，已略过：{path}",
    "The flat field {name} is {flat_shape}; the frame is {shape}.": "平场 {name} 的大小是 {flat_shape}；这一帧是 {shape}。",
    "The flat field {name} is not one 2D image (it has {dimensions} dimensions).":
        "平场 {name} 不是一幅二维图像（它有 {dimensions} 维）。",
    "The flat field {name} has no pixel above 0.": "平场 {name} 没有大于 0 的像素。",
    "The flat field {name} holds no numbers.": "平场 {name} 里没有数值。",
}

__all__ = ["FLAT_FIELD_ZH"]

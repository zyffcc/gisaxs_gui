"""Chinese for the dark frame, the Normalisation section and the Lorentz factor of Analyze ▸ Mask & Corrections
(``analyze/presentation/views/normalisation_view.py``, ``bindings/normalisation.py``), what the page says of them
(``domain/normalisation.py`` through ``texts.message_text``), the set-up summaries (Batch Export, Undo, the Mask
step), Export Curves… asking for its folder, the Panels menu of the image and the αi of the Results geometry."""

NORMALISATION_ZH = {
    # -- the dark frame
    "No dark frame": "无暗场",
    "Dark frame (detector without beam: dark current, offset), subtracted from every frame before the background. "
    "σ then comes from the scatter of the pixels":
        "暗场（无光束时的探测器读数：暗电流、偏置），在扣背景之前从每一帧中扣除。此后 σ 由像素的离散度给出",
    "Stop subtracting the dark frame": "不再扣除暗场",
    "frame − scale × dark (e.g. the exposure-time ratio)": "帧 − 比例 × 暗场（例如曝光时间之比）",
    "Frame of a NeXus dark file": "NeXus 暗场文件中的帧",
    "Dark scale": "暗场比例",
    "Dark frame": "暗场",
    "dark frame": "暗场扣除",
    "dark {name}": "暗场 {name}",
    "Dark frame not found, left out: {path}": "找不到暗场文件，已略过：{path}",
    # -- the normalisation
    "Normalisation": "归一化",
    "Off by default. After the other corrections every frame is multiplied by A / (t · M · T · d): exposure time t, "
    "monitor M, transmission T, thickness d, absolute-scale factor A. σ is multiplied by the same factor; q and the "
    "shape of the curves do not change.":
        "默认关闭。在其他校正之后，每一帧乘以 A / (t · M · T · d)：曝光时间 t、监视器 M、透射率 T、厚度 d、绝对强度因子 A。"
        "σ 乘以同一个因子；q 和曲线的形状不变。",
    "From the file header": "取自文件头",
    "Typed": "手动输入",
    "Exposure t": "曝光时间 t",
    "Monitor M": "监视器 M",
    "Transmission T": "透射率 T",
    "Thickness d": "厚度 d",
    "Absolute scale A": "绝对强度因子 A",
    "Divide by the exposure time: the one the file header states, or a typed one":
        "除以曝光时间：文件头给出的，或手动输入的",
    "Divide by a monitor (ion chamber, diode, flux or ring current): a value of the file header, or a typed one":
        "除以监视器读数（电离室、二极管、通量或储存环流强）：文件头中的一个值，或手动输入的",
    "The monitor value (any positive number, e.g. 1.2e9)": "监视器读数（任意正数，例如 1.2e9）",
    "Transmission of the sample, above 0 and at most 1": "样品的透射率，大于 0 且不大于 1",
    "Absolute-scale factor, e.g. from a glassy-carbon or water measurement in the same set-up":
        "绝对强度因子，例如由同一装置下玻璃碳或水的测量得到",
    "In the file header: {items}": "文件头中：{items}",
    "The file header gives no exposure time and no monitor value.": "文件头中没有曝光时间，也没有监视器读数。",
    "Header: {key} = {value}": "文件头：{key} = {value}",
    "Header: {key} (not in this frame)": "文件头：{key}（这一帧没有）",
    "file header": "文件头",
    "typed": "手动输入",
    "Multiplied by {factor}: {terms}; intensity in {unit}": "已乘以 {factor}：{terms}；强度单位 {unit}",
    "Normalisation not changed: {error}": "归一化未更改：{error}",
    "normalised": "已归一化",
    "normalisation": "归一化",
    "The header of {name} gives no exposure time: type it or turn the exposure off.":
        "{name} 的文件头没有曝光时间：请手动输入，或关闭曝光时间。",
    "The header of {name} has no monitor value “{key}”: choose another or type it.":
        "{name} 的文件头没有监视器读数“{key}”：请另选一个，或手动输入。",
    "{what} must be a positive number.": "{what}必须是正数。",
    "The exposure time": "曝光时间",
    "The monitor value": "监视器读数",
    "The thickness": "厚度",
    "The absolute-scale factor": "绝对强度因子",
    "The transmission must be above 0 and at most 1.": "透射率必须大于 0 且不大于 1。",
    # -- the Lorentz factor
    "Lorentz factor (fibre texture)": "Lorentz 因子（纤维织构）",
    "Crystallites oriented at random about the surface normal: divide by L = 1/(cos αi cos αf |sin 2θf|) (GIXSGUI). "
    "I(χ) then counts the crystallites per degree of χ; the Hermans factor of the automatic analysis is then not "
    "weighted by sin χ again. Only for crystalline peaks: it also scales diffuse scattering and the I(q) of a ring":
        "晶粒绕表面法线随机取向：除以 L = 1/(cos αi cos αf |sin 2θf|)（GIXSGUI）。此后 I(χ) 是每度 χ 内的晶粒数；"
        "自动分析的 Hermans 取向因子不再乘 sin χ 权重。只适用于晶体衍射峰：它也会改变漫散射和整环的 I(q)",
    # -- Export Curves…, the panels, the αi of the Results
    "The I(χ) is divided by the Lorentz factor of a fibre texture: Hermans f is not weighted by sin χ again.":
        "I(χ) 已除以纤维织构的 Lorentz 因子：Hermans 取向因子 f 不再乘 sin χ 权重。",
    "The radial background under the ring ({background} counts/pixel) was subtracted, scaled like an isotropic ring "
    "(∝ sin χ).": "已扣除环下方的径向背景（{background} counts/pixel），并按各向同性环的形状（∝ sin χ）分配。",
    "Export Curves: choose a folder": "导出曲线：选择文件夹",
    "One CSV per curve (q, I, σ, pixels), a cut region's under its name, and a JSON record of the settings, to a "
    "folder you choose (it starts in the last one).":
        "每条曲线一个 CSV（q、I、σ、像素数；切线区域的曲线以区域名命名）以及一份设置的 JSON 记录，保存到你选择的文件夹"
        "（从上次选择的文件夹开始）。",
    "Panels": "面板",
    "Hide the steps or the curves and results, so the image has the room; show them again here":
        "隐藏步骤栏或曲线与结果栏，把空间留给图像；在这里重新显示",
    "Curves and Results": "曲线与结果",
    "{now}° (the run used {used}°)": "{now}°（这次运行用的是 {used}°）",
}

__all__ = ["NORMALISATION_ZH"]

"""Chinese for Sputter, 2D Prediction and Trainset Build: the shorter on-screen texts and the tooltips
that took over the explanations which used to stand under the controls (2026-11 small-text review)."""

TIPS_LABS_ZH = {
    # -- Sputter: the page --------------------------------------------------------------------------------------
    "Sputter series": "溅射序列",
    "Particle size R and spacing D over time, the metal thickness where it can be stated (island-volume "
    "equivalent; nominal, from a deposition rate), and how far each value can be trusted.":
        "颗粒尺寸 R 和间距 D 随时间的变化，能给出时的金属厚度（岛体积等效；由沉积速率得到的名义厚度），以及每个值有多可信。",
    "The folder of one scan (its CBF frames, or a scan folder with them in one subfolder such as p2m/): its first "
    "frame tells the detector, the exposure and the time.":
        "一个扫描的文件夹（它的 CBF 帧所在的文件夹，或只有一个子文件夹存放帧的扫描文件夹，例如 p2m/）：它的第一帧给出探测器、曝光和时间。",
    "Show details": "显示细节",
    # -- Sputter: where each value comes from ---------------------------------------------------------------------
    "not in the files": "文件中没有",
    "estimated from the frames": "由帧估计",
    "⚠ header says {energy} keV": "⚠ 文件头：{energy} keV",
    "⚠ NeXus file says {energy} keV": "⚠ NeXus 文件：{energy} keV",
    "⚠ flat-field file name: E = {energy} keV": "⚠ 平场文件名：E = {energy} keV",
    "Not in the frame headers or the scan's files: enter it.": "帧的文件头和扫描的文件里都没有：请填写。",
    "Empty: the beam centre is estimated from the frames. A centre you give is in Analyze's pixel coordinates.":
        "留空：光束中心由帧估计。你给出的光束中心使用分析页的像素坐标。",
    "The value used last, for another folder: check that it holds for this scan.":
        "上次（为另一个文件夹）使用的值：请核对它是否适用于这个扫描。",
    "Not in the files: 0 is used. Enter the sample roll if it was not 0.":
        "文件中没有：使用 0。如果样品滚转角不是 0，请填写。",
    "The frame header gives {energy} keV, which differs from the value used: check which is the beam's energy.":
        "帧的文件头给出 {energy} keV，与所用的值不同：请核对哪一个是光束的能量。",
    "The scan's NeXus file gives {energy} keV, which differs from the value used: check which is the beam's energy.":
        "该扫描的 NeXus 文件给出 {energy} keV，与所用的值不同：请核对哪一个是光束的能量。",
    "The E in the flat-field file name is the detector's calibration energy, not necessarily the beam's: it is "
    "never filled in. Enter the beam energy.":
        "平场文件名里的 E 是探测器的标定能量，不一定是光束的能量：它从不填入。请填写光束的能量。",
    "S/N {serial}": "序列号 {serial}",
    "{pixel} µm pixels": "像素 {pixel} µm",
    "first frame {first_frame}": "第一帧 {first_frame}",
    "flat field {flat_field}": "平场 {flat_field}",
    # -- Sputter: the model's environment ---------------------------------------------------------------------------
    "PyTorch {version}: ready": "PyTorch {version}：已就绪",
    "⚠ PyTorch is installed on the first Predict ({size}, needs the internet)":
        "⚠ 第一次预测时安装 PyTorch（{size}，需要联网）",
    "PyTorch is checked when you press Predict": "按“预测”时检查 PyTorch",
    "Model: bundled with GIMaP · Python: {python}": "模型：随 GIMaP 提供 · Python：{python}",
    # -- 2D Prediction ----------------------------------------------------------------------------------------------
    "A CBF detector frame (.cbf), or the first of a numbered series, to predict from: prediction reads CBF frames "
    "only.": "用于预测的 CBF 探测器帧（.cbf），或编号序列的第一帧：预测只读取 CBF 帧。",
    "A folder of numbered CBF frames (.cbf): those in the range below are used.":
        "带编号的 CBF 帧（.cbf）所在的文件夹：使用下面范围内的帧。",
    "How many consecutive detector files contribute to this prediction.": "叠加数决定有多少个连续的探测器文件参与这次预测。",
    "Model: Canceled": "模型：已取消",
    # -- Trainset Build -------------------------------------------------------------------------------------------
    "Design a simulated GISAXS dataset, validate it locally, then prepare training jobs.":
        "设计模拟 GISAXS 数据集，在本地验证，然后准备训练任务。",
    "Remembered settings restored": "已还原记住的设置",
    "A measured detector image, to place the beam centre, ROI, thresholds and fixed masks on. Guidance only: "
    "training images and previews are simulated by BornAgain.":
        "一张测量的探测器图像，用来确定光束中心、ROI、阈值和固定掩膜。只作几何参考：训练图像和预览都由 BornAgain 模拟。",
    "Click the direct-beam position on the image on the right. Moving over the full detector shows x, y and "
    "intensity below this button.":
        "在右边的图上点击直射束的位置。鼠标在“整个探测器”图上移动时，这个按钮下面显示 x、y 和强度。",
    "Drag a rectangle on the image: the region the training images keep. A rectangle keeps the tensor shape and "
    "the angular limits explicit.":
        "在图上拖一个矩形：训练图像保留的区域。矩形让张量形状和角度范围保持明确。",
    "⚠ BornAgain not installed locally · reference preview only": "⚠ 本地未安装 BornAgain · 只能预览参考图",
    "The sampled physics, background or noise range to compare: BornAgain simulates its minimum, midpoint and "
    "maximum. The experimental image is never used as training data.":
        "要比较的采样物理量、背景或噪声范围：BornAgain 模拟它的最小值、中点和最大值。实验图像从不用作训练数据。",
    "The feature extractor, one layer per row. The regression output, sized by the variable physics parameters, "
    "is added automatically.": "特征提取器，每行一层。回归输出按可变物理参数的个数自动加上。",
    "Use the precomputed matrix": "使用预先算好的矩阵",
    "Use the precomputed particle-parameter matrix during dataset generation.": "生成数据集时使用预先算好的颗粒参数矩阵。",
    "Where generation writes the HDF5 shards. Empty: <output>/<project name>/dataset.":
        "生成的 HDF5 分片写到这里。留空：<output>/<project name>/dataset。",
    "Where training writes its results. Empty: <output>/<project name>/results.":
        "训练结果写到这里。留空：<output>/<project name>/results。",
    "Orientation: detector x increases to the right and detector y downward; exit angle/qz is higher at the top. "
    "BornAgain is flipped vertically once to match the experimental detector display.":
        "方向：探测器 x 向右增大，y 向下增大；出射角 / qz 越往上越大。BornAgain 的图像上下翻转一次，与实验探测器的显示一致。",
    "Matrix: no particle-parameter axes yet": "矩阵：还没有颗粒参数轴",
    "Matrix: {shape} = {nodes:,} BornAgain basis images ({names}) · float16 cache ≈ {gib:.2f} GiB per file":
        "矩阵：{shape} = {nodes:,} 张 BornAgain 基图像（{names}）· float16 缓存约每个文件 {gib:.2f} GiB",
}

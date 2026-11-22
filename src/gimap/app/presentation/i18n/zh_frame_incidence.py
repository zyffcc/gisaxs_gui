"""Incidence measured from the detector frame in Analyze."""

FRAME_INCIDENCE_ZH = {
    "Nominal αi {nominal}° and measured αi {value} ± {error}° differ by more than 3σ. Used {used}°.":
        "参考 αi {nominal}° 与测量值 {value} ± {error}° 相差超过 3σ。本次使用 {used}°。",
    "Sample roll ψ and incidence αi": "样品滚转角 ψ 与入射角 αi",
    "Measure sample roll and incidence angle from this frame.": "从当前帧测量样品滚转角和入射角。",
    "Keep entered αi": "保留输入的 αi",
    "Use frame-header αi": "使用帧头的 αi",
    "Frame header αi  {value}°": "帧头 αi  {value}°",
    "Entered αi  {value}° · frame header  {header}°": "输入的 αi  {value}° · 帧头  {header}°",
    "Yoneda critical angle": "Yoneda 临界角",
    "Enter angle": "输入角度",
    "Plate glass": "平板玻璃",
    "Choose the material that produces the Yoneda band.": "选择产生 Yoneda 带的材料。",
    "{angle}° at {wavelength} Å": "{angle}°（波长 {wavelength} Å）",
    "Outside the optical table": "超出光学常数表范围",
    "Unknown Yoneda material.": "未知的 Yoneda 材料。",
    "The Yoneda critical angle must be between 0 and 90 degrees.": "Yoneda 临界角必须在 0° 到 90° 之间。",
    "Invalid incidence measurement.": "无效的入射角测量记录。",
    "From film": "取自薄膜",
    "Critical angle of the film or substrate producing the band. From film uses the refraction setting when available.":
        "产生 Yoneda 带的薄膜或衬底的临界角。选择“取自薄膜”时，采用已设置的薄膜折射临界角。",
    "Keep the entered, header or profile angle even when the frame measurement disagrees.":
        "即使测量结果有分歧，也保留输入的入射角、帧头或仪器配置中的入射角。",
    "Measuring ψ and αi…": "正在测量 ψ 和 αi…",
    "The frame or settings changed during the angle measurement.": "测量角度期间，帧或设置发生了变化。",
    "Could not measure ψ and αi. {error}": "未能测量 ψ 和 αi。{error}",
    "Measured αi  {value} ± {error}°": "实测 αi  {value} ± {error}°",
    "Shadow edge": "阴影边缘",
    "Specular spot": "镜面反射斑点",
    "Yoneda band": "Yoneda 带",
    "Needs a material αc": "需要材料临界角 αc",
    "Incidence estimates disagree.": "各方法测得的入射角不一致。",
    "No incidence angle could be measured.": "未能测得入射角。",
    "Using measured αi. The reference ({value}°) differs by more than 3σ.":
        "使用实测 αi。参考值（{value}°）与实测值相差超过 3σ。",
    "Using reference αi ({value}°), within 3σ of the measurement.":
        "使用参考 αi（{value}°），与实测值的差在 3σ 以内。",
    "Using the entered αi.": "使用输入的 αi。",
    "Keeping αi at {value}° by request. It differs from the measurement by more than 3σ.":
        "按用户选择将 αi 保持为 {value}°。该值与实测值相差超过 3σ。",
}

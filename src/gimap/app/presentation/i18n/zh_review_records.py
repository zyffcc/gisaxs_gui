"""Chinese translations: first-user review fixes (records).

αi of the last session that belongs to another instrument profile (``analyze/presentation/bindings/incidence.py``)
and the sentence the automatic analysis adds to its αi question (``assistant/application/pipeline_geometry.py``,
shown through ``guided_words.words``)."""

REVIEW_RECORDS_ZH: dict[str, str] = {
    "αi {value}° of your last session is not used: it was set with another instrument profile "
    "(this profile: {profile}°)":
        "未使用上次会话的 αi {value}°：它是配合另一个仪器配置设定的（当前配置：{profile}°）",
    "αi {value}° of your last session is not used: this detector has no instrument profile yet":
        "未使用上次会话的 αi {value}°：这个探测器还没有仪器配置",
    "Use {value}°": "使用 {value}°",
    "Your last session used αi = {value}° with another instrument profile; it is not used here, give it again if "
    "it holds for this sample.":
        "上次会话配合另一个仪器配置使用了 αi = {value}°；这里没有采用，如果它也适用于这个样品，请重新填写。",
}

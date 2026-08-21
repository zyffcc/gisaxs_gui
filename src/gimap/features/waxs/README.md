# WAXS feature

WAXS/GIWAXS 按 `presentation → application → domain` 分层；嵌入式页面由本 feature
拥有，文件、batch、export、path 与 JobRunner 实现位于 infrastructure。顶层
`ui.waxs_page` 和 `WAXS/WAXS.py` 只提供 public import/standalone launcher 兼容。

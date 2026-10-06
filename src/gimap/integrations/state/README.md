# State integrations

这里实现 application state ports：用户数据目录中的 UserStore（`user_store.py`：settings.json、session.json、
instrument_profiles.json、model_parameters.json；缺失时从旧的 `config/*.json` 复制），settings / preferences /
session / instrument profile 仓库（JSON 与内存两种），以及默认设置。模块不创建全局实例。

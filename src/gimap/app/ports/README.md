# App ports

这里定义应用级状态边界。`SettingsRepository` 管理需要跨启动保留的用户设置；
`SessionRepository` 管理项目和 feature 的会话快照。具体 JSON 或 legacy
`global_params` 实现位于 integrations。

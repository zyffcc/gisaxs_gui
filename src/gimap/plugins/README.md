# `plugins`

`plugins` 负责稳定的插件 contracts、发现机制和生命周期管理。插件通过公开 application
API 或明确 ports 接入，不得导入 feature 的 presentation、controller、ViewModel 或
内部实现。

本目录不是任意扩展代码的垃圾桶。具体 scientific behavior 和 UI 仍应归属于明确的
feature；外部 runtime 的具体实现归属于 infrastructure/integrations。

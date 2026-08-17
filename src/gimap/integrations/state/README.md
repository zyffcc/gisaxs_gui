# State integrations

这里实现 application state ports：legacy `global_params` adapter、保持旧 JSON 结构的
settings repository，以及内存/JSON session repositories。模块不创建全局实例。

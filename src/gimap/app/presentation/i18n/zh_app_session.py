"""Chinese for the start and the session of the application: splash screen, one GIMaP per user folder, unsaved
changes, autosave and recovery, the log folder, Recent and the command line (``app/launcher.py``,
``app/window.py``, ``app/document_session.py``, ``presentation/app_dialogs.py``, ``presentation/splash.py``)."""

APP_SESSION_ZH = {
    # -- splash screen ------------------------------------------------------------------------------------
    "Starting GIMaP…": "正在启动 GIMaP…",
    "Loading the analysis modules…": "正在载入分析模块…",
    "Reading the settings…": "正在读取设置…",
    "Building Analyze…": "正在构建分析页面…",
    "Building Fitting…": "正在构建拟合页面…",
    "Building the Labs…": "正在构建实验室页面…",
    "Building the automatic analysis…": "正在构建自动分析…",
    "Opening the window…": "正在打开窗口…",
    "Opening Compare…": "正在打开比较页面…",
    "Opening Sputter…": "正在打开溅射页面…",
    # -- one GIMaP per user folder ------------------------------------------------------------------------
    "GIMaP is already running.": "GIMaP 已经在运行。",
    "Switch to It": "切换到它",
    "Start Anyway (Read-only Settings)": "仍然启动（设置只读）",
    "Two windows on one user folder would overwrite each other's settings and session. A second one started "
    "anyway keeps its settings in memory only.":
        "两个窗口共用一个用户文件夹，会互相覆盖设置和会话。仍然启动的第二个窗口只在内存里保留它的设置。",
    "(read-only settings)": "（设置只读）",
    # -- unsaved changes, autosave, recovery --------------------------------------------------------------
    "Unsaved Changes": "未保存的更改",
    "Save the changes to the project {name}?": "保存对项目 {name} 的更改吗？",
    "Save the current work as a project?": "把当前的工作保存为项目吗？",
    "Changes that are not saved are lost.": "未保存的更改会丢失。",
    "Don't Save": "不保存",
    "Restore the Session": "恢复会话",
    "GIMaP was not closed properly. Restore the work autosaved at {time}?":
        "GIMaP 上次没有正常关闭。恢复 {time} 自动保存的工作吗？",
    "Restore": "恢复",
    "an unknown time": "未知时间",
    "Restored the autosaved session. Save it to keep it.": "已恢复自动保存的会话。要保留它，请保存。",
    "The autosaved session could not be read: {error}": "无法读取自动保存的会话：{error}",
    # -- Recent, the command line ---------------------------------------------------------------------------
    "{name} could not be opened and was removed from Recent.": "无法打开 {name}，已将它从最近使用中移除。",
    "It was removed from Recent.": "已将它从最近使用中移除。",
    "Could not open {name}: it is not there.": "无法打开 {name}：文件不存在。",
    # -- Help -------------------------------------------------------------------------------------------------
    "Open &Log Folder": "打开日志文件夹(&L)",
    "The folder of GIMaP's logs: gimap.log (what each session did), crash.log (native crashes) and errors.log "
    "(every unexpected error)":
        "GIMaP 日志所在的文件夹：gimap.log（每次会话做了什么）、crash.log（程序崩溃）和 errors.log（每个意外错误）",
    "The manual of GIMaP, opened in your web browser at the chapter of the workspace shown":
        "GIMaP 的用户手册，在网页浏览器中打开，并定位到当前工作区的章节",
    "Logs": "日志",
    "Nothing has been logged yet.": "还没有记录任何日志。",
}

__all__ = ["APP_SESSION_ZH"]

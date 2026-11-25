"""Help ▸ Check for Updates… for a source folder: the update dialog and the messages of ``app/source_update.py``."""

SOURCE_UPDATE_ZH = {
    # menu
    "You are running {name} {version}. A source folder (a Git clone, or a ZIP from GitHub's Code button) is brought "
    "to the newest version on GitHub and GIMaP starts again; an installed copy opens the releases page in your "
    "browser. GitHub is asked only when you choose this.":
        "你正在运行 {name} {version}。源码文件夹（Git 克隆，或从 GitHub 的 Code 按钮下载的 ZIP）会更新到 GitHub 上的最新版本，"
        "然后 GIMaP 重新启动；安装的版本在浏览器中打开发布页。只有选择这一项时才会访问 GitHub。",
    # dialog
    "Update GIMaP": "更新 GIMaP",
    "Open Releases Page": "打开发布页",
    "Update and Restart": "更新并重启",
    "Restart": "重启",
    "GIMaP {version} runs from a Git clone: {folder}.": "GIMaP {version} 从 Git 克隆运行：{folder}。",
    "GIMaP {version} runs from a folder downloaded from GitHub: {folder}.":
        "GIMaP {version} 从 GitHub 下载的文件夹运行：{folder}。",
    "Last updated here to {commit}.": "上次在这里更新到 {commit}。",
    "Update fetches the newest version of the branch from GitHub, updates this folder and starts GIMaP again. "
    "Your data, results and settings are not touched.":
        "更新会从 GitHub 取得该分支的最新版本，更新这个文件夹，然后重新启动 GIMaP。你的数据、结果和设置不会被改动。",
    "This is the newest version ({commit}, {date}).": "这已是最新版本（{commit}，{date}）。",
    "{n} new commits on GitHub ({upstream}, newest {commit} of {date}).":
        "GitHub 上有 {n} 个新提交（{upstream}，最新的是 {date} 的 {commit}）。",
    "{files} files differ from the newest version on GitHub ({commit} of {date}): {size} to download.":
        "有 {files} 个文件与 GitHub 上的最新版本（{date} 的 {commit}）不同：需下载 {size}。",
    "Nothing to update.": "没有需要更新的内容。",
    "Checking GitHub for the newest version…": "正在向 GitHub 查询最新版本…",
    "Updating…": "正在更新…",
    "Updated to {commit}: {files} files changed.": "已更新到 {commit}：改动了 {files} 个文件。",
    "The files it replaced are kept in {folder}.": "被替换的文件保存在 {folder}。",
    "The update changed the list of Python packages ({files}). Start_GIMaP.bat adds them at its next start; in an "
    "environment of your own, run this first:":
        "这次更新改动了 Python 包列表（{files}）。Start_GIMaP.bat 会在下次启动时自动补装；在你自己配置的环境中，请先运行：",
    "Then restart GIMaP.": "然后重启 GIMaP。",
    "GIMaP starts again now…": "GIMaP 正在重新启动…",
    "GIMaP was not closed. Restart it to use the new version.": "GIMaP 没有关闭。重启后才会使用新版本。",
    "The update failed: {error}": "更新失败：{error}",
    # app/source_update.py
    "Git was not found. This folder is a Git clone: install Git (git-scm.com) to update it.":
        "没有找到 Git。这个文件夹是 Git 克隆：要更新它，请先安装 Git（git-scm.com）。",
    "Git did not answer within {seconds} s (git {command}).": "Git 在 {seconds} 秒内没有响应（git {command}）。",
    "git {command} failed: {error}": "git {command} 失败：{error}",
    "This clone is not on a branch (a detached HEAD): nothing to update.":
        "这个克隆不在任何分支上（detached HEAD）：没有可更新的内容。",
    "The branch {branch} has no branch on GitHub to follow (no upstream).":
        "分支 {branch} 没有对应的 GitHub 分支可跟随（没有 upstream）。",
    "The branch {branch} has commits that are not on GitHub ({upstream}): nothing to update.":
        "分支 {branch} 有 GitHub（{upstream}）上没有的提交：没有可更新的内容。",
    "The branch {branch} and {upstream} have gone apart (commits of your own, or the history on GitHub was "
    "rewritten). Nothing was changed: see “Updating a source checkout” in README.md (on Windows, Update_GIMaP.bat).":
        "分支 {branch} 和 {upstream} 已经分叉（有你自己的提交，或 GitHub 上的历史被改写过）。没有做任何改动："
        "见 README.md 的 “Updating a source checkout”（Windows 上用 Update_GIMaP.bat）。",
    "You have edited files that the new version changes: {files}. Nothing was changed. Keep a copy of your edits, "
    "undo them (or commit them), then update again.":
        "你改过的文件新版本也改了：{files}。没有做任何改动。请先备份你的修改，撤销（或提交）它们，再更新。",
    "The clone changed since the check: check again.": "检查之后克隆有了变化：请重新检查。",
    "Moving {branch} forward to {commit}…": "正在把 {branch} 前移到 {commit}…",
    "Git did not update the clone (nothing was changed): {error}": "Git 没有更新这个克隆（没有做任何改动）：{error}",
    "GitHub could not be reached ({error}). Check the internet connection and try again.":
        "无法连接 GitHub（{error}）。请检查网络连接后重试。",
    "GitHub sent an incomplete file list. Download the new version as a ZIP instead.":
        "GitHub 返回的文件列表不完整。请改为下载新版本的 ZIP。",
    "The update was stopped. Nothing was changed.": "更新已停止。没有做任何改动。",
    "Downloading {done} of {total} files…": "正在下载第 {done} / {total} 个文件…",
    "{file} arrived damaged (its hash does not match GitHub's list). Nothing was changed: try again.":
        "{file} 下载后已损坏（哈希与 GitHub 的列表不符）。没有做任何改动：请重试。",
    "Putting {total} new files in place…": "正在放入 {total} 个新文件…",
    "A file could not be replaced ({error}); the folder was put back as it was. Close programs that use GIMaP's "
    "files and try again.":
        "有文件无法替换（{error}）；文件夹已恢复原样。请关闭正在使用 GIMaP 文件的程序后重试。",
    "This copy of GIMaP ({reason}) is updated by installing the new version.":
        "这份 GIMaP（{reason}）需要安装新版本来更新。",
    "There is nothing to update.": "没有需要更新的内容。",
}

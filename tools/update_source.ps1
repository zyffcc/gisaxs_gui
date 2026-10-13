# Updates this source checkout of GIMaP from origin (GitHub) - also after the history of origin was
# rewritten, when "git pull" stops with an error or wants to merge the old history with the new one.
#
#   powershell -ExecutionPolicy Bypass -File tools\update_source.ps1            update
#   powershell -ExecutionPolicy Bypass -File tools\update_source.ps1 -DryRun    report only
#
# It updates the branch that is checked out, in the checkout this file lives in:
#   1. git fetch origin. When origin has nothing new, nothing is done.
#   2. origin only added commits: the branch moves forward to them (a fast-forward).
#   3. The branch holds commits that are not on origin, and origin moved on - what a rewritten
#      history looks like from here: the state you have is kept first, as a branch
#      backup/<date>-<time>-<commit>; then the branch is moved to origin/<branch> (git reset --hard).
# Modifications of tracked files (all of them in 3, those in the way in 2) are put aside with
# git stash and put back on top of the new version; a copy stays in the stash. When they do not fit
# the new version, the new version is left complete and the script says what to do. When git cannot
# replace a file (another program has it open), the version you had is put back whole.
# Files git does not track or ignores (settings, data, notes) are never changed. When the new version
# brings a file to a place where you have a different file of your own, the script stops before it
# changes anything.
# Tracked files the repository wants byte for byte (.gitattributes: -text - the model files of
# 1D Predict) and that an older checkout holds with converted line endings are written again, also
# when there is nothing to update. Files you modified are left alone.
# -DryRun runs step 1 (a download: it changes no file, no branch and no stash of yours), reports what
# an update would do, and stops.
#
# Exit code: 0  updated, or nothing to do
#            1  not updated, or not completely (the reason is printed)
#            2  updated, but your modifications could not be put back: they wait in the stash
#
# Keep this file ASCII: Windows PowerShell 5.1 reads a script without a byte-order mark in the
# system code page.
[CmdletBinding()]
param(
    # Report what was found and what an update would do; change nothing.
    [Alias("WhatIf")]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$RunAgain = "powershell -ExecutionPolicy Bypass -File tools\update_source.ps1"
$GitExe = ""

function Say([string[]]$Text = @("")) { foreach ($Line in $Text) { Write-Host $Line } }

function Stop-NotUpdated([string[]]$Lines, [string]$Because = "nothing was changed.") {
    # The default is for a refusal before the first change: the checkout is exactly as it was.
    Say "", "NOT UPDATED: $Because"
    foreach ($Line in $Lines) { Say "  $Line" }
    exit 1
}

function Invoke-Git([string[]]$Arguments) {
    # Windows PowerShell turns every stderr line of a redirected program into an error record; with
    # "Stop" the first progress line of git would end the script.
    $ErrorActionPreference = "Continue"
    $Raw = @(& $script:GitExe -C $script:Root --literal-pathspecs -c core.quotepath=false @Arguments 2>&1)
    $Code = $LASTEXITCODE
    $Err = @($Raw | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] } | ForEach-Object { "$_" })
    $Out = @($Raw | Where-Object { $_ -isnot [System.Management.Automation.ErrorRecord] } | ForEach-Object { "$_" })
    return New-Object PSObject -Property @{ Code = $Code; Out = $Out; Err = $Err }
}

function Get-GitLine([string[]]$Arguments) {
    $Result = Invoke-Git $Arguments
    if ($Result.Code -ne 0 -or $Result.Out.Count -eq 0) { return "" }
    return [string]$Result.Out[0]
}

function Get-GitLines([string[]]$Arguments, [string[]]$Paths = $null) {
    # With paths: the same command for 50 of them at a time (a command line has a length limit).
    if ($null -eq $Paths) {
        $Result = Invoke-Git $Arguments
        if ($Result.Code -ne 0) { return @() }
        return @($Result.Out | Where-Object { $_ })
    }
    $Lines = @()
    for ($Start = 0; $Start -lt $Paths.Count; $Start += 50) {
        $Chunk = @($Paths[$Start..([Math]::Min($Start + 49, $Paths.Count - 1))])
        $Lines += @(Get-GitLines ($Arguments + @("--") + $Chunk))
    }
    return $Lines
}

function Get-GitSaid($Result) {
    # What git printed, indented, for a message of this script.
    return @(@($Result.Err) + @($Result.Out) | Where-Object { $_ } | ForEach-Object { "    git: $_" })
}

function Join-Names([string[]]$Names, [int]$Most = 6) {
    $Names = @($Names)
    if ($Names.Count -le $Most) { return ($Names -join ", ") }
    return ((@($Names[0..($Most - 1)]) -join ", ") + " and $($Names.Count - $Most) more")
}

function Get-Commit([string]$Commit) { return (Get-GitLine @("log", "-1", "--format=%h %s", $Commit)) }   # "1a2b3c4 subject"
function New-NameSet { return ,[System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase) }

function Get-FullPath([string]$Path) {
    # Windows PowerShell does not see a path of 260 characters or more without this prefix, and a
    # file it does not see would be overwritten.
    $Full = Join-Path $script:Root $Path.Replace("/", "\")
    if ($Full.Length -ge 248 -and -not $Full.StartsWith("\\")) { $Full = "\\?\$Full" }
    return $Full
}

function Get-LocalChanges([switch]$OrStop) {
    # Tracked files that differ from the commit you have: in the folder, or staged.
    $Seen = New-NameSet
    $Paths = @()
    foreach ($Extra in @("--no-renames", "--cached")) {
        $Result = Invoke-Git @("diff", "--name-only", "--no-renames", $Extra, "HEAD", "--")
        if ($Result.Code -ne 0 -and $OrStop) {
            # Never "no modifications" because git failed: the update would overwrite them.
            Stop-NotUpdated (@("git could not compare your files with the commit you have, so the script did not go on.") + (Get-GitSaid $Result))
        }
        foreach ($Path in @($Result.Out | Where-Object { $_ })) { if ($Seen.Add($Path)) { $Paths += $Path } }
    }
    return $Paths
}

function Get-DifferentFrom([string]$Commit, [string[]]$Paths) {
    # Of these tracked files, the ones whose content here is not the content of the commit: in the
    # folder, or staged by you (what git has staged is the commit you have, unless you staged it).
    $Paths = @($Paths)      # never without paths: the command would then compare every file
    $Different = New-NameSet
    foreach ($Path in (Get-GitLines @("diff", "--name-only", "--no-renames", $Commit) $Paths)) { [void]$Different.Add($Path) }
    $Staged = New-NameSet
    foreach ($Path in (Get-GitLines @("diff", "--name-only", "--no-renames", "--cached", "HEAD") $Paths)) { [void]$Staged.Add($Path) }
    foreach ($Path in (Get-GitLines @("diff", "--name-only", "--no-renames", "--cached", $Commit) $Paths)) {
        if ($Staged.Contains($Path)) { [void]$Different.Add($Path) }
    }
    return @($Paths | Where-Object { $Different.Contains($_) })
}

function Get-InTheWay([string]$Commit) {
    # Files of yours that git does not track here (untracked or ignored), at places where the
    # commit has a file of its own: git would overwrite them without a word.
    $Tracked = New-NameSet
    foreach ($Path in (Get-GitLines @("ls-files"))) { [void]$Tracked.Add($Path) }
    $Same = @()
    $Different = @()
    $Folders = New-NameSet
    foreach ($Line in (Get-GitLines @("ls-tree", "-r", $Commit))) {
        # <mode> <type> <object>TAB<path>
        $Tab = $Line.IndexOf("`t")
        if ($Tab -lt 0) { continue }
        $Path = $Line.Substring($Tab + 1)
        if ($Tracked.Contains($Path)) { continue }
        $Fields = $Line.Substring(0, $Tab).Split(" ")
        # A file of yours where the new version needs a folder.
        $Parts = $Path.Split("/")
        for ($Depth = 1; $Depth -lt $Parts.Count; $Depth++) {
            $Folder = ($Parts[0..($Depth - 1)] -join "/")
            if ($Folders.Add($Folder) -and -not $Tracked.Contains($Folder) -and
                [System.IO.File]::Exists((Get-FullPath $Folder))) {
                $Different += $Folder
            }
        }
        $Full = Get-FullPath $Path
        $IsFile = [System.IO.File]::Exists($Full)
        if (-not $IsFile -and -not [System.IO.Directory]::Exists($Full)) { continue }
        $IsSame = $false
        if ($IsFile -and $Fields[1] -eq "blob" -and $Fields[0] -ne "120000") {
            # The bytes on disk, then the bytes as git would store them (line endings).
            if ((Get-GitLine @("hash-object", "--no-filters", "--", $Path)) -eq $Fields[2]) { $IsSame = $true }
            elseif ((Get-GitLine @("hash-object", "--", $Path)) -eq $Fields[2]) { $IsSame = $true }
        }
        if ($IsSame) { $Same += $Path } else { $Different += $Path }
    }
    return New-Object PSObject -Property @{ Same = $Same; Different = $Different }
}

function Get-ConvertedFiles {
    # .gitattributes tells git to hand some files out byte for byte ("-text": the model files of
    # 1D Predict, which the program checks against SHA-256 sums). A checkout made before that rule
    # holds them with converted line endings. git mostly does not notice (it trusts its own record
    # of the file); when it does, it shows them as modified.
    $Suspects = @()
    foreach ($Line in (Get-GitLines @("ls-files", "--eol"))) {
        # i/<line endings git has> w/<line endings on disk> attr/<attributes>TAB<path>
        if ($Line -match '^i/(\S*)\s+w/(\S+)\s+attr/-text[^\t]*\t(.+)$' -and $Matches[1] -ne $Matches[2]) { $Suspects += $Matches[3] }
    }
    if ($Suspects.Count -eq 0) { return @() }
    $Modified = New-NameSet
    foreach ($Path in @(Get-LocalChanges)) { [void]$Modified.Add($Path) }
    $Converted = @()
    foreach ($Path in $Suspects) {
        # Shown as modified: only the line endings, or did you change the file as well? Then it is yours.
        if (-not $Modified.Contains($Path) -or
            (Invoke-Git @("diff", "--ignore-cr-at-eol", "--quiet", "HEAD", "--", $Path)).Code -eq 0) { $Converted += $Path }
    }
    return $Converted
}

function Repair-ConvertedFiles {
    # Writes those files again and says so. Returns $false when some are still converted afterwards.
    $Converted = @(Get-ConvertedFiles)
    if ($Converted.Count -eq 0) { return $true }
    foreach ($Path in $Converted) {
        # git writes a file again only when it looks changed, and its record says these are untouched.
        try { [System.IO.File]::SetLastWriteTimeUtc((Get-FullPath $Path), [DateTime]::UtcNow) } catch { }
    }
    [void](Get-GitLines @("checkout-index", "--force", "--index") $Converted)
    $Left = @(Get-ConvertedFiles)
    if ($Left.Count -gt 0) {
        Say "  - $($Left.Count) file(s) still hold converted line endings: git could not write them again (is GIMaP"
        Say "    still open?). 1D Predict will not accept them: $(Join-Names $Left 3)"
        return $false
    }
    Say "  - wrote $($Converted.Count) file(s) again, byte for byte as the repository has them: an older checkout"
    Say "    had converted their line endings, and 1D Predict checks these files"
    return $true
}

function Get-EarlierOriginTips([string]$RemoteRef, [string]$Before, [string]$Target) {
    # The commits this computer has seen origin/<branch> at: before this fetch, and in git's log of
    # that reference. A rewritten history shows as commits that were there once and are gone.
    $Tips = @()
    if ($Before) { $Tips += $Before }
    $LogPath = Get-GitLine @("rev-parse", "--git-path", "logs/$RemoteRef")
    if ($LogPath -and -not [System.IO.Path]::IsPathRooted($LogPath)) { $LogPath = Join-Path $script:Root $LogPath }
    if ($LogPath -and (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        foreach ($Line in @(Get-Content -LiteralPath $LogPath -Tail 300)) {
            if ($Line -match '^([0-9a-f]{40,64}) ([0-9a-f]{40,64}) ') { $Tips += @($Matches[1], $Matches[2]) }
        }
    }
    return @($Tips | Where-Object { $_ -ne $Target -and $_ -notmatch '^0+$' } | Select-Object -Unique)
}

# git writes file names and commit subjects in UTF-8.
$PreviousEncoding = $null
try { $PreviousEncoding = [Console]::OutputEncoding } catch { }
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }

try {
    if ($DryRun) { Say "GIMaP source update - dry run: it reports, and changes nothing" }
    else { Say "GIMaP source update" }

    # ---- Can this folder be updated at all?
    $GitCommand = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $GitCommand) {
        Stop-NotUpdated @(
            "git was not found on this computer (it is not on the PATH). Install Git for Windows",
            "(https://git-scm.com/download/win), open a new terminal and run this script again.")
    }
    $GitExe = $GitCommand.Path
    $VersionText = "$(& $GitExe --version)"
    if ($VersionText -match '(\d+)\.(\d+)' -and ([int]$Matches[1] * 100 + [int]$Matches[2]) -lt 216) {
        Stop-NotUpdated @(
            "This git is too old for the script ($VersionText; it needs 2.16 or newer).",
            "Install a current Git for Windows (https://git-scm.com/download/win) and run this script again.")
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Root ".git"))) {
        Stop-NotUpdated @(
            "This folder is not a git checkout (it has no .git): $Root",
            "A copy from the portable ZIP is updated by downloading the new release, not with this script.")
    }
    $Inside = Invoke-Git @("rev-parse", "--is-inside-work-tree")
    if ($Inside.Code -ne 0 -or $Inside.Out -notcontains "true") {
        Stop-NotUpdated (@("git cannot use this folder as a checkout: $Root") + (Get-GitSaid $Inside))
    }
    $GitDir = Get-GitLine @("rev-parse", "--absolute-git-dir")
    $Unfinished = @(
        @{ File = "MERGE_HEAD"; What = "A merge (a 'git pull' of a rewritten history starts one)"; GiveUp = "git merge --abort" },
        @{ File = "rebase-merge"; What = "A rebase"; GiveUp = "git rebase --abort" },
        @{ File = "rebase-apply"; What = "A rebase"; GiveUp = "git rebase --abort" },
        @{ File = "CHERRY_PICK_HEAD"; What = "A cherry-pick"; GiveUp = "git cherry-pick --abort" },
        @{ File = "REVERT_HEAD"; What = "A revert"; GiveUp = "git revert --abort" }
    )
    foreach ($Operation in $Unfinished) {
        if ($GitDir -and (Test-Path -LiteralPath (Join-Path $GitDir $Operation.File))) {
            Stop-NotUpdated @(
                "$($Operation.What) is in progress in this checkout: it was started and not finished.",
                "Finish it, or give it up with:  $($Operation.GiveUp)",
                "Then run this script again:     $RunAgain")
        }
    }
    $Conflicted = @(Get-GitLines @("diff", "--name-only", "--diff-filter=U"))
    if ($Conflicted.Count -gt 0) {
        Stop-NotUpdated @(
            "$($Conflicted.Count) file(s) hold an unresolved conflict: $(Join-Names $Conflicted)",
            "Resolve them first ('git status' shows them), then run this script again.")
    }
    $Branch = Get-GitLine @("symbolic-ref", "--quiet", "--short", "HEAD")
    if (-not $Branch) {
        Stop-NotUpdated @(
            "No branch is checked out here (git calls it a detached HEAD), so there is no branch to update.",
            "Switch to the branch first (for example: git checkout main), then run this script again.")
    }
    $Head = Get-GitLine @("rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    if (-not $Head) { Stop-NotUpdated @("The branch '$Branch' has no commit yet: there is nothing to update.") }
    $OriginUrl = Get-GitLine @("remote", "get-url", "origin")
    if (-not $OriginUrl) {
        Stop-NotUpdated @(
            "This checkout has no remote named 'origin', so the script does not know where to update from.",
            "A checkout made with 'git clone' has one; 'git remote -v' lists what this one has.")
    }
    $Short = Get-GitLine @("rev-parse", "--short", "HEAD")
    Say "  folder : $Root"
    Say "  branch : $Branch, at $(Get-Commit $Head)"
    Say "  origin : $OriginUrl"

    # ---- What does origin have?
    $RemoteRef = "refs/remotes/origin/$Branch"
    $RemoteBefore = Get-GitLine @("rev-parse", "--verify", "--quiet", "$RemoteRef^{commit}")
    Say "", "Asking origin what it has (git fetch origin) ..."
    # Not captured: git's progress, its questions (an SSH passphrase) and its errors reach the terminal.
    # "Continue": where the caller redirects the error stream, PowerShell makes errors of those lines.
    $ErrorActionPreference = "Continue"
    & $GitExe -C $Root fetch origin | Out-Host
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) {
        Stop-NotUpdated @(
            "git could not download from origin (its own message is above). Check the network connection",
            "and your access to the repository (for an SSH address: the key), then run this script again.")
    }
    $Target = Get-GitLine @("rev-parse", "--verify", "--quiet", "$RemoteRef^{commit}")
    if (-not $Target) {
        Stop-NotUpdated @(
            "origin has no branch named '$Branch': this branch exists on this computer only.",
            "To update the main line, switch to it (git checkout main) and run this script again.")
    }

    # Commits you have that origin/<branch> does not have, oldest first.
    $LocalOnly = @()
    foreach ($Line in (Get-GitLines @("log", "--reverse", "--format=%H%x09%P%x09%h%x09%s", "$Target..$Head"))) {
        $Fields = @($Line -split "`t", 4) + @("", "", "")
        $LocalOnly += New-Object PSObject -Property @{
            Id = $Fields[0]; Merge = ($Fields[1].Trim().Contains(" ")); Short = $Fields[2]; Text = "$($Fields[2]) $($Fields[3])"
        }
    }
    $Behind = 0
    [void][int]::TryParse((Get-GitLine @("rev-list", "--count", "$Head..$Target")), [ref]$Behind)

    # Of those commits: which were on origin once (and were rewritten away), which are your own?
    $Gone = [System.Collections.Generic.HashSet[string]]::new()
    if ($LocalOnly.Count -gt 0) {
        $Tips = @(Get-EarlierOriginTips $RemoteRef $RemoteBefore $Target)
        if ($Tips.Count -gt 0) {
            foreach ($Id in (Get-GitLines (@("rev-list", "--ignore-missing") + $Tips + @("--not", $Target)))) {
                [void]$Gone.Add($Id)
            }
        }
    }
    $Rewritten = @($LocalOnly | Where-Object { $Gone.Contains($_.Id) })
    $Own = @($LocalOnly | Where-Object { -not $Gone.Contains($_.Id) -and -not $_.Merge })
    $OwnMerges = @($LocalOnly | Where-Object { -not $Gone.Contains($_.Id) -and $_.Merge })

    # "forward" only when git says so itself: that case moves the branch without a backup branch.
    if ($Head -eq $Target) { $Case = "same" }
    elseif ($LocalOnly.Count -eq 0 -and (Invoke-Git @("merge-base", "--is-ancestor", $Head, $Target)).Code -eq 0) { $Case = "forward" }
    elseif ($Behind -eq 0 -and $Rewritten.Count -eq 0 -and $LocalOnly.Count -gt 0) { $Case = "ahead" }
    else { $Case = "replace" }

    # Line endings an older checkout converted are not modifications of yours.
    $Changed = @(Get-LocalChanges -OrStop)
    $Converted = @(Get-ConvertedFiles)
    $Changed = @($Changed | Where-Object { $Converted -notcontains $_ })
    Say "", "What was found"
    if ($Case -eq "same" -or $Case -eq "ahead") {
        if ($Case -eq "same") { Say "  origin/$Branch is at the commit you have: there is nothing to update." }
        else {
            Say "  origin/$Branch has nothing new. Your branch has $($LocalOnly.Count) commit(s) of your own on top of it:"
            foreach ($Commit in @($LocalOnly | Select-Object -Last 10)) { Say "    $($Commit.Text)" }
        }
        if ($Changed.Count -gt 0) { Say "  Your modifications of $($Changed.Count) tracked file(s) stay as they are." }
        if ($Converted.Count -eq 0 -or $DryRun) {
            if ($Converted.Count -gt 0) {
                Say "  $($Converted.Count) file(s) hold converted line endings where 1D Predict checks every byte:"
                Say "  an update would write them again."
            }
            Say "", "ALREADY UP TO DATE: nothing was changed."
            exit 0
        }
        Say "  $($Converted.Count) file(s) hold converted line endings where 1D Predict checks every byte."
        Say "", "What was done"
        if (Repair-ConvertedFiles) {
            Say "", "ALREADY UP TO DATE: only line endings were corrected."
            exit 0
        }
        Say "", "ALREADY UP TO DATE, BUT NOT COMPLETE (see above). Run this script again."
        exit 1
    }
    if ($Case -eq "forward") {
        Say "  origin/$Branch has $Behind new commit(s); your branch only has to move forward to them."
    }
    else {
        if ($Rewritten.Count -gt 0) {
            Say "  The history of origin/$Branch was rewritten: $($Rewritten.Count) commit(s) you have are no longer part of it."
        }
        else { Say "  Your branch and origin/$Branch have gone different ways: it cannot simply move forward." }
        if ($Behind -gt 0) { Say "  origin/$Branch has $Behind commit(s) you do not have." }
        else { Say "  origin/$Branch has nothing you do not have, but your branch still carries the old history." }
        if ($Own.Count -gt 0) {
            # Without a record of a rewrite, a commit origin does not have may also be one it replaced.
            if ($Rewritten.Count -gt 0) { Say "  Your branch has $($Own.Count) commit(s) of your own (never seen on origin):" }
            else { Say "  Your branch has $($Own.Count) commit(s) that are not on origin (your own, or ones origin has replaced):" }
            foreach ($Commit in @($Own | Select-Object -Last 10)) { Say "    $($Commit.Text)" }
        }
        if ($OwnMerges.Count -gt 0) {
            Say "  Your branch also holds $($OwnMerges.Count) merge(s) made here (a 'git pull' of the rewritten history makes one)."
        }
    }

    $Keep = @(Get-DifferentFrom $Target $Changed)
    $InTheWay = Get-InTheWay $Target
    $Updated = New-NameSet
    foreach ($Path in (Get-GitLines @("diff", "--name-only", "--no-renames", $Head, $Target, "--"))) { [void]$Updated.Add($Path) }
    $Overlap = @($Changed | Where-Object { $Updated.Contains($_) })
    # git moves a branch forward around modified files, unless the update changes the same files.
    $Plain = ($Case -eq "forward" -and $Overlap.Count -eq 0 -and $InTheWay.Same.Count -eq 0)
    $NeedStash = (-not $Plain -and $Keep.Count -gt 0)

    if ($Changed.Count -eq 0) { Say "  Modifications of tracked files here: none." }
    else {
        Say "  Modifications of tracked files here: $($Changed.Count) ($(Join-Names $Changed))."
        if ($Keep.Count -lt $Changed.Count) { Say "    $($Changed.Count - $Keep.Count) of them already hold what the new version holds." }
        if ($Plain) { Say "    The update does not change these files: they stay as they are." }
    }
    if ($Converted.Count -gt 0) {
        Say "  $($Converted.Count) file(s) hold converted line endings where 1D Predict checks every byte."
    }
    if ($InTheWay.Same.Count -gt 0) {
        Say "  $($InTheWay.Same.Count) file(s) git does not track here are part of the new version, with exactly the content"
        Say "  you have: their content stays, and git tracks them from now on."
        foreach ($Path in @($InTheWay.Same | Select-Object -First 3)) { Say "    $Path" }
        if ($InTheWay.Same.Count -gt 3) { Say "    ... and $($InTheWay.Same.Count - 3) more" }
    }
    if ($InTheWay.Different.Count -eq 0) { Say "  Files git does not track or ignores (settings, data, notes): never touched." }
    else {
        $Lines = @(
            "The new version brings $($InTheWay.Different.Count) file(s) to places where you have different files of your own.",
            "git does not track yours, so an update would overwrite them:")
        foreach ($Path in @($InTheWay.Different | Select-Object -First 20)) { $Lines += "    $Path" }
        if ($InTheWay.Different.Count -gt 20) { $Lines += "    ... and $($InTheWay.Different.Count - 20) more" }
        $Lines += "Move or rename them (or delete them when you do not need them), then run this script again:"
        $Lines += "    $RunAgain"
        Stop-NotUpdated $Lines
    }

    # ---- The plan, in the words of the report.
    $Stamp = (Get-Date).ToString("yyyyMMdd-HHmmss", [System.Globalization.CultureInfo]::InvariantCulture)
    $Backup = ""
    $BackupIsNew = $true
    if ($Case -eq "replace") {
        # A second run after an interrupted one finds the backup of the first.
        $Backup = Get-GitLine @("for-each-ref", "--count=1", "--points-at", $Head, "--format=%(refname:short)", "refs/heads/backup/")
        if ($Backup) { $BackupIsNew = $false }
        else {
            $Backup = "backup/$Stamp-$Short"
            $Number = 2
            while (Get-GitLine @("rev-parse", "--verify", "--quiet", "refs/heads/$Backup")) {
                $Backup = "backup/$Stamp-$Short-$Number"
                $Number++
            }
        }
    }
    $StashName = "update_source ${Stamp}: modifications on $Short before the update"

    if ($DryRun) {
        Say "", "What an update would do"
        if ($Backup -and $BackupIsNew) { Say "  - keep the state you have as a new branch, named like $Backup" }
        elseif ($Backup) { Say "  - the state you have is already kept as branch $Backup (nothing to do)" }
        if ($NeedStash) { Say "  - put your modifications of $($Keep.Count) tracked file(s) aside (git stash)" }
        if ($Plain) { Say "  - move $Branch forward to origin/$Branch (git merge --ff-only): $(Get-Commit $Target)" }
        else { Say "  - move $Branch to origin/$Branch (git reset --hard): $(Get-Commit $Target)" }
        if ($NeedStash) { Say "  - put your modifications back on top of the new version, and keep a copy in the stash" }
        if ($Updated.Contains(".gitattributes") -or $Converted.Count -gt 0) {
            Say "  - write again the files whose line endings an older checkout converted (1D Predict checks them)"
        }
        Say "", "DRY RUN: nothing was changed. To update:"
        Say "    $RunAgain"
        exit 0
    }

    # ---- Keep what is local, move the branch, put the modifications back.
    Say "", "What was done"
    $BackupNote = @()
    if ($Backup -and $BackupIsNew) {
        $Made = Invoke-Git @("branch", "--no-track", $Backup, $Head)
        if ($Made.Code -ne 0) {
            Stop-NotUpdated (@("git could not create the backup branch $Backup, so the script did not go on.") + (Get-GitSaid $Made))
        }
        Say "  - kept the state you had as the new branch $Backup"
        $BackupNote = @("The branch $Backup was created; it changes nothing, and 'git branch -D $Backup' deletes it.")
    }
    elseif ($Backup) { Say "  - the state you had is already kept as branch $Backup" }
    # Before the move, so that git does not take them for modifications; again after it, when the
    # new version may be the one that brings the rule.
    if ($Converted.Count -gt 0) { [void](Repair-ConvertedFiles) }

    $Stash = ""
    if ($NeedStash) {
        $StashBefore = Get-GitLine @("rev-parse", "--verify", "--quiet", "refs/stash")
        $Identity = @()
        if ((Invoke-Git @("var", "GIT_COMMITTER_IDENT")).Code -ne 0) {
            # git stash stores the modifications as a commit and wants a name for it.
            $Identity = @("-c", "user.name=GIMaP-update", "-c", "user.email=gimap-update@localhost")
        }
        $Pushed = Invoke-Git ($Identity + @("stash", "push", "--quiet", "--message", $StashName))
        $Stash = Get-GitLine @("rev-parse", "--verify", "--quiet", "refs/stash")
        if ($Pushed.Code -ne 0 -or ($Stash -eq $StashBefore -and @(Get-LocalChanges).Count -gt 0)) {
            Stop-NotUpdated (@(
                "git could not put your modifications aside (git stash), so the branch was not moved.",
                "Your files are as they were.") + $BackupNote + (Get-GitSaid $Pushed)) "git stopped on the way."
        }
        if ($Stash -eq $StashBefore) { $Stash = "" }
        else { Say "  - put your modifications of $($Keep.Count) tracked file(s) aside (git stash)" }
    }

    if ($Plain) { $Moved = Invoke-Git @("merge", "--ff-only", "--quiet", $RemoteRef) }
    else { $Moved = Invoke-Git @("reset", "--hard", "--quiet", $Target) }
    $Now = Get-GitLine @("rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    if ($Moved.Code -ne 0 -or $Now -ne $Target) {
        $Lines = @("git could not bring the files to the new version. Is GIMaP, or an editor, still open on them?")
        $Lines += (Get-GitSaid $Moved)
        # Half way is the worst place to stop: write the files git had already replaced as they were
        # (not the ones you modified yourself, unless those went to the stash), then your modifications.
        $Touched = @($Updated | Where-Object { $Stash -or $Changed -notcontains $_ })
        if ($Now -eq $Head) { [void](Get-GitLines @("checkout-index", "--force", "--index") @(Get-GitLines @("ls-files") $Touched)) }
        $Whole = ($Now -eq $Head -and @(Get-DifferentFrom $Head $Touched).Count -eq 0)
        if ($Whole -and $Stash -and (Invoke-Git @("stash", "pop", "--index", "--quiet")).Code -eq 0) { $Stash = "" }
        if ($Whole) { $Lines += "Nothing is left half way: the files git had already replaced were written again as they were." }
        else { $Lines += "The folder now holds files of both versions: GIMaP may not start from it until the update went through." }
        if ($Stash) { $Lines += "Your modifications are in the stash: after a successful update, 'git stash pop' puts them back." }
        elseif ($Changed.Count -gt 0) { $Lines += "Your modifications are in the files, as before." }
        if ($Backup) { $Lines += "The state you had is kept as branch $Backup." }
        $Lines += "Close the programs that use the folder and run this script again:"
        $Lines += "    $RunAgain"
        Stop-NotUpdated $Lines "git stopped on the way."
    }
    if ($Plain) { Say "  - moved $Branch forward to origin/$Branch (git merge --ff-only)" }
    else { Say "  - moved $Branch to origin/$Branch (git reset --hard)" }

    $NeedsYou = @()
    if ($Stash) {
        $Applied = Invoke-Git @("stash", "apply", "--quiet", $Stash)
        if ($Applied.Code -eq 0) { Say "  - put your modifications back on top of the new version; a copy stays in the stash" }
        else {
            $NeedsYou = @(Get-GitLines @("diff", "--name-only", "--diff-filter=U"))
            if ($NeedsYou.Count -eq 0) { $NeedsYou = @($Keep) }
            # Leave the new version whole: half-merged files would not even start.
            [void](Invoke-Git @("reset", "--hard", "--quiet", $Target))
            Say "  - your modifications do NOT fit the new version: they were left in the stash"
        }
    }
    $Complete = Repair-ConvertedFiles

    Say ""
    if ($NeedsYou.Count -gt 0) { Say "UPDATED, BUT YOUR MODIFICATIONS NEED YOU: $Branch is now at $(Get-Commit $Target)" }
    elseif (-not $Complete) { Say "UPDATED, BUT NOT COMPLETE (see above): $Branch is now at $(Get-Commit $Target)" }
    else { Say "UPDATED: $Branch is now at $(Get-Commit $Target)" }

    if ($NeedsYou.Count -gt 0) {
        Say "", "Your modifications"
        Say "  The new version is complete and works as it is. What you had changed is not in the files now:"
        Say "  git could not combine it with the new version in $(Join-Names $NeedsYou)."
        Say "  Nothing is lost - the modifications wait in the stash:"
        Say "    git stash show -p     shows them"
        Say "    git stash pop         tries again and marks the conflicts in the files, for you to resolve"
        Say "                          ('git status' lists them; 'git reset --hard' gives the attempt up)"
    }
    if ($Case -eq "replace" -and $Own.Count -gt 0) {
        Say "", "Your own commits"
        $Which = "<its number>      (never one that origin has replaced by a new version)"
        if ($Rewritten.Count -gt 0) { $Which = @($Own | ForEach-Object { $_.Short }) -join " " }
        Say "  They are on the backup branch, not on ${Branch}. To bring your own over to the new history, oldest first:"
        Say "    git cherry-pick $Which"
    }
    if (@(Get-GitLines @("diff", "--name-only", $Head, $Target, "--", "requirements.txt")).Count -gt 0) {
        Say "", "The list of Python packages changed. In the environment you run GIMaP from:"
        Say "    python -m pip install -r requirements.txt"
    }

    Say "", "To go back to what you had"
    if ($Backup) { Say "    git reset --hard $Backup" }
    else { Say "    git reset --hard $Short" }
    if ($Stash) { Say "    git stash pop" }
    if ($Stash -and $NeedsYou.Count -eq 0) { Say "  (the first command drops uncommitted modifications of tracked files, the second puts yours back)" }
    elseif ($Plain -and $Changed.Count -gt 0) { Say "  (it drops uncommitted modifications of tracked files: run 'git stash' first to keep yours)" }
    if ($Backup -or ($Stash -and $NeedsYou.Count -eq 0)) {
        Say "", "Once things look right, delete what was kept"
        if ($Backup) {
            Say "    git branch -D $Backup"
            Say "  (until then the backup branch keeps the old history on this computer)"
        }
        if ($Stash -and $NeedsYou.Count -eq 0) {
            Say "    git stash drop"
            Say "  (it deletes the newest entry of 'git stash list': the copy of your modifications)"
        }
    }
    if ($NeedsYou.Count -gt 0) { exit 2 }
    if (-not $Complete) { exit 1 }
    exit 0
}
finally {
    if ($PreviousEncoding) { try { [Console]::OutputEncoding = $PreviousEncoding } catch { } }
}

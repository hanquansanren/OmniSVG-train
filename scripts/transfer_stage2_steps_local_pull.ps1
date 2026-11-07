# 在本机 Windows 后台拉取三个 step 到 E:\temp\2308（推荐：不依赖反向隧道）
# 用法（PowerShell）:
#   cd <OmniSVG-train 本地克隆或任意目录>
#   .\scripts\transfer_stage2_steps_local_pull.ps1
# 后台:
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','.\scripts\transfer_stage2_steps_local_pull.ps1' -WindowStyle Hidden

$ErrorActionPreference = "Stop"

$RemoteHost = "xeclogin1.xjtlu.edu.cn"
$RemoteUser = "weiguangzhang21"
$RemoteBase = "/gpfs/work/int/weiguangzhang21/project/OmniSVG-train/output_stage2/omnisvg_stage2_4b_20261020_230852"
$Steps = @("step_110000", "step_145000", "step_155000")
$Dest = "E:\temp\2308"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$Log = Join-Path $Dest ("transfer_pull_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format "o"), $msg
    Add-Content -Path $Log -Value $line
    Write-Host $line
}

Write-Log "LOG=$Log"
Write-Log "DEST=$Dest"

$rsync = Get-Command rsync -ErrorAction SilentlyContinue
$wsl = Get-Command wsl -ErrorAction SilentlyContinue

foreach ($step in $Steps) {
    $remote = "${RemoteUser}@${RemoteHost}:${RemoteBase}/${step}/"
    $local = Join-Path $Dest $step
    New-Item -ItemType Directory -Force -Path $local | Out-Null
    Write-Log "START $step -> $local"

    if ($rsync) {
        & rsync -avh --partial --inplace --info=progress2 -e "ssh" $remote $local 2>&1 | ForEach-Object { Add-Content $Log $_; Write-Host $_ }
    }
    elseif ($wsl) {
        $wslDest = "/mnt/e/temp/2308/$step/"
        $inner = "mkdir -p '$wslDest' && rsync -avh --partial --inplace --info=progress2 -e ssh '$remote' '$wslDest'"
        wsl -- bash -lc $inner 2>&1 | ForEach-Object { Add-Content $Log $_; Write-Host $_ }
    }
    else {
        Write-Log "未找到 rsync/WSL，使用 scp -r（无细粒度速度）"
        scp -r "${RemoteUser}@${RemoteHost}:${RemoteBase}/${step}" $Dest 2>&1 | ForEach-Object { Add-Content $Log $_; Write-Host $_ }
    }

    Write-Log "DONE $step"
}

Write-Log "全部完成"

@echo off
setlocal

REM Project root = script directory
set "ROOT=%~dp0"

REM Delegate everything to PowerShell (avoids chcp 65001 double-char bug and wmic parsing issues)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8;" ^
    "$root = '%ROOT%'.TrimEnd('\');" ^
    "$dbDir = Join-Path $root 'database';" ^
    "Write-Host '============================================';" ^
    "Write-Host '  Photo Identify 数据备份工具';" ^
    "Write-Host '============================================';" ^
    "Write-Host '';" ^
    "if (-not (Test-Path $dbDir)) {" ^
    "    Write-Host '[错误] 未找到 database 目录' -ForegroundColor Red; exit 1" ^
    "};" ^
    "$ts = Get-Date -Format 'yyyyMMdd_HHmmss';" ^
    "$zipName = 'photo_identify_backup_' + $ts + '.zip';" ^
    "$zipPath = Join-Path ([Environment]::GetFolderPath('Desktop')) $zipName;" ^
    "Write-Host ('备份目标: ' + $zipPath);" ^
    "Write-Host '';" ^
    "$tempDir = Join-Path $env:TEMP ('photo_identify_backup_' + [guid]::NewGuid().ToString('N'));" ^
    "$destDb = Join-Path $tempDir 'database';" ^
    "New-Item -ItemType Directory -Path $destDb -Force | Out-Null;" ^
    "$count = 0;" ^
    "Get-ChildItem -Path $dbDir -Filter '*.db' | Where-Object { $_.Length -gt 0 } | ForEach-Object {" ^
    "    Copy-Item $_.FullName -Destination $destDb;" ^
    "    Write-Host ('  + database/' + $_.Name + '  (' + [math]::Round($_.Length/1MB, 1) + ' MB)');" ^
    "    $count++;" ^
    "};" ^
    "$iniFile = Join-Path $dbDir 'photo_identify_gui.ini';" ^
    "if (Test-Path $iniFile) {" ^
    "    Copy-Item $iniFile -Destination $destDb;" ^
    "    Write-Host '  + database/photo_identify_gui.ini';" ^
    "    $count++;" ^
    "};" ^
    "if ($count -eq 0) {" ^
    "    Write-Host '[错误] 没有找到需要备份的文件' -ForegroundColor Red;" ^
    "    Remove-Item $tempDir -Recurse -Force; exit 1" ^
    "};" ^
    "Write-Host '';" ^
    "if (Test-Path $zipPath) { Remove-Item $zipPath -Force };" ^
    "Compress-Archive -Path (Join-Path $tempDir '*') -DestinationPath $zipPath -CompressionLevel Optimal;" ^
    "Remove-Item $tempDir -Recurse -Force;" ^
    "$size = (Get-Item $zipPath).Length;" ^
    "Write-Host '';" ^
    "Write-Host ('备份完成! 共 ' + $count + ' 个文件');" ^
    "Write-Host ('文件: ' + $zipPath);" ^
    "Write-Host ('大小: ' + [math]::Round($size/1MB, 2) + ' MB');" ^
    "Write-Host '';" ^
    "Write-Host '============================================';" ^
    "Write-Host '  解压后覆盖 database 目录即可还原';" ^
    "Write-Host '============================================';"

pause

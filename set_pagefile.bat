@echo off
:: Fix for WinError 1455 + MemoryError when loading the 21 GB DreamX-World-5B
:: checkpoint. The default ~4.5 GB system-managed page file is too small to back
:: a 21 GB mmap (1455) OR commit a 21 GB read (MemoryError). This sets a FIXED
:: 32 GB page file on C:, which lets the mmap path (load_file) load the checkpoint
:: at ~21 GB peak RAM, deterministically and concurrency-safe.
::
:: MUST be run as Administrator, then REBOOT for it to take effect.
:: (32768 MB needs ~32 GB free on C:. Bump to 49152/65536 if you free more disk.)

net session >nul 2>&1
if errorlevel 1 (
    echo ERROR: run this script as Administrator ^(right-click -^> Run as administrator^).
    pause
    exit /b 1
)

echo Disabling automatic page file management and setting a fixed 32 GB page file on C:...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$cs = Get-CimInstance Win32_ComputerSystem;" ^
  "if ($cs.AutomaticManagedPagefile) { Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile=$false} };" ^
  "$pf = Get-CimInstance Win32_PageFileSetting -Filter \"Name='C:\\pagefile.sys'\";" ^
  "if ($pf) { Set-CimInstance -InputObject $pf -Property @{InitialSize=32768; MaximumSize=32768} }" ^
  "else { New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name='C:\\pagefile.sys'; InitialSize=32768; MaximumSize=32768} };" ^
  "Write-Host 'Page file setting applied.'"

echo.
echo ============================================================
echo Done. REBOOT now for the 32 GB page file to take effect.
echo After reboot, re-run drive_dreamx_ar.bat / run_examples_ar.bat -
echo the checkpoint will load via mmap at ~21 GB peak (no 1455, no MemoryError).
echo ============================================================

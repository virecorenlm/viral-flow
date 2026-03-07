<# 
Why: Windows-native bootstrap so you can set up Viral-Flow without WSL while keeping `setup.sh` for Linux rigs.
#>

[CmdletBinding()]
param(
    # Why: Allow skipping pip install when you only want preflight checks.
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

<# 
Why: Ensure the script runs relative to the project folder even when invoked as `powershell -File viral-flow\setup.ps1`.
#>
Set-Location -Path $PSScriptRoot

function Write-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Text
    )
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Fail {
    param(
        [Parameter(Mandatory = $true)][string]$Text
    )
    # Why: Avoid Unicode glyphs that can render as mojibake in some Windows consoles.
    Write-Host "[FAIL] $Text" -ForegroundColor Red
    exit 1
}

function Warn {
    param(
        [Parameter(Mandatory = $true)][string]$Text
    )
    # Why: Avoid Unicode glyphs that can render as mojibake in some Windows consoles.
    Write-Host "[WARN] $Text" -ForegroundColor Yellow
}

function Ok {
    param(
        [Parameter(Mandatory = $true)][string]$Text
    )
    # Why: Avoid Unicode glyphs that can render as mojibake in some Windows consoles.
    Write-Host "[OK] $Text" -ForegroundColor Green
}

function Get-PythonCommand {
    <#
    Why: Prefer Python 3.11. We probe multiple commands and validate version at runtime.
    #>
    function Test-IsPython311 {
        param(
            [Parameter(Mandatory = $true)][string]$Exe,
            # Why: Some candidates have no args (e.g., plain `python`).
            [Parameter()][AllowEmptyCollection()][string[]]$Args = @()
        )
        try {
            # Why: Suppress launcher "no suitable runtime" chatter while probing.
            $ver = & $Exe @Args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            return ($ver -eq "3.11")
        } catch {
            return $false
        }
    }

    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += @{ Exe = "py"; Args = @("-3.11") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += @{ Exe = "python"; Args = @() }
    }
    if (Get-Command python3 -ErrorAction SilentlyContinue) {
        $candidates += @{ Exe = "python3"; Args = @() }
    }

    foreach ($c in $candidates) {
        if (Test-IsPython311 -Exe $c.Exe -Args $c.Args) {
            return $c
        }
    }

    return $null
}

Write-Step "Setting up Viral-Flow (Windows)"

$pythonCmd = Get-PythonCommand
if (-not $pythonCmd) {
    Fail "Python 3.11 not found. Install Python 3.11 and ensure it is available as `py -3.11` or `python`."
}

Write-Step "Creating virtual environment (venv)"
# Why: Use splatted args so `py -3.11` works correctly.
& $pythonCmd.Exe @($pythonCmd.Args) -m venv venv

Write-Step "Activating venv"
$activatePath = Join-Path -Path (Get-Location) -ChildPath "venv\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Fail "venv activation script not found at: $activatePath"
}

# Why: Dot-source activation so pip/python commands use the venv.
. $activatePath

if (-not $SkipInstall) {
    Write-Step "Upgrading pip and installing requirements"
    python -m pip install --upgrade pip
    pip install -r requirements.txt
} else {
    Warn "Skipping pip install (per -SkipInstall)."
}

Write-Step "Checking FFmpeg"
$ffmpegOk = $true
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    $ffmpegOk = $false
}
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    $ffmpegOk = $false
}
if (-not $ffmpegOk) {
    # Why: Avoid PowerShell backtick escaping inside double-quoted strings.
    Fail "FFmpeg not found. Install FFmpeg and add it to PATH (both ffmpeg and ffprobe must be available)."
}
Ok "FFmpeg detected on PATH."

Write-Step "Checking Ollama"
try {
    # Why: Local-only check; does not send data anywhere except localhost.
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
    Ok "Ollama is reachable at http://localhost:11434"
} catch {
    # Why: Avoid PowerShell backtick escaping inside double-quoted strings.
    Warn "Ollama not reachable. Start it with: ollama serve"
}

Write-Step "Checking Piper binary"
$piperCandidates = @(
    (Join-Path -Path (Get-Location) -ChildPath "piper\piper.exe"),
    (Join-Path -Path (Get-Location) -ChildPath "piper\piper")
)
$piperFound = $false
foreach ($p in $piperCandidates) {
    if (Test-Path $p) {
        $piperFound = $true
        Ok "Piper found at: $p"
        break
    }
}
if (-not $piperFound) {
    # Why: Avoid PowerShell backtick parsing by not using backticks for "code formatting" in strings.
    Warn 'Piper binary not found in piper\. Place piper.exe (or piper) in the piper folder.'
    Warn "Download from: https://github.com/rhasspy/piper/releases"
}

Write-Step "Ensuring project directories exist"
New-Item -ItemType Directory -Force -Path "input\footage","output","temp","models" | Out-Null
Ok "Directories ready."

Write-Host ""
Ok "Setup complete."
Write-Host "Run: python src\ui_main.py"


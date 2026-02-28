# .docker\start_claude.ps1
# Fetches OAuth token from Windows Credential Manager and passes it to docker as an env var.

$ErrorActionPreference = "Stop"

function Get-GenericCredentialPassword {
    param(
        [Parameter(Mandatory = $true)]
        [string] $TargetName
    )

    $signature = @"
using System;
using System.Runtime.InteropServices;

public static class CredMan {
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern bool CredRead(string target, int type, int reservedFlag, out IntPtr credentialPtr);

    [DllImport("advapi32.dll", SetLastError = true)]
    public static extern bool CredFree(IntPtr buffer);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct CREDENTIAL {
        public uint Flags;
        public uint Type;
        public string TargetName;
        public string Comment;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
        public uint CredentialBlobSize;
        public IntPtr CredentialBlob;
        public uint Persist;
        public uint AttributeCount;
        public IntPtr Attributes;
        public string TargetAlias;
        public string UserName;
    }

    public const int CRED_TYPE_GENERIC = 1;
}
"@

    if (-not ("CredMan" -as [type])) {
        Add-Type -TypeDefinition $signature
    }

    $credPtr = [IntPtr]::Zero
    $ok = [CredMan]::CredRead($TargetName, [CredMan]::CRED_TYPE_GENERIC, 0, [ref] $credPtr)
    if (-not $ok) {
        $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "Credential '$TargetName' not found (CredRead failed, Win32Error=$err). Add it in Credential Manager first (cmdkey /generic:$TargetName /user:token /pass:YOUR_TOKEN)."
    }

    try {
        $cred = [Runtime.InteropServices.Marshal]::PtrToStructure($credPtr, [type] [CredMan+CREDENTIAL])

        if ($cred.CredentialBlobSize -le 0 -or $cred.CredentialBlob -eq [IntPtr]::Zero) {
            throw "Credential '$TargetName' has an empty secret."
        }

        # CredentialBlob is typically UTF-16LE text for generic credentials.
        $secret = [Runtime.InteropServices.Marshal]::PtrToStringUni($cred.CredentialBlob, $cred.CredentialBlobSize / 2)
        return $secret
    }
    finally {
        [void][CredMan]::CredFree($credPtr)
    }
}

# ---- Main ----

$TargetName = "CLAUDE_CODE_OAUTH_TOKEN"
$token = Get-GenericCredentialPassword -TargetName $TargetName

if ([string]::IsNullOrWhiteSpace($token)) {
    throw "Retrieved empty token from Credential Manager target '$TargetName'."
}

# Use forward slashes for Docker bind mounts (more robust with Docker Desktop)
$CurrentDir = (Get-Location).Path -replace '\\','/'

# Ensure .claude exists so the bind mount doesn't create oddities
$claudeDir = Join-Path (Get-Location).Path ".claude"
if (-not (Test-Path $claudeDir)) {
    New-Item -ItemType Directory -Path $claudeDir | Out-Null
}

$DockerArgs = @(
    "run"
    "--rm"
    "-it"
    "-v", "${CurrentDir}:/workspace"
    "-v", "${CurrentDir}/.claude:/home/agent/.claude"
    "-e", "CLAUDE_CODE_OAUTH_TOKEN=$token"
    "sklearn-lda-claude"
)

docker @DockerArgs

"""
Persistent shell session management.

Provides a way to keep a shell process alive and send commands to it,
accumulating state across multiple command executions. This is analogous
to VenvPool's persistent connections but for shell languages.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

ShellLanguage = Literal["bash", "zsh", "sh", "powershell"]


@dataclass
class ShellExecutionResult:
    """Result from executing a command in a shell session."""

    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None
    # Optional structured classification for callers (e.g., resource_limit).
    error_type: Optional[str] = None


class ShellSession:
    """
    A persistent shell session that maintains state across command executions.

    This is analogous to VenvPool's persistent connections but for shell languages.
    Commands are sent to a long-running shell subprocess, and state (environment
    variables, functions, aliases, working directory) persists between calls.

    Usage:
        session = ShellSession(language="bash")
        await session.start()
        result = await session.execute("export FOO=bar")
        result = await session.execute("echo $FOO")  # prints "bar"
        await session.close()
    """

    def __init__(
        self,
        language: ShellLanguage = "bash",
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        """
        Initialize a shell session.

        Args:
            language: Shell interpreter to use ("bash", "zsh", "sh", "powershell").
            env: Optional environment variables to set in the shell.
            cwd: Optional working directory to start in.
        """
        self.language = language
        self.initial_env = env
        self.initial_cwd = cwd
        self._process: Optional[asyncio.subprocess.Process] = None
        self._marker = f"__UNITY_SHELL_DONE_{uuid.uuid4().hex[:8]}__"
        self._started = False

    def _get_shell_command(self) -> List[str]:
        """Get the shell interpreter command for the configured language.

        We use non-interactive mode to avoid prompts and command echoing
        which would interfere with our marker-based output parsing.
        """
        commands = {
            # Non-interactive bash - no prompts, no rc files
            "bash": ["/bin/bash", "--norc", "--noprofile"],
            # Non-interactive zsh - no prompts, no rc files
            "zsh": ["/bin/zsh", "--no-rcs", "--no-globalrcs"],
            # POSIX sh - minimal shell
            "sh": ["/bin/sh"],
            # PowerShell - read from stdin
            "powershell": ["pwsh", "-NoProfile", "-NoLogo", "-Command", "-"],
        }
        if self.language not in commands:
            raise ValueError(f"Unsupported shell language: {self.language}")
        return commands[self.language]

    def _build_env(self) -> Optional[Dict[str, str]]:
        """Build the environment dict for the subprocess.

        Always sanitized via ``build_sandbox_env`` so no raw provider OAuth
        token is inherited and the localhost proxy endpoints are present.
        """
        from unify.provider_proxy.session import build_sandbox_env

        env = build_sandbox_env()
        if self.initial_env:
            env.update(self.initial_env)
        return env

    async def start(self) -> None:
        """
        Start the shell subprocess.

        Raises:
            RuntimeError: If the session is already started.
        """
        if self._started:
            raise RuntimeError("Session already started")

        shell_cmd = self._get_shell_command()

        self._process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            env=self._build_env(),
            cwd=self.initial_cwd,
        )
        self._started = True

    async def execute(
        self,
        command: str,
        *,
        timeout: Optional[float] = 30.0,
    ) -> ShellExecutionResult:
        """
        Execute a command in the persistent session.

        Uses a marker-based approach to detect command completion:
        1. Send the command
        2. Send an echo of a unique marker with the exit code
        3. Read output until we see the marker
        4. Parse exit code from marker line

        Args:
            command: The shell command to execute.
            timeout: Maximum time to wait for command completion (default 30s).
                     Set to None for no timeout.

        Returns:
            ShellExecutionResult with stdout, stderr, exit_code, and error.

        Raises:
            RuntimeError: If the session is not started.
        """
        if not self._started or self._process is None:
            raise RuntimeError("Session not started. Call start() first.")

        if self._process.returncode is not None:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=-1,
                error="Shell process has terminated",
            )

        # Construct command with completion marker
        # The marker line format: __MARKER__ <exit_code>
        # We use a two-marker approach to ensure complete output capture:
        # 1. Echo start marker to sync output stream
        # 2. Run the command
        # 3. Capture exit code
        # 4. Echo end marker with exit code
        # The start marker ensures any pending output is flushed before we
        # begin, and the end marker indicates command completion.
        start_marker = f"__UNITY_START_{self._marker[-8:]}__"

        if self.language == "powershell":
            # PowerShell uses different syntax
            # $? is boolean success, if $? { 0 } else { 1 } gives numeric exit
            wrapped_command = (
                f'Write-Host "{start_marker}"\n'
                f"{command}\n"
                f"$__unity_ec__ = if ($?) {{ 0 }} else {{ 1 }}\n"
                f'Write-Host "{self._marker} $__unity_ec__"\n'
            )
        else:
            # Bash/zsh/sh syntax
            wrapped_command = (
                f'echo "{start_marker}"\n'
                f"{command}\n"
                f"__unity_ec__=$?\n"
                f'echo "{self._marker} $__unity_ec__"\n'
            )

        try:
            self._process.stdin.write(wrapped_command.encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=-1,
                error=f"Failed to send command: {e}",
            )

        # Read until we see the marker
        stdout_lines: List[str] = []
        exit_code = 0

        start_marker = f"__UNITY_START_{self._marker[-8:]}__"
        found_start = False

        async def read_until_marker() -> ShellExecutionResult:
            nonlocal exit_code, found_start
            while True:
                try:
                    line = await self._process.stdout.readline()
                except Exception as e:
                    return ShellExecutionResult(
                        stdout="".join(stdout_lines),
                        stderr="",
                        exit_code=-1,
                        error=f"Error reading output: {e}",
                    )

                if not line:
                    # Process ended unexpectedly
                    return ShellExecutionResult(
                        stdout="".join(stdout_lines),
                        stderr="",
                        exit_code=-1,
                        error="Shell process terminated unexpectedly",
                    )

                decoded = line.decode()

                # Skip start marker line - it's just for synchronization
                if start_marker in decoded:
                    found_start = True
                    continue

                # Check if this line contains our end marker
                if self._marker in decoded:
                    # Parse exit code from marker line
                    # Format: "__MARKER__ <exit_code>"
                    parts = decoded.strip().split()
                    if len(parts) >= 2:
                        try:
                            exit_code = int(parts[-1])
                        except ValueError:
                            exit_code = 0
                    break

                # Only capture output after we've seen the start marker
                if found_start:
                    stdout_lines.append(decoded)

            return ShellExecutionResult(
                stdout="".join(stdout_lines),
                stderr="",  # Merged into stdout
                exit_code=exit_code,
                error=None,
            )

        # Apply timeout if specified
        if timeout is not None:
            try:
                return await asyncio.wait_for(read_until_marker(), timeout=timeout)
            except asyncio.TimeoutError:
                return ShellExecutionResult(
                    stdout="".join(stdout_lines),
                    stderr="",
                    exit_code=-1,
                    error=f"Command timed out after {timeout} seconds",
                )
        else:
            return await read_until_marker()

    async def close(self) -> None:
        """
        Terminate the shell session.

        Safe to call multiple times or on an unstarted session.
        """
        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._process = None
        self._started = False

    @property
    def is_running(self) -> bool:
        """Check if the session is active."""
        return (
            self._started
            and self._process is not None
            and self._process.returncode is None
        )

    async def snapshot_state(self) -> Dict[str, str]:
        """
        Capture the current shell state for later restoration.

        Returns a dict containing the serialized state that can be passed
        to restore_state() on a fresh session.

        Returns:
            Dict with keys: cwd, variables, functions, aliases, options
        """
        if not self.is_running:
            raise RuntimeError("Cannot snapshot state: session not running")

        if self.language == "bash":
            return await self._snapshot_bash_state()
        elif self.language == "zsh":
            return await self._snapshot_zsh_state()
        elif self.language == "sh":
            return await self._snapshot_sh_state()
        elif self.language == "powershell":
            return await self._snapshot_powershell_state()
        else:
            return await self._snapshot_minimal_state()

    async def _snapshot_bash_state(self) -> Dict[str, str]:
        """Capture bash-specific state."""
        state: Dict[str, str] = {}

        # Capture working directory
        result = await self.execute("pwd", timeout=5.0)
        state["cwd"] = result.stdout.strip() if result.error is None else ""

        # Capture shell variables (declare -p outputs variable declarations)
        # Filter out readonly and special variables that can't be restored
        result = await self.execute(
            "declare -p 2>/dev/null | grep -v '^declare -[a-z]*r' | "
            "grep -v '^declare -[a-z]* BASH' | "
            "grep -v '^declare -[a-z]* EUID' | "
            "grep -v '^declare -[a-z]* PPID' | "
            "grep -v '^declare -[a-z]* UID' | "
            "grep -v '^declare -[a-z]* GROUPS' | "
            "grep -v '^declare -[a-z]* SHELLOPTS' | "
            "grep -v '^declare -[a-z]* BASHOPTS' | "
            "grep -v '^declare -[a-z]* _='",
            timeout=10.0,
        )
        state["variables"] = result.stdout if result.error is None else ""

        # Capture functions
        result = await self.execute("declare -f 2>/dev/null", timeout=10.0)
        state["functions"] = result.stdout if result.error is None else ""

        # Capture aliases
        result = await self.execute("alias 2>/dev/null", timeout=5.0)
        state["aliases"] = result.stdout if result.error is None else ""

        # Capture shell options (shopt for bash-specific, set +o for POSIX)
        result = await self.execute("shopt -p 2>/dev/null", timeout=5.0)
        shopt_output = result.stdout if result.error is None else ""
        result = await self.execute("set +o 2>/dev/null", timeout=5.0)
        set_output = result.stdout if result.error is None else ""
        state["options"] = f"{shopt_output}\n{set_output}"

        return state

    async def _snapshot_zsh_state(self) -> Dict[str, str]:
        """Capture zsh-specific state."""
        state: Dict[str, str] = {}

        # Capture working directory
        result = await self.execute("pwd", timeout=5.0)
        state["cwd"] = result.stdout.strip() if result.error is None else ""

        # Capture variables using typeset (zsh equivalent of declare)
        result = await self.execute(
            "typeset -p 2>/dev/null | grep -v '^typeset -r'",
            timeout=10.0,
        )
        state["variables"] = result.stdout if result.error is None else ""

        # Capture functions
        result = await self.execute("typeset -f 2>/dev/null", timeout=10.0)
        state["functions"] = result.stdout if result.error is None else ""

        # Capture aliases
        result = await self.execute("alias 2>/dev/null", timeout=5.0)
        state["aliases"] = result.stdout if result.error is None else ""

        # Capture options
        result = await self.execute("setopt 2>/dev/null", timeout=5.0)
        state["options"] = result.stdout if result.error is None else ""

        return state

    async def _snapshot_sh_state(self) -> Dict[str, str]:
        """Capture POSIX sh state (limited capabilities).

        Note: POSIX sh has limited introspection compared to bash/zsh:
        - Only exported variables are captured (via `export -p`)
        - Shell-local variables (VAR=value without export) are NOT captured
        - Functions are not supported in snapshot/restore
        - Aliases may or may not work depending on sh implementation

        For full state persistence, use bash or zsh instead.
        """
        state: Dict[str, str] = {}

        # Capture working directory
        result = await self.execute("pwd", timeout=5.0)
        state["cwd"] = result.stdout.strip() if result.error is None else ""

        # Use export -p which gives properly quoted, executable output
        # Note: This only captures exported variables, not shell-local variables
        result = await self.execute("export -p 2>/dev/null", timeout=10.0)
        state["variables"] = result.stdout if result.error is None else ""

        # POSIX sh doesn't have a standard way to export functions
        state["functions"] = ""

        # Aliases may or may not be supported
        result = await self.execute("alias 2>/dev/null || true", timeout=5.0)
        state["aliases"] = result.stdout if result.error is None else ""

        # Shell options
        result = await self.execute("set +o 2>/dev/null", timeout=5.0)
        state["options"] = result.stdout if result.error is None else ""

        return state

    async def _snapshot_minimal_state(self) -> Dict[str, str]:
        """Capture minimal state for unsupported shells."""
        state: Dict[str, str] = {}

        result = await self.execute("pwd", timeout=5.0)
        state["cwd"] = result.stdout.strip() if result.error is None else ""
        state["variables"] = ""
        state["functions"] = ""
        state["aliases"] = ""
        state["options"] = ""

        return state

    async def _snapshot_powershell_state(self) -> Dict[str, str]:
        """Capture PowerShell-specific state.

        PowerShell state capture includes:
        - Current working directory (Get-Location)
        - Variables (Get-Variable, filtered to user-defined)
        - Functions (Get-ChildItem Function:)
        - Aliases (Get-Alias)

        Note: PowerShell variables are captured as assignment statements
        that can be re-executed.
        """
        state: Dict[str, str] = {}

        # Capture working directory
        result = await self.execute("(Get-Location).Path", timeout=5.0)
        state["cwd"] = result.stdout.strip() if result.error is None else ""

        # Capture variables - output as PowerShell assignment statements
        # Filter out automatic/readonly variables
        result = await self.execute(
            "Get-Variable | Where-Object { "
            "$_.Options -notmatch 'ReadOnly|Constant' -and "
            "$_.Name -notmatch '^(\\$|\\?|_|args|Error|ExecutionContext|"
            "false|HOME|Host|input|MyInvocation|null|PID|PROFILE|"
            "PSBoundParameters|PSCommandPath|PSCulture|PSScriptRoot|"
            "PSUICulture|PSVersionTable|PWD|ShellId|true)$' "
            "} | ForEach-Object { "
            "\"$\" + $_.Name + ' = ' + "
            "($_.Value | ConvertTo-Json -Compress -Depth 1 -ErrorAction SilentlyContinue) "
            "}",
            timeout=10.0,
        )
        state["variables"] = result.stdout if result.error is None else ""

        # Capture functions - output as function definitions
        result = await self.execute(
            "Get-ChildItem Function: | Where-Object { "
            "$_.Source -eq '' "  # User-defined functions have empty Source
            "} | ForEach-Object { "
            "'function ' + $_.Name + ' { ' + $_.Definition + ' }' "
            "}",
            timeout=10.0,
        )
        state["functions"] = result.stdout if result.error is None else ""

        # Capture aliases - output as Set-Alias commands
        result = await self.execute(
            "Get-Alias | Where-Object { "
            "$_.Options -notmatch 'ReadOnly|Constant' "
            "} | ForEach-Object { "
            "'Set-Alias -Name ' + $_.Name + ' -Value ' + $_.Definition "
            "}",
            timeout=10.0,
        )
        state["aliases"] = result.stdout if result.error is None else ""

        # PowerShell doesn't have shell options like bash
        state["options"] = ""

        return state

    async def restore_state(self, state: Dict[str, str]) -> ShellExecutionResult:
        """
        Restore previously captured shell state.

        This should be called on a freshly started session to restore
        state from a snapshot.

        Args:
            state: State dict from snapshot_state()

        Returns:
            ShellExecutionResult indicating success/failure of restoration
        """
        if not self.is_running:
            raise RuntimeError("Cannot restore state: session not running")

        if self.language == "bash":
            return await self._restore_bash_state(state)
        elif self.language == "zsh":
            return await self._restore_zsh_state(state)
        elif self.language == "sh":
            return await self._restore_sh_state(state)
        elif self.language == "powershell":
            return await self._restore_powershell_state(state)
        else:
            return await self._restore_minimal_state(state)

    async def _restore_bash_state(self, state: Dict[str, str]) -> ShellExecutionResult:
        """Restore bash state."""
        errors: List[str] = []

        # 1. Change to saved directory
        if state.get("cwd"):
            result = await self.execute(f'cd "{state["cwd"]}"', timeout=5.0)
            if result.error:
                errors.append(f"Failed to restore cwd: {result.error}")

        # 2. Restore shell options first (affects how other things are parsed)
        if state.get("options"):
            # Options are already in executable format from shopt -p and set +o
            result = await self.execute(state["options"], timeout=10.0)
            if result.error:
                errors.append(f"Failed to restore options: {result.error}")

        # 3. Restore variables (declare statements are executable)
        if state.get("variables"):
            result = await self.execute(state["variables"], timeout=30.0)
            if result.error:
                errors.append(f"Failed to restore variables: {result.error}")

        # 4. Restore functions
        if state.get("functions"):
            result = await self.execute(state["functions"], timeout=30.0)
            if result.error:
                errors.append(f"Failed to restore functions: {result.error}")

        # 5. Restore aliases
        if state.get("aliases"):
            # Alias output format is: alias name='value'
            result = await self.execute(state["aliases"], timeout=10.0)
            if result.error:
                errors.append(f"Failed to restore aliases: {result.error}")

        if errors:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                error="; ".join(errors),
            )

        return ShellExecutionResult(
            stdout="State restored successfully",
            stderr="",
            exit_code=0,
            error=None,
        )

    async def _restore_zsh_state(self, state: Dict[str, str]) -> ShellExecutionResult:
        """Restore zsh state."""
        errors: List[str] = []

        if state.get("cwd"):
            result = await self.execute(f'cd "{state["cwd"]}"', timeout=5.0)
            if result.error:
                errors.append(f"Failed to restore cwd: {result.error}")

        if state.get("variables"):
            result = await self.execute(state["variables"], timeout=30.0)
            if result.error:
                errors.append(f"Failed to restore variables: {result.error}")

        if state.get("functions"):
            result = await self.execute(state["functions"], timeout=30.0)
            if result.error:
                errors.append(f"Failed to restore functions: {result.error}")

        if state.get("aliases"):
            result = await self.execute(state["aliases"], timeout=10.0)
            if result.error:
                errors.append(f"Failed to restore aliases: {result.error}")

        if errors:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                error="; ".join(errors),
            )

        return ShellExecutionResult(
            stdout="State restored successfully",
            stderr="",
            exit_code=0,
            error=None,
        )

    async def _restore_sh_state(self, state: Dict[str, str]) -> ShellExecutionResult:
        """Restore POSIX sh state (limited).

        Note: Only exported variables and cwd are restored. Shell-local
        variables and functions are not supported for POSIX sh.
        """
        errors: List[str] = []

        if state.get("cwd"):
            result = await self.execute(f'cd "{state["cwd"]}"', timeout=5.0)
            if result.error:
                errors.append(f"Failed to restore cwd: {result.error}")

        # export -p output is executable (e.g., export VAR="value")
        if state.get("variables"):
            result = await self.execute(state["variables"], timeout=30.0)
            if result.error:
                errors.append(f"Failed to restore variables: {result.error}")

        # Aliases might work depending on sh implementation
        if state.get("aliases"):
            result = await self.execute(state["aliases"], timeout=10.0)
            if result.error:
                errors.append(f"Failed to restore aliases: {result.error}")

        if errors:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                error="; ".join(errors),
            )

        return ShellExecutionResult(
            stdout="State restored successfully",
            stderr="",
            exit_code=0,
            error=None,
        )

    async def _restore_powershell_state(
        self,
        state: Dict[str, str],
    ) -> ShellExecutionResult:
        """Restore PowerShell state."""
        errors: List[str] = []

        # 1. Change to saved directory
        if state.get("cwd"):
            result = await self.execute(
                f'Set-Location -Path "{state["cwd"]}"',
                timeout=5.0,
            )
            if result.error:
                errors.append(f"Failed to restore cwd: {result.error}")

        # 2. Restore variables
        if state.get("variables"):
            result = await self.execute(state["variables"], timeout=30.0)
            if result.error:
                errors.append(f"Failed to restore variables: {result.error}")

        # 3. Restore functions
        if state.get("functions"):
            result = await self.execute(state["functions"], timeout=30.0)
            if result.error:
                errors.append(f"Failed to restore functions: {result.error}")

        # 4. Restore aliases
        if state.get("aliases"):
            result = await self.execute(state["aliases"], timeout=10.0)
            if result.error:
                errors.append(f"Failed to restore aliases: {result.error}")

        if errors:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=1,
                error="; ".join(errors),
            )

        return ShellExecutionResult(
            stdout="State restored successfully",
            stderr="",
            exit_code=0,
            error=None,
        )

    async def _restore_minimal_state(
        self,
        state: Dict[str, str],
    ) -> ShellExecutionResult:
        """Restore minimal state (just cwd)."""
        if state.get("cwd"):
            result = await self.execute(f'cd "{state["cwd"]}"', timeout=5.0)
            if result.error:
                return result

        return ShellExecutionResult(
            stdout="State restored successfully",
            stderr="",
            exit_code=0,
            error=None,
        )

    async def __aenter__(self) -> "ShellSession":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

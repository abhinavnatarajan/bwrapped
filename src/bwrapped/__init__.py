import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Self

from xdg_base_dirs import (
    xdg_cache_home,
    xdg_config_dirs,
    xdg_config_home,
    xdg_data_dirs,
    xdg_data_home,
    xdg_runtime_dir,
    xdg_state_home,
)

STANDARD_BIN_DIRS = [
    "/usr/local/bin",
    "/usr/local/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/bin",
    "/sbin",
    "/opt/bin",
    "/var/lib/flatpak/exports/bin",
]

DANGEROUS_CWD_PATHS = {
    "/",
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/media",
    "/mnt",
    "/opt",
    "/proc",
    "/root",
    "/run",
    "/sbin",
    "/srv",
    "/sys",
    "/tmp",
    "/usr",
    "/var",
}

FILE_MODES: dict[str, int] = {
    f"{a[0]}{b[0]}{c[0]}": 0o100 * a[1] + 0o10 * b[1] + c[1]
    for (a, b, c) in product(
        [("---", 0), ("r--", 0o4), ("r-x", 0o5), ("rw-", 0o6), ("rwx", 0o7)], repeat=3
    )
}
DEFAULT_FILE_MODE = FILE_MODES["rw----r--"]
DEFAULT_DIRECTORY_MODE = FILE_MODES["rwx------"]


class DangerousWorkspaceError(Exception): ...


@dataclass(slots=True)
class SandboxOnlyMount:
    _dest: Path
    _mode: int

    def __init__(
        self, dest: os.PathLike | str, mode: int | str = DEFAULT_DIRECTORY_MODE
    ) -> None:
        self._dest = Path(dest)
        if isinstance(mode, str):
            self._mode = FILE_MODES.get(mode, DEFAULT_DIRECTORY_MODE)
        else:
            self._mode = mode if mode in FILE_MODES.values() else DEFAULT_DIRECTORY_MODE

    def dest(self) -> Path:
        """Destination of the mount."""
        return Path(self._dest)


class DirectoryMount(SandboxOnlyMount):
    def args(self) -> list[str]:
        """Return the bubblewrap arguments that create the sandbox directory."""
        return ["--perms", f"{self._mode:04o}", "--dir", str(self._dest)]

    def __repr__(self) -> str:
        return f"DirectoryMount(dest={self._dest},mode={self._mode:04o})"


class TmpfsMount(SandboxOnlyMount):
    def args(self) -> list[str]:
        """Return the bubblewrap arguments that mount the private tmpfs."""
        return ["--perms", f"{self._mode:04o}", "--tmpfs", str(self._dest)]

    def __repr__(self) -> str:
        return f"TmpfsMount(dest={self._dest},mode={self._mode:04o})"


@dataclass(slots=True)
class BindMount:
    _src: Path
    _dest: Path
    _ignore_missing: bool
    _read_only: bool

    def __init__(
        self, src: os.PathLike | str, dest: os.PathLike | str | None = None
    ) -> None:
        self._src = Path(src)
        self._dest = Path(dest or src)
        self._ignore_missing = False
        self._read_only = False

    def ro(self) -> Self:
        """Make the bind-mount read-only."""
        self._read_only = True
        return self

    def rw(self) -> Self:
        """Make the bind-mount writeable."""
        self._read_only = False
        return self

    def is_ro(self) -> bool:
        """Returns true if the bind is read-only."""
        return self._read_only

    def src(self) -> Path:
        """Source of the bind."""
        return Path(self._src)

    def dest(self) -> Path:
        """Destination of the bind."""
        return Path(self._dest)

    def ignore_missing(self, ignore: bool = True) -> Self:
        """Control whether the bind-mount is a noop if the source is inaccessible."""
        self._ignore_missing = ignore
        return self

    def is_try(self) -> bool:
        """Returns true if the bind should fail gracefully when the source is inaccessible."""
        return self._ignore_missing

    def args(self) -> list[str]:
        """Return the bubblewrap arguments for this bind mount."""
        option = "--ro-bind" if self._read_only else "--bind"
        if self._ignore_missing:
            option += "-try"
        if self._dest is None:
            raise RuntimeError("Bind destination was not initialized.")
        return [option, str(self._src), str(self._dest)]


type Mount = BindMount | DirectoryMount | TmpfsMount


@dataclass(slots=True)
class BWrapper:
    command: str
    command_args: list[str]
    workspace_dir: Path
    allow_dangerous_workspace: bool
    _env: dict[str, str]
    _mounts: list[Mount]
    _mounted_directories: set[str]
    _xdg_config_home: Path
    _xdg_cache_home: Path
    _xdg_data_home: Path
    _xdg_state_home: Path
    _xdg_config_dirs: list[Path]
    _xdg_data_dirs: list[Path]
    _home_dir: Path

    def __init__(
        self,
        command: str | None = None,
        command_args: list[str] | None = None,
        workspace_dir: os.PathLike | None = None,
        allow_dangerous_workspace=False,
    ) -> None:
        if not command:
            command = os.environ["SHELL"] or "sh"
        if command_args is None:
            command_args = []
        if not workspace_dir:
            workspace_dir = Path.cwd()
        self.command = command
        self.command_args = command_args
        self.workspace_dir = Path(workspace_dir).resolve(strict=True)
        if not self.workspace_dir.is_dir():
            raise NotADirectoryError(
                f"Working directory is not a directory: {self.workspace_dir}"
            )
        self.allow_dangerous_workspace = allow_dangerous_workspace
        self._env = {}
        self._mounts = []
        self._mounted_directories = set()
        self._xdg_config_home = xdg_config_home().resolve(strict=False)
        self._xdg_cache_home = xdg_cache_home().resolve(strict=False)
        self._xdg_state_home = xdg_state_home().resolve(strict=False)
        self._xdg_data_home = xdg_data_home().resolve(strict=False)
        self._xdg_config_dirs = xdg_config_dirs()
        self._xdg_data_dirs = xdg_data_dirs()
        self._home_dir = Path.home().resolve(strict=False)

        self._add_system_dirs()
        self._add_xdg_base_dirs()
        self._add_base_path_dirs()
        self._add_base_env_vars()
        self._add_workspace()

    def _add_system_dirs(self) -> Self:
        self._mounts.extend(
            [
                TmpfsMount("/run", FILE_MODES["rwxr-xr-x"]),
                BindMount("/bin").ro().ignore_missing(),
                BindMount("/sbin").ro().ignore_missing(),
                BindMount("/lib").ro().ignore_missing(),
                BindMount("/lib64").ro().ignore_missing(),
                BindMount("/opt").ro().ignore_missing(),
                BindMount("/usr").ro().ignore_missing(),
                BindMount("/etc/alternatives").ro().ignore_missing(),
                BindMount("/etc/ca-certificates").ro().ignore_missing(),
                BindMount("/etc/fonts").ro().ignore_missing(),
                BindMount("/etc/ld.so.conf.d").ro().ignore_missing(),
                BindMount("/etc/pki").ro().ignore_missing(),
                BindMount("/etc/profile.d").ro().ignore_missing(),
                BindMount("/etc/ssh/ssh_config.d").ro().ignore_missing(),
                BindMount("/etc/ssl").ro().ignore_missing(),
                BindMount("/etc/terminfo").ro().ignore_missing(),
                BindMount("/etc/xml").ro().ignore_missing(),
                BindMount("/etc/gai.conf").ro().ignore_missing(),
                BindMount("/etc/gitconfig").ro().ignore_missing(),
                BindMount("/etc/group").ro().ignore_missing(),
                BindMount("/etc/hosts").ro().ignore_missing(),
                BindMount("/etc/inputrc").ro().ignore_missing(),
                BindMount("/etc/ld.so.cache").ro().ignore_missing(),
                BindMount("/etc/ld.so.conf").ro().ignore_missing(),
                BindMount("/etc/localtime").ro().ignore_missing(),
                BindMount("/etc/mime.types").ro().ignore_missing(),
                BindMount("/etc/nsswitch.conf").ro().ignore_missing(),
                BindMount("/etc/os-release").ro().ignore_missing(),
                BindMount("/etc/passwd").ro().ignore_missing(),
                BindMount("/etc/protocols").ro().ignore_missing(),
                BindMount("/etc/resolv.conf").ro().ignore_missing(),
                BindMount("/etc/services").ro().ignore_missing(),
                BindMount("/etc/ssh/ssh_config").ro().ignore_missing(),
                BindMount("/etc/timezone").ro().ignore_missing(),
                BindMount("/var/lib/flatpak/exports").ro().ignore_missing(),
            ]
        )
        return self._add_tmp_roots()

    def _add_tmp_roots(self) -> Self:
        # The current environment equivalent for "/tmp"
        # Guaranteed to exist and be writeable, or be equal to the cwd
        host_tmp = tempfile.gettempdir()

        tmp_dir_root = Path(host_tmp).joinpath(f"bwrapped-{os.geteuid()}")
        tmp_dir_root.mkdir(exist_ok=True)
        self._mounts.append(BindMount(tmp_dir_root, Path("/tmp")))

        persistent_tmp_dir_root = Path("/var/tmp").joinpath(f"bwrapped-{os.geteuid()}")
        persistent_tmp_dir_root.mkdir(exist_ok=True)
        self._mounts.append(BindMount(persistent_tmp_dir_root, Path("/var/tmp")))
        return self

    def _add_base_env_vars(self) -> Self:
        vars = {
            "USER": os.getenv("USER"),
            "LOGNAME": os.getenv("LOGNAME"),
            "SHELL": os.getenv("SHELL") or "/usr/bin/sh",
            "LANG": os.getenv("LANG") or "C.UTF-8",
            "LC_ALL": os.getenv("LC_ALL") or "C.UTF-8",
            "TERM": os.getenv("TERM") or "xterm-256color",
            "TERM_PROGRAM": os.getenv("TERM_PROGRAM") or None,
            "TERM_PROGRAM_VERSION": os.getenv("TERM_PROGRAM_VERSION") or None,
        }
        self._env |= {name: value for name, value in vars.items() if value is not None}
        return self

    def _add_xdg_base_dirs(self) -> Self:
        # Note: Path.home() returns os.path.expanduser("~"), which expands $HOME if it is set.
        # Therefore the following is consistent.
        self._mounts.append(DirectoryMount(self._home_dir))
        self.set_env_vars({"HOME": str(self._home_dir)})

        self._mounts.append(DirectoryMount(self._xdg_cache_home))
        self.set_env_vars({"XDG_CACHE_HOME": str(self._xdg_cache_home)})

        self._mounts.append(DirectoryMount(self._xdg_state_home))
        self.set_env_vars({"XDG_STATE_HOME": str(self._xdg_state_home)})

        self._mounts.append(DirectoryMount(self._xdg_config_home))
        self.set_env_vars({"XDG_CONFIG_HOME": str(self._xdg_config_home)})

        self._mounts.append(DirectoryMount(self._xdg_data_home))
        self.set_env_vars({"XDG_DATA_HOME": str(self._xdg_data_home)})

        runtime_dir = xdg_runtime_dir()
        if runtime_dir is not None:
            self._mounts.append(TmpfsMount(runtime_dir))
            self.set_env_vars({"XDG_RUNTIME_DIR": str(runtime_dir)})

        self._mounts.extend(
            [BindMount(dir).ro().ignore_missing() for dir in self._xdg_config_dirs]
        )
        self.set_env_vars(
            {
                "XDG_CONFIG_DIRS": os.pathsep.join(
                    str(dir) for dir in self._xdg_config_dirs
                )
            }
        )

        self._mounts.extend(
            [BindMount(dir).ro().ignore_missing() for dir in self._xdg_data_dirs]
        )
        self.set_env_vars(
            {"XDG_DATA_DIRS": os.pathsep.join(str(dir) for dir in self._xdg_data_dirs)}
        )

        return self

    def _add_base_path_dirs(self) -> Self:
        if "PATH" not in os.environ:
            return self

        user_bin_dirs: list[str] = [
            str(self._home_dir.joinpath(".local", "bin")),
            str(self._home_dir.joinpath("opt", "bin")),
            str(self._xdg_data_home.joinpath("flatpak", "exports", "bin")),
        ]
        bin_dirs: list[str] = user_bin_dirs + STANDARD_BIN_DIRS
        selected: list[str] = []  # actual directories to mount

        for value in os.environ["PATH"].split(os.pathsep):
            if not value or not Path(value).is_absolute():
                continue
            try:
                normalized = Path(value).absolute()
            except OSError:
                continue
            if str(normalized) not in bin_dirs or not normalized.is_dir():
                continue
            selected.append(str(normalized))
            if normalized in user_bin_dirs:
                self._mounts.append(BindMount(normalized).ro())

        self.set_env_vars({"PATH": os.pathsep.join(selected)})
        return self

    def _add_workspace(self) -> Self:
        self.workspace_dir = self.workspace_dir
        effective_parent_mount: Mount | None = None
        for mount in self._mounts[-1::-1]:
            if self.workspace_dir.is_relative_to(mount.dest().resolve()):
                effective_parent_mount = mount
                break

        if (
            isinstance(effective_parent_mount, BindMount)
            and not effective_parent_mount.is_ro()
        ):
            return self

        workspace_is_dangerous = True
        if (
            effective_parent_mount is None
            and all(
                not self.workspace_dir.is_relative_to(Path(path).resolve())
                for path in DANGEROUS_CWD_PATHS
            )
            or (
                effective_parent_mount is not None
                and effective_parent_mount.dest() == self._home_dir
                and self.workspace_dir != self._home_dir
            )
        ):
            workspace_is_dangerous = False

        if workspace_is_dangerous and not self.allow_dangerous_workspace:
            raise DangerousWorkspaceError

        self._mounts.append(BindMount(self.workspace_dir).ignore_missing())
        return self

    def _mount_args(self) -> list[str]:
        args: list[str] = []
        for mount in self._mounts:
            args.extend(mount.args())
        return args

    def _env_args(self) -> list[str]:
        args: list[str] = []
        for var, value in self._env.items():
            args.extend(["--setenv", var, value])
        return args

    @staticmethod
    def _base_args() -> list[str]:
        return [
            "--unshare-all",
            "--unshare-user",
            "--unshare-uts",
            "--disable-userns",
            "--die-with-parent",
            "--hostname",
            "sandbox",
            "--share-net",
            "--clearenv",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]

    def bwrapped_command(self) -> list[str]:
        """Build the bubblewrap command prefix for the configured sandbox."""
        executable = get_executable(self.command)
        return [
            "bwrap",
            *self._base_args(),
            *self._mount_args(),
            *self._env_args(),
            "--chdir",
            str(self.workspace_dir),
            executable,
            *self.command_args,
        ]

    def set_env_vars(self, variables: dict[str, str]) -> Self:
        """Add or replace environment variables in the sandbox."""
        self._env |= variables
        return self

    def expose_application_xdg_dirs(self, app: str) -> Self:
        """Expose host xdg directories for a specific application inside the sandbox."""
        for xdg_dir, ro in [
            (self._xdg_config_home, True),
            (self._xdg_state_home, False),
            (self._xdg_cache_home, False),
            (self._xdg_data_home, False),
        ]:
            dir = xdg_dir.joinpath(app)
            if dir.is_dir():
                mount = BindMount(dir)
                if ro:
                    mount.ro()
                self._mounts.append(mount)

        return self

    def add_to_path(self, path: os.PathLike | str) -> Self:
        # TODO: make this safe
        if "PATH" not in os.environ:
            return self

        if not path:
            return self

        if not isinstance(path, Path):
            path = Path(path)

        path = path.resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError("Expected a directory.")

        self._mounts.append(BindMount(path).ro())
        new_PATH = os.pathsep.join([str(path), self._env["PATH"]])
        self.set_env_vars({"PATH": new_PATH})

        return self


def get_executable(cmd: str) -> str:
    """Resolve an executable to an absolute path or raise an error."""
    path = shutil.which(cmd)
    if path is None:
        raise RuntimeError(f"Could not find '{cmd}'.")
    return str(Path(path))


@dataclass(slots=True)
class WrapperArgs:
    command: str | None = None
    command_args: list[str] = field(default_factory=list)
    workspace: str | None = None
    allow_dangerous_workspace: bool = False
    show_help: bool = False


def parse_wrapper_args(args: Sequence[str]) -> WrapperArgs:
    """Separate wrapper options from arguments intended for OpenCode."""
    parsed_args = WrapperArgs()

    class InvalidCliArgError(Exception): ...

    class MissingCommandError(Exception): ...

    try:
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == "--":
                if index + 1 < len(args):
                    parsed_args.command = args[index + 1]
                    parsed_args.command_args.extend(args[index + 2 :])
                    break
                else:
                    raise MissingCommandError("Missing command.")
            elif arg.startswith("--workspace="):
                parsed_args.workspace = os.path.expanduser(
                    arg.removeprefix("--workspace=")
                )
            elif arg == "--allow-dangerous-workspace":
                parsed_args.allow_dangerous_workspace = True
            elif arg == "--help" or arg == "-h":
                parsed_args.show_help = True
            elif arg == "--verbose" or arg == "-v":
                logger = logging.getLogger()
                logger.setLevel(logging.DEBUG)
                stderr_handler = logging.StreamHandler(sys.stderr)
                stderr_handler.setLevel(logging.DEBUG)
                logger.addHandler(stderr_handler)
            else:
                raise InvalidCliArgError(arg)
                break
            index += 1
        if not parsed_args.command and not parsed_args.show_help:
            raise MissingCommandError("Missing command.")

    except InvalidCliArgError as e:
        print(f"Invalid CLI argument: {e}")
        parsed_args.show_help = True
    except MissingCommandError as e:
        print(e)
        parsed_args.show_help = True

    return parsed_args


def _wrapper_help() -> str:
    return """Usage: bwrapped [WRAPPER_OPTIONS] [--] COMMAND [COMMAND_ARGS...]

Run a command inside the bubblewrap sandbox with sensible defaults, and forward other arguments to the program.

Wrapper options:
  --workspace=<workspace directory> Set the working directory for the sandbox child process.
  -h, --help                        Show this help message and exit.
  -v, --verbose                     Enable debug logging on standard error.
      --allow-dangerous-workspace   Allow a broad or sensitive working-directory bind.
  --                                Stop parsing wrapper options and forward all remaining
                                    arguments to the command.
"""


def main() -> int:
    """Build and execute the sandboxed OpenCode command."""
    wrapper_args = parse_wrapper_args(sys.argv[1:])
    logger = logging.getLogger(__name__)
    logger.debug(wrapper_args)
    if wrapper_args.show_help:
        print(_wrapper_help())
        return 0

    try:
        command = BWrapper(
            command=wrapper_args.command,
            workspace_dir=Path(wrapper_args.workspace)
            if wrapper_args.workspace
            else None,
            command_args=wrapper_args.command_args,
            allow_dangerous_workspace=wrapper_args.allow_dangerous_workspace,
        ).bwrapped_command()
        logging.getLogger().debug("Command:\n" + " ".join(command))
    except DangerousWorkspaceError:
        raise RuntimeError("Please run with --allow-dangerous-workspace")

    os.execvp(command[0], command[1:])


if __name__ == "__main__":
    raise SystemExit(main())

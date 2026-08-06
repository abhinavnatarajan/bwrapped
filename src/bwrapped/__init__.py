"""Construct and run commands in a Linux Bubblewrap sandbox.

The mount classes describe Bubblewrap filesystem arguments, while :class:`BWrapper`
assembles an isolated environment around a command and workspace. The :func:`main`
entry point provides the ``bwrapped`` command-line interface and replaces the current
process with Bubblewrap unless invoked in dry-run mode.
"""

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from itertools import product
from optparse import OptionParser
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
		[("---", 0), ("r--", 0o4), ("r-x", 0o5), ("rw-", 0o6), ("rwx", 0o7)],
		repeat=3,
	)
}
DEFAULT_FILE_MODE = "rw----r--"
DEFAULT_DIRECTORY_MODE = "rwx------"


class DangerousWorkspaceError(Exception):
	"""Indicate that a workspace would require a dangerous writable bind.

	A workspace already covered by a writable mount is accepted. Otherwise, BWrapper
	raises this exception for broad or sensitive locations unless dangerous workspaces
	are explicitly allowed.
	"""


class SandboxOnlyMount:
	"""Represent a destination-only directory or tmpfs mount in the sandbox."""

	__slots__ = ("_dest", "_mode")
	_dest: Path
	_mode: int

	def __init__(self, dest: os.PathLike | str, mode: int | str = DEFAULT_DIRECTORY_MODE) -> None:
		"""Configure a destination-only mount and its permission mode.

		Args:
			dest: Path at which Bubblewrap creates the mount in the sandbox.
			mode: Symbolic mode from ``FILE_MODES`` or an integer contained in its values.
				Unrecognized modes, including a symbolic mode whose value is zero, silently
				fall back to ``DEFAULT_DIRECTORY_MODE``.

		This constructor only records the mount configuration; it does not invoke
		Bubblewrap or create anything on the host.
		"""
		self._dest = Path(dest)
		self._mode = FILE_MODES[DEFAULT_DIRECTORY_MODE]
		if isinstance(mode, str) and (mode_int := FILE_MODES.get(mode)):
			self._mode = mode_int
		elif isinstance(mode, int) and mode in FILE_MODES.values():
			self._mode = mode

	def dest(self) -> Path:
		"""Return the configured sandbox destination as a path."""
		return Path(self._dest)


class DirectoryMount(SandboxOnlyMount):
	"""Represent an ordinary directory created in the sandbox filesystem."""

	def args(self) -> list[str]:
		"""Return ``--perms MODE --dir DEST`` arguments for Bubblewrap.

		The mode is rendered as a four-digit octal value. The resulting directory
		contains no host files and exists only as part of the sandbox filesystem.
		"""
		return ["--perms", f"{self._mode:04o}", "--dir", str(self._dest)]

	def __repr__(self) -> str:
		return f"DirectoryMount(dest={self._dest},mode={self._mode:04o})"


class TmpfsMount(SandboxOnlyMount):
	"""Represent an empty, memory-backed tmpfs mounted in the sandbox."""

	def args(self) -> list[str]:
		"""Return ``--perms MODE --tmpfs DEST`` arguments for Bubblewrap.

		The mode is rendered as a four-digit octal value.
		"""
		return ["--perms", f"{self._mode:04o}", "--tmpfs", str(self._dest)]

	def __repr__(self) -> str:
		return f"TmpfsMount(dest={self._dest},mode={self._mode:04o})"


@dataclass(slots=True)
class BindMount:
	"""Represent a host path bound to a destination in the sandbox."""

	_src: Path
	_dest: Path
	_ignore_missing: bool
	_read_only: bool

	def __init__(self, src: os.PathLike | str, dest: os.PathLike | str | None = None) -> None:
		"""Configure a writable host-to-sandbox bind mount.

		Args:
			src: Host path to expose in the sandbox.
			dest: Sandbox destination. An omitted or otherwise false value uses ``src``.

		The source is not checked for existence or restricted to a directory. This
		constructor only records configuration; Bubblewrap performs the eventual mount.
		Missing sources are errors by default unless :meth:`ignore_missing` is enabled.
		"""
		self._src = Path(src)
		self._dest = Path(dest or src)
		self._ignore_missing = False
		self._read_only = False

	def ro(self) -> Self:
		"""Make this bind mount read-only and return it for method chaining."""
		self._read_only = True
		return self

	def rw(self) -> Self:
		"""Make this bind mount writable and return it for method chaining."""
		self._read_only = False
		return self

	def is_ro(self) -> bool:
		"""Return whether this mount emits a read-only bind option."""
		return self._read_only

	def src(self) -> Path:
		"""Return the configured host source as a path."""
		return Path(self._src)

	def dest(self) -> Path:
		"""Return the configured sandbox destination as a path."""
		return Path(self._dest)

	def ignore_missing(self, *, ignore: bool = True) -> Self:
		"""Select whether to emit a Bubblewrap try-bind option.

		Args:
			ignore: If true, use ``--bind-try`` or ``--ro-bind-try`` instead of the
				corresponding regular bind option.

		Returns:
			This mount, for method chaining.
		"""
		self._ignore_missing = ignore
		return self

	def is_try(self) -> bool:
		"""Return whether this mount emits a Bubblewrap try-bind option."""
		return self._ignore_missing

	def args(self) -> list[str]:
		"""Return the Bubblewrap option, host source, and sandbox destination.

		The option is ``--bind`` or ``--ro-bind`` according to the access mode and
		gains the ``-try`` suffix when missing sources should be ignored.

		Raises:
			RuntimeError: If the mount's destination was not initialized.
		"""
		option = "--ro-bind" if self._read_only else "--bind"
		if self._ignore_missing:
			option += "-try"
		if self._dest is None:
			err_msg = "Bind destination was not initialized."
			raise RuntimeError(err_msg)
		return [option, str(self._src), str(self._dest)]


type Mount = BindMount | DirectoryMount | TmpfsMount


@dataclass(slots=True)
class BWrapper:
	"""Build a Bubblewrap command around a host command and workspace.

	Initialization configures namespace isolation, system and XDG mounts, temporary
	directories, environment variables, the executable search path, and workspace
	access. Constructing an instance does not run Bubblewrap.
	"""

	command: str
	command_args: tuple[str, ...]
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
		*command_with_args: str,
		workspace_dir: os.PathLike | None = None,
		allow_dangerous_workspace: bool = False,
	) -> None:
		"""Configure a sandbox command, workspace, mounts, and environment.

		Args:
			*command_with_args: Command followed by its arguments. If empty, use the
				value of ``SHELL``, or ``sh`` when that value is empty.
			workspace_dir: Existing directory in which the sandboxed command starts.
				Defaults to the current directory and is resolved to a canonical path.
			allow_dangerous_workspace: Allow a new writable bind for an overly broad or
				sensitive workspace.

		Raises:
			KeyError: If ``SHELL`` is absent from the host environment.
			FileNotFoundError: If ``workspace_dir`` does not exist.
			NotADirectoryError: If ``workspace_dir`` is not a directory.
			DangerousWorkspaceError: If the workspace requires a dangerous writable bind
				and ``allow_dangerous_workspace`` is false.
			OSError: If a path cannot be resolved or a per-user host temporary directory
				cannot be created.

		The setup creates ``bwrapped-UID`` directories below the host temporary directory
		and ``/var/tmp`` when needed. It only assembles configuration and does not invoke
		Bubblewrap.
		"""
		command = os.environ["SHELL"] or "sh"
		if len(command_with_args) > 0:
			command = command_with_args[0]
		command_args = command_with_args[1:]
		if not workspace_dir:
			workspace_dir = Path.cwd()
		self.command = command
		self.command_args = command_args
		self.workspace_dir = Path(workspace_dir).resolve(strict=True)
		if not self.workspace_dir.is_dir():
			err_msg = f"Invalid workspace directory: {self.workspace_dir}"
			raise NotADirectoryError(err_msg)
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
		"""Add isolated runtime storage and optional read-only system mounts."""
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
			],
		)
		return self._add_tmp_roots()

	def _add_tmp_roots(self) -> Self:
		"""Create and bind per-user host directories for sandbox temporary storage."""
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
		"""Copy selected host variables and add defaults to the sandbox environment."""
		variables = {
			"USER": os.getenv("USER"),
			"LOGNAME": os.getenv("LOGNAME"),
			"SHELL": os.getenv("SHELL") or "/usr/bin/sh",
			"LANG": os.getenv("LANG") or "C.UTF-8",
			"LC_ALL": os.getenv("LC_ALL") or "C.UTF-8",
			"TERM": os.getenv("TERM") or "xterm-256color",
			"TERM_PROGRAM": os.getenv("TERM_PROGRAM") or None,
			"TERM_PROGRAM_VERSION": os.getenv("TERM_PROGRAM_VERSION") or None,
		}
		self._env |= {name: value for name, value in variables.items() if value is not None}
		return self

	def _add_xdg_base_dirs(self) -> Self:
		"""Configure sandbox directories, mounts, and variables for the XDG layout."""
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
			[BindMount(directory).ro().ignore_missing() for directory in self._xdg_config_dirs],
		)
		self.set_env_vars(
			{
				"XDG_CONFIG_DIRS": os.pathsep.join(
					str(directory) for directory in self._xdg_config_dirs
				),
			},
		)

		self._mounts.extend(
			[BindMount(directory).ro().ignore_missing() for directory in self._xdg_data_dirs],
		)
		self.set_env_vars(
			{"XDG_DATA_DIRS": os.pathsep.join(str(directory) for directory in self._xdg_data_dirs)},
		)

		return self

	def _add_base_path_dirs(self) -> Self:
		"""Populate sandbox ``PATH`` with recognized, existing host directories."""
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
		"""Ensure the workspace is covered by an allowed writable mount.

		Returns:
			This wrapper after retaining an existing writable mount or adding a new one.

		Raises:
			DangerousWorkspaceError: If a new bind would expose a broad or sensitive
				workspace and dangerous workspaces are not allowed.
		"""
		self.workspace_dir = self.workspace_dir
		effective_parent_mount: Mount | None = None
		for mount in self._mounts[-1::-1]:
			if self.workspace_dir.is_relative_to(mount.dest().resolve()):
				effective_parent_mount = mount
				break

		if isinstance(effective_parent_mount, BindMount) and not effective_parent_mount.is_ro():
			return self

		workspace_is_dangerous = True
		if (
			effective_parent_mount is None
			and all(
				not self.workspace_dir.is_relative_to(Path(path).resolve())
				for path in DANGEROUS_CWD_PATHS
			)
		) or (
			effective_parent_mount is not None
			and effective_parent_mount.dest() == self._home_dir
			and self.workspace_dir != self._home_dir
		):
			workspace_is_dangerous = False

		if workspace_is_dangerous and not self.allow_dangerous_workspace:
			raise DangerousWorkspaceError

		self._mounts.append(BindMount(self.workspace_dir).ignore_missing())
		return self

	def _mount_args(self) -> list[str]:
		"""Return all configured mount arguments in insertion order."""
		args: list[str] = []
		for mount in self._mounts:
			args.extend(mount.args())
		return args

	def _env_args(self) -> list[str]:
		"""Return ``--setenv`` arguments for the configured sandbox environment."""
		args: list[str] = []
		for var, value in self._env.items():
			args.extend(["--setenv", var, value])
		return args

	@staticmethod
	def _base_args() -> list[str]:
		"""Return the fixed namespace, process, network, device, and proc arguments."""
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
		"""Build the complete Bubblewrap command argument vector.

		The result contains the ``bwrap`` executable, isolation settings, mounts,
		environment, workspace, located command path, and original command arguments.

		Returns:
			Arguments suitable for starting Bubblewrap.

		Raises:
			RuntimeError: If the configured command cannot be found.
		"""
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
		"""Update the configured sandbox environment and return this wrapper.

		Args:
			variables: Names and values to add. Existing configured names are replaced.

		This method does not modify the host's ``os.environ`` mapping.
		"""
		self._env |= variables
		return self

	def expose_application_xdg_dirs(self, app: str) -> Self:
		"""Expose existing host XDG directories for an application.

		Args:
			app: Path joined to each configured XDG home to locate application data.

		Returns:
			This wrapper after adding mounts for directories that exist. Configuration is
			mounted read-only; state, cache, and data are mounted writable.
		"""
		for xdg_dir, ro in [
			(self._xdg_config_home, True),
			(self._xdg_state_home, False),
			(self._xdg_cache_home, False),
			(self._xdg_data_home, False),
		]:
			directory = xdg_dir.joinpath(app)
			if directory.is_dir():
				mount = BindMount(directory)
				if ro:
					mount.ro()
				self._mounts.append(mount)

		return self

	def add_to_path(self, path: os.PathLike | str) -> Self:
		"""Prepend an existing host directory to the sandbox executable path.

		Args:
			path: Directory to expose through a read-only bind and prepend to ``PATH``.

		Returns:
			This wrapper. It is returned unchanged when the host has no ``PATH`` or
			``path`` is false.

		Raises:
			FileNotFoundError: If ``path`` does not exist.
			NotADirectoryError: If ``path`` is not a directory.
			OSError: If ``path`` cannot be resolved.
		"""
		if "PATH" not in os.environ:
			return self

		if not path:
			return self

		if not isinstance(path, Path):
			path = Path(path)

		path = path.resolve(strict=True)
		if not path.is_dir():
			err_msg = "Expected a directory."
			raise NotADirectoryError(err_msg)

		self._mounts.append(BindMount(path).ro())
		new_path = os.pathsep.join([str(path), self._env["PATH"]])
		self.set_env_vars({"PATH": new_path})

		return self


def get_executable(cmd: str) -> str:
	"""Locate an executable using :func:`shutil.which`.

	Args:
		cmd: Executable name or path to locate using the host environment.

	Returns:
		The path reported by :func:`shutil.which`, resolved to an absolute path.
		Symlinks are resolved and '..' components in the path are normalized.

	Raises:
		RuntimeError: If no executable is found.
	"""
	path = shutil.which(cmd)
	if path is None:
		err_msg = f"Could not find '{cmd}'."
		raise RuntimeError(err_msg)
	return str(Path(path).resolve(strict=True))


def main() -> int:
	"""Parse CLI arguments and run the requested command through Bubblewrap.

	The CLI accepts verbose and dry-run output, a workspace override, and the
	``--allow-dangerous_workspace`` override. Dry-run mode writes the generated command
	to standard output without a trailing newline and returns zero. Otherwise this
	function calls :func:`os.execvp`; successful execution replaces the current process
	and does not return. Verbose mode writes the command before that replacement.

	Returns:
		Zero after a dry run, or if process replacement unexpectedly returns.

	Raises:
		SystemExit: If option parsing fails or no command is supplied.
		KeyError: If ``SHELL`` is absent from the host environment.
		RuntimeError: If the workspace is rejected or the command cannot be found.
		OSError: If sandbox setup or process replacement fails.
	"""
	argparser = OptionParser()
	argparser.disable_interspersed_args()
	argparser.add_option("-v", "--verbose", action="store_true", help="Show the generated command")
	argparser.add_option(
		"--allow-dangerous_workspace",
		action="store_true",
		help="Allow a broad or sensitive working-directory bind.",
	)
	argparser.add_option(
		"--workspace",
		type="string",
		help="Set the working directory for the sandbox child process.",
	)
	argparser.add_option(
		"--dry-run",
		action="store_true",
		help="Do not run the command. Implies --verbose.",
	)

	(options, command_with_args) = argparser.parse_args()
	if len(command_with_args) == 0:
		argparser.error("No command specified.")
		return 1

	try:
		command = BWrapper(
			*command_with_args,
			workspace_dir=Path(options.workspace) if options.workspace else None,
			allow_dangerous_workspace=options.allow_dangerous_workspace,
		).bwrapped_command()

	except DangerousWorkspaceError as err:
		err_msg = "Please run with --allow-dangerous-workspace"
		raise RuntimeError(err_msg) from err

	if options.dry_run or options.verbose:
		sys.stdout.write(" ".join(command))
	if not options.dry_run:
		os.execvp(command[0], command[1:])
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

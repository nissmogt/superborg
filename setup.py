"""Packaging hook: compile the C engine into superborg/_lib/ whenever the package is built.

Metadata lives in pyproject.toml. The engine is plain C11 with no dependencies, so it is
compiled directly with the system C compiler (no Makefile or CMake needed):

    pip install .                        # optimized for this machine (-march=native)
    SUPERBORG_PORTABLE=1 pip wheel .     # baseline ISA, for wheels that run on other machines
    CC=clang pip install .               # any C11 compiler

The library is loaded through ctypes, so the wheel is tagged py3-none-<platform>.
"""
from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

ROOT = os.path.dirname(os.path.abspath(__file__))


def _lib_name() -> str:
    if sys.platform == "darwin":
        return "libborg.dylib"
    if sys.platform == "win32":
        return "borg.dll"
    return "libborg.so"


def _compiler() -> list[str]:
    candidates = [os.environ.get("CC"), sysconfig.get_config_var("CC"), "cc", "gcc", "clang"]
    for c in candidates:
        if c and shutil.which(shlex.split(c)[0]):
            return shlex.split(c)
    raise RuntimeError("superborg needs a C compiler to build its engine (set CC)")


def compile_engine(out_dir: str) -> str:
    src_dir = os.path.join(ROOT, "engine")
    sources = sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir)
                     if f.endswith(".c") and f != "main.c")  # main.c is the standalone CLI
    if not sources:
        raise RuntimeError(f"no engine sources in {src_dir}")
    flags = ["-O3", "-std=c11", "-fPIC", "-fvisibility=hidden", "-D_GNU_SOURCE", "-D_DEFAULT_SOURCE"]
    libs = ["-lm"]
    if sys.platform != "win32":
        flags.append("-pthread")
        libs.append("-pthread")
    if os.environ.get("SUPERBORG_PORTABLE", "0") in ("", "0"):
        arm = platform.machine().lower() in ("arm64", "aarch64")
        flags.append("-mcpu=native" if arm else "-march=native")
    shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, _lib_name())
    cmd = _compiler() + flags + shared + ["-o", out] + sources + libs
    print("compiling the borg engine:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)
    return out


class BuildPyWithEngine(build_py):
    """build_py that also compiles libborg next to the Python sources."""

    def run(self):
        super().run()
        if getattr(self, "editable_mode", False):
            # Editable installs import from the source tree. The binding looks in superborg/_lib
            # before build/, so a library compiled into _lib would shadow every later `make`;
            # build/ (where `make` writes) keeps one library for both workflows.
            compile_engine(os.path.join(ROOT, "build"))
        else:
            compile_engine(os.path.join(self.build_lib, "superborg", "_lib"))


class BinaryDistribution(Distribution):
    """Marks the distribution as platform-specific so wheels get a platform tag."""

    def has_ext_modules(self):
        return True


cmdclass = {"build_py": BuildPyWithEngine}

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # setuptools < 70.1
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        _bdist_wheel = None

if _bdist_wheel is not None:
    class BdistWheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            # no Python C-API is used (ctypes), so one wheel serves every Python 3 version
            _, _, plat = super().get_tag()
            return "py3", "none", plat

    cmdclass["bdist_wheel"] = BdistWheel

setup(cmdclass=cmdclass, distclass=BinaryDistribution)

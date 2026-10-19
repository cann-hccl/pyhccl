# -*- coding: utf-8 -*-
"""pyhccl 扩展的构建脚本（参考 PyTorch 的 pybind11 扩展构建方式）。

本工程只做 ABI 绑定：不对 libhccl.so / libhcomm.so 做编译期链接，而是在运行时
通过 dlopen 延迟加载（见 src/csrc/loader.cpp）。因此在无 CANN 环境的 CI 上也能
正常构建出 wheel，只是调用需要真实库的功能时会抛出 RuntimeError。

用法：
    pip install .            # 构建并安装
    pip install -e .         # 开发模式（可编辑安装）
"""

import platform
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# 编译选项：C++17 是 pybind11 2.13 对较新 Python 的最低要求。
extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
# 非 Windows 平台追加优化与符号可见性控制。
if sys.platform != "win32":
    extra_compile_args += ["-O2", "-fvisibility=hidden"]

libraries = []
if sys.platform.startswith("linux"):
    # dlopen/dlsym 在旧 glibc 上位于 libdl 中，显式链接可保证跨发行版兼容。
    libraries.append("dl")

ext_modules = [
    Pybind11Extension(
        "pyhccl._C",
        sources=[
            "src/csrc/bindings.cpp",
            "src/csrc/loader.cpp",
        ],
        include_dirs=["src/csrc"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        libraries=libraries,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

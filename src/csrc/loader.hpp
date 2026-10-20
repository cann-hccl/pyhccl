// -*- coding: utf-8 -*-
// 动态库加载器：运行时通过 dlopen/dlsym 延迟解析 hccl/hcomm 的导出符号。
//
// 设计目标：pyhccl 的 wheel 可以在没有 CANN 运行库的机器上导入（用于查看常量、
// 枚举、数据结构等），只有在真正调用需要原生库的功能时才报错。

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyhccl
{

// 通过 dlopen 加载原生通信库。候选名称顺序：
//   1. 环境变量 PYHCCL_LIBRARY 指定的确切路径；
//   2. 环境变量 PYHCCL_LIBRARY_PATH 目录下的 libhccl.so / libhcomm.so；
//   3. 系统默认搜索路径下的常用库名。
// 返回成功加载的库数量；多次调用幂等。
int load_native_libraries();

// 按符号名解析函数指针；找不到返回 nullptr。
// 会依次在所有已加载的原生库以及全局符号表（RTLD_DEFAULT）中查找。
void* resolve_symbol(const char* name);

// 返回是否至少加载了一个原生库。
bool has_native_library();

} // namespace pyhccl

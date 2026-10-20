// -*- coding: utf-8 -*-
// 动态库加载器实现（Linux 使用 dlfcn.h，其余平台退化为空实现）。
//
// 注意：日志统一使用英文（符合工程约定），注释使用中文。

#include "loader.hpp"

#include <cstdlib>
#include <mutex>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace pyhccl
{

namespace
{

// 已 dlopen 的库句柄（不调用 dlclose，随进程生命周期持有）。
std::vector<void*>& handles()
{
    static std::vector<void*> h;
    return h;
}

// 保护 handles() 的互斥量：多个线程在“首次调用”同时触发延迟加载时，
// 会并发修改 handles() 向量，属未定义行为，故所有读写均需加锁。
std::mutex& loader_mutex()
{
    static std::mutex m;
    return m;
}

// 尝试加载单个库，失败时打印一行英文日志并返回 false。
bool try_load(const char* path)
{
#if defined(__linux__) || defined(__APPLE__)
    // RTLD_GLOBAL：让库自身的依赖符号能被其他 dlopen 库互相解析到。
    void* h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr)
    {
        return false;
    }
    handles().push_back(h);
    return true;
#else
    (void)path;
    return false;
#endif
}

} // namespace

int load_native_libraries()
{
    std::lock_guard<std::mutex> lock(loader_mutex());

    // 幂等：已经加载成功就不再重复加载。
    if (!handles().empty())
    {
        return static_cast<int>(handles().size());
    }

    // 1. 显式路径（最高优先级）。
    if (const char* exact = std::getenv("PYHCCL_LIBRARY"))
    {
        if (try_load(exact))
        {
            return static_cast<int>(handles().size());
        }
    }

    // 2. 指定目录下的库名（不带 lib 前缀与 .so 后缀）。
    if (const char* dir = std::getenv("PYHCCL_LIBRARY_PATH"))
    {
        const std::string base(dir);
        const char* names[] = {"libhccl.so", "libhcomm.so", "libccl.so"};
        for (const char* name : names)
        {
            try_load((base + "/" + name).c_str());
        }
        if (!handles().empty())
        {
            return static_cast<int>(handles().size());
        }
    }

    // 3. 系统默认搜索路径。
    const char* names[] = {"libhccl.so", "libhcomm.so", "libccl.so", "libascendcl.so"};
    for (const char* name : names)
    {
        try_load(name);
    }
    return static_cast<int>(handles().size());
}

void* resolve_symbol(const char* name)
{
    std::lock_guard<std::mutex> lock(loader_mutex());

#if defined(__linux__) || defined(__APPLE__)
    // 优先在已加载的库中查找。
    for (void* h : handles())
    {
        void* sym = dlsym(h, name);
        if (sym != nullptr)
        {
            return sym;
        }
    }
    // 兜底：全局符号表（例如库被外部进程预先加载的场景）。
    return dlsym(RTLD_DEFAULT, name);
#else
    (void)name;
    return nullptr;
#endif
}

bool has_native_library()
{
    std::lock_guard<std::mutex> lock(loader_mutex());
    return !handles().empty();
}

} // namespace pyhccl

"""pyhccl：Huawei HCCL / HCOMM 集合通信库的 Python bindings。

本包在导入时不会加载原生库（libhccl.so / libhcomm.so），因此可以在任何机器上
导入并查看常量、枚举与数据结构。只有在调用真正需要原生库的函数时才会触发延迟
加载；若原生库不存在则抛出 RuntimeError（英文消息）。

用法：:

    import pyhccl

    pyhccl.load_library()                          # 显式加载原生库（可选）
    ret, rank_size = pyhccl.HcclGetRankSize(comm)  # 带出参的接口返回 (ret, out, ...)
"""

from . import _C

__version__ = "0.1.0"

# 导出 _C 中的所有公开符号（枚举、结构体、常量与 C 接口），既支持
# `from pyhccl import *`，也支持 `pyhccl.HCCL_SUCCESS` 这样的便捷访问。
__all__ = []
for _name in dir(_C):
    if not _name.startswith("_"):
        __all__.append(_name)
        globals()[_name] = getattr(_C, _name)

# 各数据类型占用的字节数（用于计算通信数据量），键为 HcclDataType 枚举值。
_DATA_TYPE_SIZES = {
    _C.HCCL_DATA_TYPE_INT8: 1,
    _C.HCCL_DATA_TYPE_INT16: 2,
    _C.HCCL_DATA_TYPE_INT32: 4,
    _C.HCCL_DATA_TYPE_FP16: 2,
    _C.HCCL_DATA_TYPE_FP32: 4,
    _C.HCCL_DATA_TYPE_INT64: 8,
    _C.HCCL_DATA_TYPE_UINT64: 8,
    _C.HCCL_DATA_TYPE_UINT8: 1,
    _C.HCCL_DATA_TYPE_UINT16: 2,
    _C.HCCL_DATA_TYPE_UINT32: 4,
    _C.HCCL_DATA_TYPE_FP64: 8,
    _C.HCCL_DATA_TYPE_BFP16: 2,
    _C.HCCL_DATA_TYPE_INT128: 16,
    _C.HCCL_DATA_TYPE_HIF8: 1,
    _C.HCCL_DATA_TYPE_FP8E4M3: 1,
    _C.HCCL_DATA_TYPE_FP8E5M2: 1,
    _C.HCCL_DATA_TYPE_FP8E8M0: 1,
}


def data_type_size(data_type: int) -> int:
    """返回给定 HcclDataType 枚举值对应的字节数（未知类型返回 0）。"""
    return _DATA_TYPE_SIZES.get(int(data_type), 0)


def get_error_string(code: int):
    """返回结果码对应的错误描述；原生库未加载时返回 None 而不是抛异常。"""
    if not _C.has_native_library():
        return None
    return _C.HcclGetErrorString(code)

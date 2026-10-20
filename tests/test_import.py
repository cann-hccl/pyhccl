"""导入与包级公开 API 的基本测试。"""

import pyhccl


def test_import_and_version():
    assert pyhccl.__version__ == "0.1.0"


def test_module_docstring():
    assert isinstance(pyhccl._C.__doc__, str)


def test_key_constants_exported():
    assert pyhccl.HCCL_SUCCESS == 0
    assert pyhccl.HCCL_ROOT_INFO_BYTES == 4108
    assert pyhccl.COMM_NAME_MAX_LENGTH == 128


def test_data_type_size():
    assert pyhccl.data_type_size(pyhccl.HCCL_DATA_TYPE_FP32) == 4
    assert pyhccl.data_type_size(pyhccl.HCCL_DATA_TYPE_FP16) == 2
    assert pyhccl.data_type_size(pyhccl.HCCL_DATA_TYPE_INT64) == 8
    # 未知类型返回 0
    assert pyhccl.data_type_size(12345) == 0

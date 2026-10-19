"""枚举与原生结果码的取值测试（不依赖原生库）。"""

import pyhccl


def test_hccl_result_values():
    assert pyhccl.HCCL_SUCCESS == 0
    assert pyhccl.HCCL_E_PARA == 1
    assert pyhccl.HCCL_E_PTR == 2
    assert pyhccl.HCCL_E_INTERNAL == 4


def test_enum_member_and_value():
    assert int(pyhccl.HcclDataType.HCCL_DATA_TYPE_FP32) == 4
    assert int(pyhccl.HcclReduceOp.HCCL_REDUCE_SUM) == 0
    assert int(pyhccl.CommEngine.COMM_ENGINE_CCU) == 5
    assert int(pyhccl.CcuResult.CCU_SUCCESS) == 0


def test_module_level_aliases():
    # export_values 使枚举常量同时作为模块级属性可用。
    assert pyhccl.HCCL_DATA_TYPE_FP32 == pyhccl.HcclDataType.HCCL_DATA_TYPE_FP32
    assert pyhccl.COMM_ENGINE_CCU == pyhccl.CommEngine.COMM_ENGINE_CCU

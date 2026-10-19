"""结构体与函数绑定行为测试。

注意：部分测试在有真机 CANN 环境的机器上（原生库已加载）与纯 CI 环境（无库）
行为不同，因此对“未加载原生库”相关的断言做了环境判断，保证两处都能通过。
"""

import pytest

import pyhccl


def test_struct_init_in_place():
    cfg = pyhccl.HcclCommConfig()
    pyhccl.HcclCommConfigInit(cfg)
    # 初始化后为“未配置”哨兵值。
    assert cfg.hcclBufferSize == 0xFFFFFFFF
    assert cfg.hcclDeterministic == 0xFFFFFFFF


def test_char_field_roundtrip():
    cfg = pyhccl.HcclCommConfig()
    cfg.hcclCommName = "rank_0"
    assert cfg.hcclCommName == "rank_0"


def test_channel_desc_init():
    desc = pyhccl.HcclChannelDesc()
    ret = pyhccl.HcclChannelDescInit(desc)
    assert ret == pyhccl.HCCL_SUCCESS
    assert desc.remoteRank == 0xFFFFFFFF


def test_endpoint_desc_init():
    ep = pyhccl.EndpointDesc()
    assert pyhccl.EndpointDescInit(ep) == pyhccl.HCCL_SUCCESS
    assert int(ep.protocol) == int(pyhccl.COMM_PROTOCOL_RESERVED)


def test_raws_bytes_roundtrip():
    desc = pyhccl.HcclChannelDesc()
    payload = b"\x01\x02\x03\x04"
    desc.raws = payload
    assert desc.raws[:4] == payload


def test_load_library_returns_int():
    # 显式加载原生库（CI 中无库，返回 0；真机上返回加载数量）。
    assert isinstance(pyhccl.load_library(), int)


def test_lazy_load_contract():
    # 无原生库时应抛出“not available”的 RuntimeError；有库时符号可解析、可调用。
    if not pyhccl.has_native_library():
        with pytest.raises(RuntimeError, match="not available"):
            pyhccl.HcclBarrier(0, 0)
    else:
        addr = pyhccl.resolve_symbol("HcclBarrier")
        assert addr != 0

// -*- coding: utf-8 -*-
// pyhccl 的 pybind11 绑定模块（Python 侧包名 pyhccl._C，参考 PyTorch 的 torch._C）。
//
// 本文件负责：
//   1. 导出所有枚举与结构体；
//   2. 导出 hccl/hcomm/ccu 的全部 host 侧 C 接口；
//   3. 提供 load_library / has_native_library 等运行时辅助函数。
//
// 约定：函数返回码为原始 int（可用模块级常量如 HCCL_SUCCESS 比较）；带出参的
// 函数返回 (ret, out0, out1, ...) 元组；原生库未加载时抛 RuntimeError（英文消息）。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>
#include <vector>

#include "abi.hpp"
#include "loader.hpp"
#include "wrappers.hpp"

namespace py = pybind11;
using namespace pyhccl;

// =============================================================================
// 辅助：序列转换与数组回读
// =============================================================================
namespace
{

std::vector<uint64_t> to_u64(py::handle h)
{
    std::vector<uint64_t> v;
    for (auto item : h)
    {
        v.push_back(item.cast<uint64_t>());
    }
    return v;
}

std::vector<uint32_t> to_u32(py::handle h)
{
    std::vector<uint32_t> v;
    for (auto item : h)
    {
        v.push_back(item.cast<uint32_t>());
    }
    return v;
}

std::vector<int32_t> to_i32(py::handle h)
{
    std::vector<int32_t> v;
    for (auto item : h)
    {
        v.push_back(item.cast<int32_t>());
    }
    return v;
}

std::vector<uintptr_t> to_ptr(py::handle h)
{
    std::vector<uintptr_t> v;
    for (auto item : h)
    {
        v.push_back(item.cast<uintptr_t>());
    }
    return v;
}

// bool 数组：C++ bool 占 1 字节，用 uint8_t 承载并按 bool* 传给南向接口。
std::vector<uint8_t> to_bool_bytes(py::handle h)
{
    std::vector<uint8_t> v;
    for (auto item : h)
    {
        v.push_back(item.cast<bool>() ? 1 : 0);
    }
    return v;
}

py::list u32list(const uint32_t* p, uint32_t n)
{
    py::list l;
    for (uint32_t i = 0; i < n; ++i)
    {
        l.append(p[i]);
    }
    return l;
}

py::list u64list(const uint64_t* p, uint64_t n)
{
    py::list l;
    for (uint64_t i = 0; i < n; ++i)
    {
        l.append(p[i]);
    }
    return l;
}

} // namespace

// 将定长 char 数组转换为 std::string（遇到 \0 或达到 N 为止，跨平台安全）。
template <size_t N> static std::string char_array_to_string(const char (&arr)[N])
{
    size_t len = 0;
    while (len < N && arr[len] != '\0')
    {
        ++len;
    }
    return std::string(arr, len);
}

// 绑定定长 char 数组字段（以 str 形式暴露，末尾 \0 截断）。
#define DEF_CHAR_FIELD(T, NAME, MEMBER)                                                                                \
    .def_property(                                                                                                     \
        NAME, [](const T& self) -> std::string { return char_array_to_string(self.MEMBER); },                          \
        [](T& self, const std::string& v)                                                                              \
        {                                                                                                              \
            std::memset(self.MEMBER, 0, sizeof(self.MEMBER));                                                          \
            std::memcpy(self.MEMBER, v.data(), std::min(v.size(), sizeof(self.MEMBER) - 1));                           \
        })

// 绑定 union 的 raws 字节缓冲区（以 bytes 形式暴露）。
#define DEF_RAW_FIELD(T, NAME, MEMBER)                                                                                 \
    .def_property(                                                                                                     \
        NAME,                                                                                                          \
        [](const T& self) { return py::bytes(reinterpret_cast<const char*>(&self.MEMBER), sizeof(self.MEMBER)); },     \
        [](T& self, const py::bytes& b)                                                                                \
        {                                                                                                              \
            std::string data = b;                                                                                      \
            size_t n = std::min(data.size(), sizeof(self.MEMBER));                                                     \
            std::memcpy(reinterpret_cast<char*>(&self.MEMBER), data.data(), n);                                        \
            if (n < sizeof(self.MEMBER))                                                                               \
            {                                                                                                          \
                std::memset(reinterpret_cast<char*>(&self.MEMBER) + n, 0, sizeof(self.MEMBER) - n);                    \
            }                                                                                                          \
        })

// =============================================================================
// 枚举绑定
// =============================================================================
static void bind_enums(py::module_& m)
{
    py::enum_<HcclResultEnum>(m, "HcclResult")
        .value("HCCL_SUCCESS", HCCL_SUCCESS)
        .value("HCCL_E_PARA", HCCL_E_PARA)
        .value("HCCL_E_PTR", HCCL_E_PTR)
        .value("HCCL_E_MEMORY", HCCL_E_MEMORY)
        .value("HCCL_E_INTERNAL", HCCL_E_INTERNAL)
        .value("HCCL_E_NOT_SUPPORT", HCCL_E_NOT_SUPPORT)
        .value("HCCL_E_NOT_FOUND", HCCL_E_NOT_FOUND)
        .value("HCCL_E_UNAVAIL", HCCL_E_UNAVAIL)
        .value("HCCL_E_SYSCALL", HCCL_E_SYSCALL)
        .value("HCCL_E_TIMEOUT", HCCL_E_TIMEOUT)
        .value("HCCL_E_OPEN_FILE_FAILURE", HCCL_E_OPEN_FILE_FAILURE)
        .value("HCCL_E_TCP_CONNECT", HCCL_E_TCP_CONNECT)
        .value("HCCL_E_ROCE_CONNECT", HCCL_E_ROCE_CONNECT)
        .value("HCCL_E_TCP_TRANSFER", HCCL_E_TCP_TRANSFER)
        .value("HCCL_E_ROCE_TRANSFER", HCCL_E_ROCE_TRANSFER)
        .value("HCCL_E_RUNTIME", HCCL_E_RUNTIME)
        .value("HCCL_E_DRV", HCCL_E_DRV)
        .value("HCCL_E_PROFILING", HCCL_E_PROFILING)
        .value("HCCL_E_CCE", HCCL_E_CCE)
        .value("HCCL_E_NETWORK", HCCL_E_NETWORK)
        .value("HCCL_E_AGAIN", HCCL_E_AGAIN)
        .value("HCCL_E_REMOTE", HCCL_E_REMOTE)
        .value("HCCL_E_SUSPENDING", HCCL_E_SUSPENDING)
        .value("HCCL_E_OPRETRY_FAIL", HCCL_E_OPRETRY_FAIL)
        .value("HCCL_E_OOM", HCCL_E_OOM)
        .value("HCCL_E_IN_STATUS", HCCL_E_IN_STATUS)
        .value("HCCL_E_RESERVED", HCCL_E_RESERVED)
        .export_values();

    py::enum_<HcclReduceOp>(m, "HcclReduceOp")
        .value("HCCL_REDUCE_SUM", HCCL_REDUCE_SUM)
        .value("HCCL_REDUCE_PROD", HCCL_REDUCE_PROD)
        .value("HCCL_REDUCE_MAX", HCCL_REDUCE_MAX)
        .value("HCCL_REDUCE_MIN", HCCL_REDUCE_MIN)
        .value("HCCL_REDUCE_RESERVED", HCCL_REDUCE_RESERVED)
        .export_values();

    py::enum_<HcclDataType>(m, "HcclDataType")
        .value("HCCL_DATA_TYPE_INT8", HCCL_DATA_TYPE_INT8)
        .value("HCCL_DATA_TYPE_INT16", HCCL_DATA_TYPE_INT16)
        .value("HCCL_DATA_TYPE_INT32", HCCL_DATA_TYPE_INT32)
        .value("HCCL_DATA_TYPE_FP16", HCCL_DATA_TYPE_FP16)
        .value("HCCL_DATA_TYPE_FP32", HCCL_DATA_TYPE_FP32)
        .value("HCCL_DATA_TYPE_INT64", HCCL_DATA_TYPE_INT64)
        .value("HCCL_DATA_TYPE_UINT64", HCCL_DATA_TYPE_UINT64)
        .value("HCCL_DATA_TYPE_UINT8", HCCL_DATA_TYPE_UINT8)
        .value("HCCL_DATA_TYPE_UINT16", HCCL_DATA_TYPE_UINT16)
        .value("HCCL_DATA_TYPE_UINT32", HCCL_DATA_TYPE_UINT32)
        .value("HCCL_DATA_TYPE_FP64", HCCL_DATA_TYPE_FP64)
        .value("HCCL_DATA_TYPE_BFP16", HCCL_DATA_TYPE_BFP16)
        .value("HCCL_DATA_TYPE_INT128", HCCL_DATA_TYPE_INT128)
        .value("HCCL_DATA_TYPE_HIF8", HCCL_DATA_TYPE_HIF8)
        .value("HCCL_DATA_TYPE_FP8E4M3", HCCL_DATA_TYPE_FP8E4M3)
        .value("HCCL_DATA_TYPE_FP8E5M2", HCCL_DATA_TYPE_FP8E5M2)
        .value("HCCL_DATA_TYPE_FP8E8M0", HCCL_DATA_TYPE_FP8E8M0)
        .value("HCCL_DATA_TYPE_RESERVED", HCCL_DATA_TYPE_RESERVED)
        .export_values();

    py::enum_<HcclConfig>(m, "HcclConfig")
        .value("HCCL_DETERMINISTIC", HCCL_DETERMINISTIC)
        .value("HCCL_CONFIG_RESERVED", HCCL_CONFIG_RESERVED)
        .export_values();

    py::enum_<HcclCommConfigCapability>(m, "HcclCommConfigCapability")
        .value("HCCL_COMM_CONFIG_BUFFER_SIZE", HCCL_COMM_CONFIG_BUFFER_SIZE)
        .value("HCCL_COMM_CONFIG_DETERMINISTIC", HCCL_COMM_CONFIG_DETERMINISTIC)
        .value("HCCL_COMM_CONFIG_COMM_NAME", HCCL_COMM_CONFIG_COMM_NAME)
        .value("HCCL_COMM_CONFIG_OP_EXPANSION_MODE", HCCL_COMM_CONFIG_OP_EXPANSION_MODE)
        .value("HCCL_COMM_CONFIG_SUPPORT_INIT_BY_ENV", HCCL_COMM_CONFIG_SUPPORT_INIT_BY_ENV)
        .value("HCCL_COMM_CONFIG_WORLD_RANKID", HCCL_COMM_CONFIG_WORLD_RANKID)
        .value("HCCL_COMM_CONFIG_JOBID", HCCL_COMM_CONFIG_JOBID)
        .value("HCCL_COMM_CONFIG_ACLGRAPH_ZEROCOPY_ENABLE", HCCL_COMM_CONFIG_ACLGRAPH_ZEROCOPY_ENABLE)
        .value("HCCL_COMM_CONFIG_EXEC_TIMEOUT", HCCL_COMM_CONFIG_EXEC_TIMEOUT)
        .value("HCCL_COMM_CONFIG_ALGO", HCCL_COMM_CONFIG_ALGO)
        .value("HCCL_COMM_CONFIG_RETRY", HCCL_COMM_CONFIG_RETRY)
        .value("HCCL_COMM_CONFIG_BUFFER_NAME", HCCL_COMM_CONFIG_BUFFER_NAME)
        .value("HCCL_COMM_CONFIG_RESERVED", HCCL_COMM_CONFIG_RESERVED)
        .export_values();

    py::enum_<HcclSendRecvType>(m, "HcclSendRecvType")
        .value("HCCL_SEND", HCCL_SEND)
        .value("HCCL_RECV", HCCL_RECV)
        .value("HCCL_SEND_RECV_RESERVED", HCCL_SEND_RECV_RESERVED)
        .export_values();

    py::enum_<HcclCMDType>(m, "HcclCMDType")
        .value("HCCL_CMD_INVALID", HCCL_CMD_INVALID)
        .value("HCCL_CMD_BROADCAST", HCCL_CMD_BROADCAST)
        .value("HCCL_CMD_ALLREDUCE", HCCL_CMD_ALLREDUCE)
        .value("HCCL_CMD_REDUCE", HCCL_CMD_REDUCE)
        .value("HCCL_CMD_SEND", HCCL_CMD_SEND)
        .value("HCCL_CMD_RECEIVE", HCCL_CMD_RECEIVE)
        .value("HCCL_CMD_ALLGATHER", HCCL_CMD_ALLGATHER)
        .value("HCCL_CMD_REDUCE_SCATTER", HCCL_CMD_REDUCE_SCATTER)
        .value("HCCL_CMD_ALLTOALLV", HCCL_CMD_ALLTOALLV)
        .value("HCCL_CMD_ALLTOALLVC", HCCL_CMD_ALLTOALLVC)
        .value("HCCL_CMD_ALLTOALL", HCCL_CMD_ALLTOALL)
        .value("HCCL_CMD_GATHER", HCCL_CMD_GATHER)
        .value("HCCL_CMD_SCATTER", HCCL_CMD_SCATTER)
        .value("HCCL_CMD_BATCH_SEND_RECV", HCCL_CMD_BATCH_SEND_RECV)
        .value("HCCL_CMD_BATCH_PUT", HCCL_CMD_BATCH_PUT)
        .value("HCCL_CMD_BATCH_GET", HCCL_CMD_BATCH_GET)
        .value("HCCL_CMD_ALLGATHER_V", HCCL_CMD_ALLGATHER_V)
        .value("HCCL_CMD_REDUCE_SCATTER_V", HCCL_CMD_REDUCE_SCATTER_V)
        .value("HCCL_CMD_BATCH_WRITE", HCCL_CMD_BATCH_WRITE)
        .value("HCCL_CMD_HALF_ALLTOALLV", HCCL_CMD_HALF_ALLTOALLV)
        .value("HCCL_CMD_ALL", HCCL_CMD_ALL)
        .value("HCCL_CMD_FINALIZE", HCCL_CMD_FINALIZE)
        .value("HCCL_CMD_INTER_GROUP_SYNC", HCCL_CMD_INTER_GROUP_SYNC)
        .value("HCCL_CMD_INIT", HCCL_CMD_INIT)
        .value("HCCL_CMD_BARRIER", HCCL_CMD_BARRIER)
        .value("HCCL_CMD_MAX", HCCL_CMD_MAX)
        .export_values();

    py::enum_<HcclCommStatus>(m, "HcclCommStatus")
        .value("HCCL_COMM_STATUS_READY", HCCL_COMM_STATUS_READY)
        .value("HCCL_COMM_STATUS_SUSPENDING", HCCL_COMM_STATUS_SUSPENDING)
        .value("HCCL_COMM_STATUS_INVALID", HCCL_COMM_STATUS_INVALID)
        .value("HCCL_COMM_STATUS_RESERVED", HCCL_COMM_STATUS_RESERVED)
        .export_values();

    py::enum_<HcclCommStatePhase>(m, "HcclCommStatePhase")
        .value("HCCL_COMM_STATE_PHASE_INVALID", HCCL_COMM_STATE_PHASE_INVALID)
        .value("HCCL_COMM_STATE_PHASE_DESTROY_PRE", HCCL_COMM_STATE_PHASE_DESTROY_PRE)
        .value("HCCL_COMM_STATE_PHASE_DESTROY_POST", HCCL_COMM_STATE_PHASE_DESTROY_POST)
        .value("HCCL_COMM_STATE_PHASE_RESUME_PRE", HCCL_COMM_STATE_PHASE_RESUME_PRE)
        .value("HCCL_COMM_STATE_PHASE_RESUME_POST", HCCL_COMM_STATE_PHASE_RESUME_POST)
        .export_values();

    py::enum_<HcclOpExpansionMode>(m, "HcclOpExpansionMode")
        .value("HCCL_OP_EXPANSION_MODE_INVALID", HCCL_OP_EXPANSION_MODE_INVALID)
        .value("HCCL_OP_EXPANSION_MODE_AI_CPU", HCCL_OP_EXPANSION_MODE_AI_CPU)
        .value("HCCL_OP_EXPANSION_MODE_AIV", HCCL_OP_EXPANSION_MODE_AIV)
        .value("HCCL_OP_EXPANSION_MODE_HOST", HCCL_OP_EXPANSION_MODE_HOST)
        .value("HCCL_OP_EXPANSION_MODE_HOST_TS", HCCL_OP_EXPANSION_MODE_HOST_TS)
        .value("HCCL_OP_EXPANSION_MODE_CCU_MS", HCCL_OP_EXPANSION_MODE_CCU_MS)
        .value("HCCL_OP_EXPANSION_MODE_CCU_SCHED", HCCL_OP_EXPANSION_MODE_CCU_SCHED)
        .value("HCCL_OP_EXPANSION_AIV_ONLY", HCCL_OP_EXPANSION_AIV_ONLY)
        .export_values();

    py::enum_<HcclConfigType>(m, "HcclConfigType")
        .value("HCCL_CONFIG_TYPE_INVALID", HCCL_CONFIG_TYPE_INVALID)
        .value("HCCL_CONFIG_TYPE_OP_EXPANSION_MODE", HCCL_CONFIG_TYPE_OP_EXPANSION_MODE)
        .value("HCCL_CONFIG_TYPE_HCCL_ALGO", HCCL_CONFIG_TYPE_HCCL_ALGO)
        .value("HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM", HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM)
        .value("HCCL_CONFIG_TYPE_DETERMINISTIC", HCCL_CONFIG_TYPE_DETERMINISTIC)
        .value("HCCL_CONFIG_TYPE_UDI", HCCL_CONFIG_TYPE_UDI)
        .export_values();

    py::enum_<HcclDedicatedThreadType>(m, "HcclDedicatedThreadType")
        .value("HCCL_DED_THREAD_TYPE_INVALID", HCCL_DED_THREAD_TYPE_INVALID)
        .value("HCCL_DED_THREAD_TYPE_AICPU_LAUNCH", HCCL_DED_THREAD_TYPE_AICPU_LAUNCH)
        .value("HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE", HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE)
        .value("HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE", HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE)
        .value("HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH", HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH)
        .value("HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE", HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE)
        .value("HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE", HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE)
        .export_values();

    py::enum_<ThreadType>(m, "ThreadType")
        .value("THREAD_TYPE_INVALID", THREAD_TYPE_INVALID)
        .value("THREAD_TYPE_TS", THREAD_TYPE_TS)
        .value("THREAD_TYPE_CPU", THREAD_TYPE_CPU)
        .export_values();

    py::enum_<CommEngine>(m, "CommEngine")
        .value("COMM_ENGINE_RESERVED", COMM_ENGINE_RESERVED)
        .value("COMM_ENGINE_CPU", COMM_ENGINE_CPU)
        .value("COMM_ENGINE_CPU_TS", COMM_ENGINE_CPU_TS)
        .value("COMM_ENGINE_AICPU", COMM_ENGINE_AICPU)
        .value("COMM_ENGINE_AICPU_TS", COMM_ENGINE_AICPU_TS)
        .value("COMM_ENGINE_AIV", COMM_ENGINE_AIV)
        .value("COMM_ENGINE_CCU", COMM_ENGINE_CCU)
        .export_values();

    py::enum_<CommProtocol>(m, "CommProtocol")
        .value("COMM_PROTOCOL_RESERVED", COMM_PROTOCOL_RESERVED)
        .value("COMM_PROTOCOL_HCCS", COMM_PROTOCOL_HCCS)
        .value("COMM_PROTOCOL_ROCE", COMM_PROTOCOL_ROCE)
        .value("COMM_PROTOCOL_PCIE", COMM_PROTOCOL_PCIE)
        .value("COMM_PROTOCOL_SIO", COMM_PROTOCOL_SIO)
        .value("COMM_PROTOCOL_UB_CTP", COMM_PROTOCOL_UB_CTP)
        .value("COMM_PROTOCOL_UBC_TP", COMM_PROTOCOL_UBC_TP)
        .value("COMM_PROTOCOL_UB_MEM", COMM_PROTOCOL_UB_MEM)
        .value("COMM_PROTOCOL_UBOE", COMM_PROTOCOL_UBOE)
        .value("COMM_PROTOCOL_HCCS_ONLY", COMM_PROTOCOL_HCCS_ONLY)
        .value("COMM_PROTOCOL_UB_RTP", COMM_PROTOCOL_UB_RTP)
        .export_values();

    py::enum_<CommAddrType>(m, "CommAddrType")
        .value("COMM_ADDR_TYPE_RESERVED", COMM_ADDR_TYPE_RESERVED)
        .value("COMM_ADDR_TYPE_IP_V4", COMM_ADDR_TYPE_IP_V4)
        .value("COMM_ADDR_TYPE_IP_V6", COMM_ADDR_TYPE_IP_V6)
        .value("COMM_ADDR_TYPE_ID", COMM_ADDR_TYPE_ID)
        .value("COMM_ADDR_TYPE_EID", COMM_ADDR_TYPE_EID)
        .export_values();

    py::enum_<EndpointLocType>(m, "EndpointLocType")
        .value("ENDPOINT_LOC_TYPE_RESERVED", ENDPOINT_LOC_TYPE_RESERVED)
        .value("ENDPOINT_LOC_TYPE_DEVICE", ENDPOINT_LOC_TYPE_DEVICE)
        .value("ENDPOINT_LOC_TYPE_HOST", ENDPOINT_LOC_TYPE_HOST)
        .export_values();

    py::enum_<CommMemType>(m, "CommMemType")
        .value("COMM_MEM_TYPE_INVALID", COMM_MEM_TYPE_INVALID)
        .value("COMM_MEM_TYPE_DEVICE", COMM_MEM_TYPE_DEVICE)
        .value("COMM_MEM_TYPE_HOST", COMM_MEM_TYPE_HOST)
        .value("COMM_MEM_TYPE_CCU", COMM_MEM_TYPE_CCU)
        .export_values();

    py::enum_<HcommSocketRole>(m, "HcommSocketRole")
        .value("HCOMM_SOCKET_ROLE_RESERVED", HCOMM_SOCKET_ROLE_RESERVED)
        .value("HCOMM_SOCKET_ROLE_CLIENT", HCOMM_SOCKET_ROLE_CLIENT)
        .value("HCOMM_SOCKET_ROLE_SERVER", HCOMM_SOCKET_ROLE_SERVER)
        .export_values();

    py::enum_<HcommEndpointFeatureType>(m, "HcommEndpointFeatureType")
        .value("HCOMM_ENDPOINT_FEATURE_INVALID", HCOMM_ENDPOINT_FEATURE_INVALID)
        .value("HCOMM_ENDPOINT_FEATURE_NDA", HCOMM_ENDPOINT_FEATURE_NDA)
        .export_values();

    py::enum_<ThreadResType>(m, "ThreadResType")
        .value("THREAD_RES_TYPE_INVALID", THREAD_RES_TYPE_INVALID)
        .value("THREAD_RES_TYPE_STREAM", THREAD_RES_TYPE_STREAM)
        .export_values();

    py::enum_<HcclChannelConfigType>(m, "HcclChannelConfigType")
        .value("HCCL_CHANNEL_CONFIG_TYPE_INVALID", HCCL_CHANNEL_CONFIG_TYPE_INVALID)
        .value("HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE", HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE)
        .value("HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG", HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG)
        .export_values();

    py::enum_<HcommChannelConfigType>(m, "HcommChannelConfigType")
        .value("HCOMM_CHANNEL_CONFIG_TYPE_INVALID", HCOMM_CHANNEL_CONFIG_TYPE_INVALID)
        .value("HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE", HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE)
        .export_values();

    py::enum_<HcommChannelStatus>(m, "HcommChannelStatus")
        .value("HCOMM_CHANNEL_STATUS_READY", HCOMM_CHANNEL_STATUS_READY)
        .value("HCOMM_CHANNEL_STATUS_CONNECTING", HCOMM_CHANNEL_STATUS_CONNECTING)
        .value("HCOMM_CHANNEL_STATUS_FAILED_INTERNAL", HCOMM_CHANNEL_STATUS_FAILED_INTERNAL)
        .export_values();

    py::enum_<HcommReduceOp>(m, "HcommReduceOp")
        .value("HCOMM_REDUCE_SUM", HCOMM_REDUCE_SUM)
        .value("HCOMM_REDUCE_PROD", HCOMM_REDUCE_PROD)
        .value("HCOMM_REDUCE_MAX", HCOMM_REDUCE_MAX)
        .value("HCOMM_REDUCE_MIN", HCOMM_REDUCE_MIN)
        .value("HCOMM_REDUCE_RESERVED", HCOMM_REDUCE_RESERVED)
        .export_values();

    py::enum_<HcommDataType>(m, "HcommDataType")
        .value("HCOMM_DATA_TYPE_INT8", HCOMM_DATA_TYPE_INT8)
        .value("HCOMM_DATA_TYPE_INT16", HCOMM_DATA_TYPE_INT16)
        .value("HCOMM_DATA_TYPE_INT32", HCOMM_DATA_TYPE_INT32)
        .value("HCOMM_DATA_TYPE_FP16", HCOMM_DATA_TYPE_FP16)
        .value("HCOMM_DATA_TYPE_FP32", HCOMM_DATA_TYPE_FP32)
        .value("HCOMM_DATA_TYPE_INT64", HCOMM_DATA_TYPE_INT64)
        .value("HCOMM_DATA_TYPE_UINT64", HCOMM_DATA_TYPE_UINT64)
        .value("HCOMM_DATA_TYPE_UINT8", HCOMM_DATA_TYPE_UINT8)
        .value("HCOMM_DATA_TYPE_UINT16", HCOMM_DATA_TYPE_UINT16)
        .value("HCOMM_DATA_TYPE_UINT32", HCOMM_DATA_TYPE_UINT32)
        .value("HCOMM_DATA_TYPE_FP64", HCOMM_DATA_TYPE_FP64)
        .value("HCOMM_DATA_TYPE_BFP16", HCOMM_DATA_TYPE_BFP16)
        .value("HCOMM_DATA_TYPE_INT128", HCOMM_DATA_TYPE_INT128)
        .value("HCOMM_DATA_TYPE_HIF8", HCOMM_DATA_TYPE_HIF8)
        .value("HCOMM_DATA_TYPE_FP8E4M3", HCOMM_DATA_TYPE_FP8E4M3)
        .value("HCOMM_DATA_TYPE_FP8E5M2", HCOMM_DATA_TYPE_FP8E5M2)
        .value("HCOMM_DATA_TYPE_FP8E8M0", HCOMM_DATA_TYPE_FP8E8M0)
        .value("HCOMM_DATA_TYPE_RESERVED", HCOMM_DATA_TYPE_RESERVED)
        .export_values();

    py::enum_<HcommTransferType>(m, "HcommTransferType")
        .value("HCOMM_TRANSFER_TYPE_INVALID", HCOMM_TRANSFER_TYPE_INVALID)
        .value("HCOMM_TRANSFER_TYPE_WRITE", HCOMM_TRANSFER_TYPE_WRITE)
        .value("HCOMM_TRANSFER_TYPE_WRITE_REDUCE", HCOMM_TRANSFER_TYPE_WRITE_REDUCE)
        .value("HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY", HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY)
        .value("HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY", HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY)
        .value("HCOMM_TRANSFER_TYPE_READ", HCOMM_TRANSFER_TYPE_READ)
        .value("HCOMM_TRANSFER_TYPE_READ_REDUCE", HCOMM_TRANSFER_TYPE_READ_REDUCE)
        .value("HCOMM_TRANSFER_TYPE_NOTIFY_RECORD", HCOMM_TRANSFER_TYPE_NOTIFY_RECORD)
        .export_values();

    py::enum_<EndpointAttr>(m, "EndpointAttr")
        .value("ENDPOINT_ATTR_INVALID", ENDPOINT_ATTR_INVALID)
        .value("ENDPOINT_ATTR_BW_COEFF", ENDPOINT_ATTR_BW_COEFF)
        .value("ENDPOINT_ATTR_DIE_ID", ENDPOINT_ATTR_DIE_ID)
        .value("ENDPOINT_ATTR_LOCATION", ENDPOINT_ATTR_LOCATION)
        .export_values();

    py::enum_<CommTopo>(m, "CommTopo")
        .value("COMM_TOPO_RESERVED", COMM_TOPO_RESERVED)
        .value("COMM_TOPO_CLOS", COMM_TOPO_CLOS)
        .value("COMM_TOPO_1DMESH", COMM_TOPO_1DMESH)
        .value("COMM_TOPO_910_93", COMM_TOPO_910_93)
        .value("COMM_TOPO_310P", COMM_TOPO_310P)
        .value("COMM_TOPO_A2AXSERVER", COMM_TOPO_A2AXSERVER)
        .value("COMM_TOPO_CUSTOM", COMM_TOPO_CUSTOM)
        .export_values();

    py::enum_<HcclHeterogMode>(m, "HcclHeterogMode")
        .value("HCCL_HETEROG_MODE_INVALID", HCCL_HETEROG_MODE_INVALID)
        .value("HCCL_HETEROG_MODE_HOMOGENEOUS", HCCL_HETEROG_MODE_HOMOGENEOUS)
        .value("HCCL_HETEROG_MODE_MIX_A2_A3", HCCL_HETEROG_MODE_MIX_A2_A3)
        .export_values();

    py::enum_<CcuResultEnum>(m, "CcuResult")
        .value("CCU_SUCCESS", CCU_SUCCESS)
        .value("CCU_E_PARA", CCU_E_PARA)
        .value("CCU_E_PTR", CCU_E_PTR)
        .value("CCU_E_INTERNAL", CCU_E_INTERNAL)
        .value("CCU_E_NOT_SUPPORT", CCU_E_NOT_SUPPORT)
        .value("CCU_E_NOT_FOUND", CCU_E_NOT_FOUND)
        .value("CCU_E_UNAVAIL", CCU_E_UNAVAIL)
        .value("CCU_E_RUNTIME", CCU_E_RUNTIME)
        .value("CCU_E_DRV_START", CCU_E_DRV_START)
        .value("CCU_E_DRV_INIT_FAILED", CCU_E_DRV_INIT_FAILED)
        .value("CCU_E_DRV_BUSY", CCU_E_DRV_BUSY)
        .value("CCU_E_DRV_END", CCU_E_DRV_END)
        .value("CCU_E_RESERVED", CCU_E_RESERVED)
        .export_values();

    py::enum_<CcuConditionType>(m, "CcuConditionType")
        .value("CCU_CONDITION_EQ", CCU_CONDITION_EQ)
        .value("CCU_CONDITION_NE", CCU_CONDITION_NE)
        .value("CCU_CONDITION_LT", CCU_CONDITION_LT)
        .value("CCU_CONDITION_LE", CCU_CONDITION_LE)
        .value("CCU_CONDITION_GT", CCU_CONDITION_GT)
        .value("CCU_CONDITION_GE", CCU_CONDITION_GE)
        .export_values();

    py::enum_<HcommCcuResType>(m, "HcommCcuResType")
        .value("HCOMM_CCU_RES_TYPE_INVALID", HCOMM_CCU_RES_TYPE_INVALID)
        .value("HCOMM_CCU_RES_TYPE_LOOP", HCOMM_CCU_RES_TYPE_LOOP)
        .value("HCOMM_CCU_RES_TYPE_CCU_BUF", HCOMM_CCU_RES_TYPE_CCU_BUF)
        .value("HCOMM_CCU_RES_TYPE_VARIABLE", HCOMM_CCU_RES_TYPE_VARIABLE)
        .value("HCOMM_CCU_RES_TYPE_ADDRESS", HCOMM_CCU_RES_TYPE_ADDRESS)
        .value("HCOMM_CCU_RES_TYPE_EVENT", HCOMM_CCU_RES_TYPE_EVENT)
        .value("HCOMM_CCU_RES_TYPE_CCU_THREAD", HCOMM_CCU_RES_TYPE_CCU_THREAD)
        .value("HCOMM_CCU_RES_TYPE_INSTRUCTION", HCOMM_CCU_RES_TYPE_INSTRUCTION)
        .value("HCOMM_CCU_RES_TYPE_CASC_CNT", HCOMM_CCU_RES_TYPE_CASC_CNT)
        .export_values();
}

// =============================================================================
// 结构体绑定
// =============================================================================
static void bind_structs(py::module_& m)
{
    py::class_<CommAbiHeader>(m, "CommAbiHeader")
        .def(py::init<>())
        .def_readwrite("version", &CommAbiHeader::version)
        .def_readwrite("magicWord", &CommAbiHeader::magicWord)
        .def_readwrite("size", &CommAbiHeader::size)
        .def_readwrite("reserved", &CommAbiHeader::reserved);

    py::class_<ThreadConfig>(m, "ThreadConfig")
        .def(py::init<>())
        .def_readwrite("header", &ThreadConfig::header)
        .def_readwrite("notifyNumPerThread", &ThreadConfig::notifyNumPerThread);

    py::class_<CommAddr>(m, "CommAddr")
        .def(py::init<>())
        .def_readwrite("type", &CommAddr::type)
        .def_property(
            "id", [](const CommAddr& s) { return s.id; }, [](CommAddr& s, uint32_t v) { s.id = v; })
            DEF_RAW_FIELD(CommAddr, "raws", raws);

    py::class_<EndpointLoc>(m, "EndpointLoc")
        .def(py::init<>())
        .def_readwrite("locType", &EndpointLoc::locType)
        .def_property(
            "devPhyId", [](const EndpointLoc& s) { return s.device.devPhyId; },
            [](EndpointLoc& s, uint32_t v) { s.device.devPhyId = v; })
        .def_property(
            "superDevId", [](const EndpointLoc& s) { return s.device.superDevId; },
            [](EndpointLoc& s, uint32_t v) { s.device.superDevId = v; })
        .def_property(
            "serverIdx", [](const EndpointLoc& s) { return s.device.serverIdx; },
            [](EndpointLoc& s, uint32_t v) { s.device.serverIdx = v; })
        .def_property(
            "superPodIdx", [](const EndpointLoc& s) { return s.device.superPodIdx; },
            [](EndpointLoc& s, uint32_t v) { s.device.superPodIdx = v; })
        .def_property(
            "hostId", [](const EndpointLoc& s) { return s.host.id; }, [](EndpointLoc& s, uint32_t v) { s.host.id = v; })
            DEF_RAW_FIELD(EndpointLoc, "raws", raws);

    py::class_<EndpointDesc>(m, "EndpointDesc")
        .def(py::init<>())
        .def_readwrite("protocol", &EndpointDesc::protocol)
        .def_readwrite("commAddr", &EndpointDesc::commAddr)
        .def_readwrite("loc", &EndpointDesc::loc) DEF_RAW_FIELD(EndpointDesc, "raws", raws);

    py::class_<CommMem>(m, "CommMem")
        .def(py::init<>())
        .def_readwrite("type", &CommMem::type)
        .def_readwrite("addr", &CommMem::addr)
        .def_readwrite("size", &CommMem::size);

    py::class_<HcommTeamSyncMemRequirement>(m, "HcommTeamSyncMemRequirement")
        .def(py::init<>())
        .def_readwrite("signalCount", &HcommTeamSyncMemRequirement::signalCount)
        .def_readwrite("counterCount", &HcommTeamSyncMemRequirement::counterCount)
        .def_readwrite("barrierCount", &HcommTeamSyncMemRequirement::barrierCount);

    py::class_<HcclRootInfo>(m, "HcclRootInfo").def(py::init<>()) DEF_RAW_FIELD(HcclRootInfo, "internal", internal);

    py::class_<HcclCommConfig>(m, "HcclCommConfig")
        .def(py::init<>())
        .def_readwrite("hcclBufferSize", &HcclCommConfig::hcclBufferSize)
        .def_readwrite("hcclDeterministic", &HcclCommConfig::hcclDeterministic)
            DEF_CHAR_FIELD(HcclCommConfig, "hcclCommName", hcclCommName)
                DEF_CHAR_FIELD(HcclCommConfig, "hcclUdi", hcclUdi)
        .def_readwrite("hcclOpExpansionMode", &HcclCommConfig::hcclOpExpansionMode)
        .def_readwrite("hcclRdmaTrafficClass", &HcclCommConfig::hcclRdmaTrafficClass)
        .def_readwrite("hcclRdmaServiceLevel", &HcclCommConfig::hcclRdmaServiceLevel)
        .def_readwrite("hcclWorldRankID", &HcclCommConfig::hcclWorldRankID)
        .def_readwrite("hcclJobID", &HcclCommConfig::hcclJobID)
        .def_readwrite("aclGraphZeroCopyEnable", &HcclCommConfig::aclGraphZeroCopyEnable)
        .def_readwrite("hcclExecTimeOut", &HcclCommConfig::hcclExecTimeOut)
            DEF_CHAR_FIELD(HcclCommConfig, "hcclAlgo", hcclAlgo)
                DEF_CHAR_FIELD(HcclCommConfig, "hcclRetryEnable", hcclRetryEnable)
                    DEF_CHAR_FIELD(HcclCommConfig, "hcclRetryParams", hcclRetryParams)
                        DEF_CHAR_FIELD(HcclCommConfig, "hcclBufferName", hcclBufferName)
        .def_readwrite("hcclQos", &HcclCommConfig::hcclQos)
        .def_readwrite("hcclSymWinMaxMemSizePerRank", &HcclCommConfig::hcclSymWinMaxMemSizePerRank)
        .def_readwrite("hcclChannelSqDepth", &HcclCommConfig::hcclChannelSqDepth);

    py::class_<HcclSendRecvItem>(m, "HcclSendRecvItem")
        .def(py::init<>())
        .def_readwrite("sendRecvType", &HcclSendRecvItem::sendRecvType)
        .def_readwrite("buf", &HcclSendRecvItem::buf)
        .def_readwrite("count", &HcclSendRecvItem::count)
        .def_readwrite("dataType", &HcclSendRecvItem::dataType)
        .def_readwrite("remoteRank", &HcclSendRecvItem::remoteRank);

    py::class_<HcclP2pKernelParam>(m, "HcclP2pKernelParam")
        .def(py::init<>())
        .def_readwrite("sendRecvThread", &HcclP2pKernelParam::sendRecvThread)
            DEF_RAW_FIELD(HcclP2pKernelParam, "opParams", opParams);

    py::class_<HcclOpP2pDesc>(m, "HcclOpP2pDesc")
        .def(py::init<>())
        .def_readwrite("buffer", &HcclOpP2pDesc::buffer)
        .def_readwrite("cmdType", &HcclOpP2pDesc::cmdType)
        .def_readwrite("dataType", &HcclOpP2pDesc::dataType)
        .def_readwrite("count", &HcclOpP2pDesc::count)
        .def_readwrite("remoteRank", &HcclOpP2pDesc::remoteRank)
        .def_readwrite("unfoldStream", &HcclOpP2pDesc::unfoldStream);

    py::class_<HcclOpDesc>(m, "HcclOpDesc")
        .def(py::init<>())
        .def_readwrite("header", &HcclOpDesc::header)
        .def_readwrite("opDescType", &HcclOpDesc::opDescType) DEF_CHAR_FIELD(HcclOpDesc, "opName", opName)
            DEF_RAW_FIELD(HcclOpDesc, "raws", raws);

    py::class_<HcclKernelFuncInfo>(m, "HcclKernelFuncInfo")
        .def(py::init<>()) DEF_CHAR_FIELD(HcclKernelFuncInfo, "kernelSoName", kernelSoName)
            DEF_CHAR_FIELD(HcclKernelFuncInfo, "kernelFuncName", kernelFuncName)
        .def_readwrite("args", &HcclKernelFuncInfo::args)
        .def_readwrite("argSize", &HcclKernelFuncInfo::argSize);

    py::class_<HcclKernelLaunchCfg>(m, "HcclKernelLaunchCfg")
        .def(py::init<>())
        .def_readwrite("header", &HcclKernelLaunchCfg::header)
        .def_readwrite("timeOut", &HcclKernelLaunchCfg::timeOut)
            DEF_RAW_FIELD(HcclKernelLaunchCfg, "reserved", reserved);

    py::class_<HcclChannelDesc>(m, "HcclChannelDesc")
        .def(py::init<>())
        .def_readwrite("header", &HcclChannelDesc::header)
        .def_readwrite("remoteRank", &HcclChannelDesc::remoteRank)
        .def_readwrite("channelProtocol", &HcclChannelDesc::channelProtocol)
        .def_readwrite("localEndpoint", &HcclChannelDesc::localEndpoint)
        .def_readwrite("remoteEndpoint", &HcclChannelDesc::remoteEndpoint)
        .def_readwrite("notifyNum", &HcclChannelDesc::notifyNum)
        .def_readwrite("memHandles", &HcclChannelDesc::memHandles)
        .def_readwrite("memHandleNum", &HcclChannelDesc::memHandleNum)
        .def_property(
            "queueNum", [](const HcclChannelDesc& s) { return s.roceAttr.queueNum; },
            [](HcclChannelDesc& s, uint32_t v) { s.roceAttr.queueNum = v; })
        .def_property(
            "retryCnt", [](const HcclChannelDesc& s) { return s.roceAttr.retryCnt; },
            [](HcclChannelDesc& s, uint32_t v) { s.roceAttr.retryCnt = v; })
        .def_property(
            "retryInterval", [](const HcclChannelDesc& s) { return s.roceAttr.retryInterval; },
            [](HcclChannelDesc& s, uint32_t v) { s.roceAttr.retryInterval = v; })
        .def_property(
            "tc", [](const HcclChannelDesc& s) { return s.roceAttr.tc; },
            [](HcclChannelDesc& s, uint8_t v) { s.roceAttr.tc = v; })
        .def_property(
            "sl", [](const HcclChannelDesc& s) { return s.roceAttr.sl; },
            [](HcclChannelDesc& s, uint8_t v) { s.roceAttr.sl = v; })
        .def_property(
            "pathMode", [](const HcclChannelDesc& s) { return s.ubMemAttr.pathMode; }, [](HcclChannelDesc& s, uint8_t v)
            { s.ubMemAttr.pathMode = v; }) DEF_RAW_FIELD(HcclChannelDesc, "raws", raws);

    py::class_<HcommChannelDesc>(m, "HcommChannelDesc")
        .def(py::init<>())
        .def_readwrite("header", &HcommChannelDesc::header)
        .def_readwrite("remoteEndpoint", &HcommChannelDesc::remoteEndpoint)
        .def_readwrite("notifyNum", &HcommChannelDesc::notifyNum)
        .def_readwrite("exchangeAllMems", &HcommChannelDesc::exchangeAllMems)
        .def_readwrite("memHandles", &HcommChannelDesc::memHandles)
        .def_readwrite("memHandleNum", &HcommChannelDesc::memHandleNum)
        .def_readwrite("socket", &HcommChannelDesc::socket)
        .def_readwrite("role", &HcommChannelDesc::role)
        .def_readwrite("port", &HcommChannelDesc::port)
        .def_readwrite("qos", &HcommChannelDesc::qos)
        .def_readwrite("channelName", &HcommChannelDesc::channelName) DEF_RAW_FIELD(HcommChannelDesc, "raws", raws);

    py::class_<HcclTeamCreateDesc>(m, "HcclTeamCreateDesc")
        .def(py::init<>())
        .def_readwrite("header", &HcclTeamCreateDesc::header)
        .def_readwrite("rankIds", &HcclTeamCreateDesc::rankIds)
        .def_readwrite("rankNum", &HcclTeamCreateDesc::rankNum)
        .def_readwrite("selfRankId", &HcclTeamCreateDesc::selfRankId)
        .def_readwrite("netLayer", &HcclTeamCreateDesc::netLayer)
        .def_readwrite("protocol", &HcclTeamCreateDesc::protocol)
        .def_readwrite("requirement", &HcclTeamCreateDesc::requirement)
        .def_readwrite("engine", &HcclTeamCreateDesc::engine)
        .def_readwrite("notifyNum", &HcclTeamCreateDesc::notifyNum)
        .def_readwrite("channelCnt", &HcclTeamCreateDesc::channelCnt)
        .def_readwrite("isSharedQueue", &HcclTeamCreateDesc::isSharedQueue)
            DEF_CHAR_FIELD(HcclTeamCreateDesc, "sharedQueueTag", sharedQueueTag);

    py::class_<CommLink>(m, "CommLink")
        .def(py::init<>())
        .def_readwrite("header", &CommLink::header)
        .def_readwrite("srcEndpointDesc", &CommLink::srcEndpointDesc)
        .def_readwrite("dstEndpointDesc", &CommLink::dstEndpointDesc)
            DEF_RAW_FIELD(CommLink, "linkAttrRaws", linkAttr.raws);

    py::class_<HcommBatchTransferDesc>(m, "HcommBatchTransferDesc")
        .def(py::init<>())
        .def_readwrite("transType", &HcommBatchTransferDesc::transType)
            DEF_RAW_FIELD(HcommBatchTransferDesc, "transferInfoRaws", transferInfo.raws);

    py::class_<CcuCfgHeader>(m, "CcuCfgHeader")
        .def(py::init<>())
        .def_readwrite("magic", &CcuCfgHeader::magic)
        .def_readwrite("version", &CcuCfgHeader::version)
        .def_readwrite("size", &CcuCfgHeader::size)
        .def_readwrite("reserved", &CcuCfgHeader::reserved);

    py::class_<CcuLoopConfig>(m, "CcuLoopConfig")
        .def(py::init<>())
        .def_readwrite("addrOffset", &CcuLoopConfig::addrOffset)
        .def_readwrite("iterNum", &CcuLoopConfig::iterNum);

    py::class_<CcuLoopGroupConfig>(m, "CcuLoopGroupConfig")
        .def(py::init<>())
        .def_readwrite("cloneNum", &CcuLoopGroupConfig::cloneNum)
        .def_readwrite("cloneLoopOffset", &CcuLoopGroupConfig::cloneLoopOffset)
        .def_readwrite("addrOffset", &CcuLoopGroupConfig::addrOffset)
        .def_readwrite("ccuBufferOffset", &CcuLoopGroupConfig::ccuBufferOffset)
        .def_readwrite("eventOffset", &CcuLoopGroupConfig::eventOffset);

    py::class_<CcuLoopCfg>(m, "CcuLoopCfg")
        .def(py::init<>())
        .def_readwrite("header", &CcuLoopCfg::header)
        .def_readwrite("addrOffset", &CcuLoopCfg::addrOffset)
        .def_readwrite("iterNum", &CcuLoopCfg::iterNum);

    py::class_<CcuLoopGroupCfg>(m, "CcuLoopGroupCfg")
        .def(py::init<>())
        .def_readwrite("header", &CcuLoopGroupCfg::header)
        .def_readwrite("cloneNum", &CcuLoopGroupCfg::cloneNum)
        .def_readwrite("cloneLoopOffset", &CcuLoopGroupCfg::cloneLoopOffset)
        .def_readwrite("addrOffset", &CcuLoopGroupCfg::addrOffset)
        .def_readwrite("ccuBufferOffset", &CcuLoopGroupCfg::ccuBufferOffset)
        .def_readwrite("eventOffset", &CcuLoopGroupCfg::eventOffset)
        .def_readwrite("varOffset", &CcuLoopGroupCfg::varOffset);
}

// =============================================================================
// 函数绑定
// =============================================================================
static void bind_functions(py::module_& m)
{
    using namespace pybind11::literals;

    // ---------- 集合通信（hccl.h） ----------
    m.def("HcclAllReduce", &HcclAllReduce, "sendBuf"_a, "recvBuf"_a, "count"_a, "dataType"_a, "op"_a, "comm"_a,
          "stream"_a);
    m.def("HcclBroadcast", &HcclBroadcast, "buf"_a, "count"_a, "dataType"_a, "root"_a, "comm"_a, "stream"_a);
    m.def("HcclReduceScatter", &HcclReduceScatter, "sendBuf"_a, "recvBuf"_a, "recvCount"_a, "dataType"_a, "op"_a,
          "comm"_a, "stream"_a);
    m.def("HcclScatter", &HcclScatter, "sendBuf"_a, "recvBuf"_a, "recvCount"_a, "dataType"_a, "root"_a, "comm"_a,
          "stream"_a);
    m.def("HcclAllGather", &HcclAllGather, "sendBuf"_a, "recvBuf"_a, "sendCount"_a, "dataType"_a, "comm"_a, "stream"_a);
    m.def("HcclSend", &HcclSend, "sendBuf"_a, "count"_a, "dataType"_a, "destRank"_a, "comm"_a, "stream"_a);
    m.def("HcclRecv", &HcclRecv, "recvBuf"_a, "count"_a, "dataType"_a, "srcRank"_a, "comm"_a, "stream"_a);
    m.def("HcclAlltoAll", &HcclAlltoAll, "sendBuf"_a, "sendCount"_a, "sendType"_a, "recvBuf"_a, "recvCount"_a,
          "recvType"_a, "comm"_a, "stream"_a);
    m.def("HcclReduce", &HcclReduce, "sendBuf"_a, "recvBuf"_a, "count"_a, "dataType"_a, "op"_a, "root"_a, "comm"_a,
          "stream"_a);

    // V 变体：count/displ 以 Python 序列传入（内部转为 uint64 数组）。
    m.def(
        "HcclReduceScatterV",
        [](uintptr_t sendBuf, py::handle sendCounts, py::handle sendDispls, uintptr_t recvBuf, uint64_t recvCount,
           HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
        {
            auto sc = to_u64(sendCounts);
            auto sd = to_u64(sendDispls);
            return HcclReduceScatterV(sendBuf, sc.data(), sd.data(), recvBuf, recvCount, dataType, op, comm, stream);
        },
        "sendBuf"_a, "sendCounts"_a, "sendDispls"_a, "recvBuf"_a, "recvCount"_a, "dataType"_a, "op"_a, "comm"_a,
        "stream"_a);

    m.def(
        "HcclAllGatherV",
        [](uintptr_t sendBuf, uint64_t sendCount, uintptr_t recvBuf, py::handle recvCounts, py::handle recvDispls,
           HcclDataType dataType, HcclComm comm, aclrtStream stream)
        {
            auto rc = to_u64(recvCounts);
            auto rd = to_u64(recvDispls);
            return HcclAllGatherV(sendBuf, sendCount, recvBuf, rc.data(), rd.data(), dataType, comm, stream);
        },
        "sendBuf"_a, "sendCount"_a, "recvBuf"_a, "recvCounts"_a, "recvDispls"_a, "dataType"_a, "comm"_a, "stream"_a);

    m.def(
        "HcclAlltoAllVC",
        [](uintptr_t sendBuf, py::handle sendCountMatrix, HcclDataType sendType, uintptr_t recvBuf,
           HcclDataType recvType, HcclComm comm, aclrtStream stream)
        {
            auto mtx = to_u64(sendCountMatrix);
            return HcclAlltoAllVC(sendBuf, mtx.data(), sendType, recvBuf, recvType, comm, stream);
        },
        "sendBuf"_a, "sendCountMatrix"_a, "sendType"_a, "recvBuf"_a, "recvType"_a, "comm"_a, "stream"_a);

    m.def(
        "HcclAlltoAllV",
        [](uintptr_t sendBuf, py::handle sendCounts, py::handle sdispls, HcclDataType sendType, uintptr_t recvBuf,
           py::handle recvCounts, py::handle rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
        {
            auto sc = to_u64(sendCounts);
            auto sd = to_u64(sdispls);
            auto rc = to_u64(recvCounts);
            auto rd = to_u64(rdispls);
            return HcclAlltoAllV(sendBuf, sc.data(), sd.data(), sendType, recvBuf, rc.data(), rd.data(), recvType, comm,
                                 stream);
        },
        "sendBuf"_a, "sendCounts"_a, "sdispls"_a, "sendType"_a, "recvBuf"_a, "recvCounts"_a, "rdispls"_a, "recvType"_a,
        "comm"_a, "stream"_a);

    m.def(
        "HcclBatchSendRecv",
        [](py::handle items, HcclComm comm, aclrtStream stream)
        {
            std::vector<HcclSendRecvItem> v;
            for (auto item : items)
            {
                v.push_back(item.cast<HcclSendRecvItem>());
            }
            return HcclBatchSendRecv(v.data(), static_cast<uint32_t>(v.size()), comm, stream);
        },
        "sendRecvInfo"_a, "comm"_a, "stream"_a);

    // ---------- mc2 算子参数（hccl_mc2.h） ----------
    m.def("HcclKfcAllocOpArgs",
          [](void)
          {
              uintptr_t args = 0;
              HcclResult ret = HcclKfcAllocOpArgs(&args);
              return py::make_tuple(ret, args);
          });
    m.def("HcclKfcFreeOpArgs", &HcclKfcFreeOpArgs, "opArgs"_a);
    m.def("HcclKfcOpArgsSetSrcDataType", &HcclKfcOpArgsSetSrcDataType, "opArgs"_a, "srcDataType"_a);
    m.def("HcclKfcOpArgsSetDstDataType", &HcclKfcOpArgsSetDstDataType, "opArgs"_a, "dstDataType"_a);
    m.def("HcclKfcOpArgsSetReduceType", &HcclKfcOpArgsSetReduceType, "opArgs"_a, "reduceType"_a);
    m.def("HcclKfcOpArgsSetCount", &HcclKfcOpArgsSetCount, "opArgs"_a, "count"_a);
    m.def("HcclKfcOpArgsSetAlgConfig", &HcclKfcOpArgsSetAlgConfig, "opArgs"_a, "algConfig"_a);
    m.def("HcclKfcOpArgsSetCommEngine", &HcclKfcOpArgsSetCommEngine, "opArgs"_a, "commEngine"_a);
    m.def(
        "HcclCreateOpResCtx",
        [](HcclComm comm, uint8_t opType, uintptr_t opArgs)
        {
            uintptr_t ctx = 0;
            HcclResult ret = HcclCreateOpResCtx(comm, opType, opArgs, &ctx);
            return py::make_tuple(ret, ctx);
        },
        "comm"_a, "opType"_a, "opArgs"_a);

    // ---------- 通信域初始化与管理（hccl_comm.h） ----------
    m.def(
        "HcclCommInitClusterInfo",
        [](const std::string& clusterInfo, uint32_t rank)
        {
            HcclComm comm = 0;
            HcclResult ret = HcclCommInitClusterInfo(clusterInfo.c_str(), rank, &comm);
            return py::make_tuple(ret, comm);
        },
        "clusterInfo"_a, "rank"_a);
    m.def(
        "HcclCommInitClusterInfoConfig",
        [](const std::string& clusterInfo, uint32_t rank, HcclCommConfig& config)
        {
            HcclComm comm = 0;
            HcclResult ret = HcclCommInitClusterInfoConfig(clusterInfo.c_str(), rank, &config, &comm);
            return py::make_tuple(ret, comm);
        },
        "clusterInfo"_a, "rank"_a, "config"_a);
    m.def(
        "HcclCreateSubCommConfig",
        [](HcclComm comm, py::handle rankIds, uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig& config)
        {
            auto ids = to_u32(rankIds);
            HcclComm sub = 0;
            HcclResult ret = HcclCreateSubCommConfig(&comm, static_cast<uint32_t>(ids.size()), ids.data(), subCommId,
                                                     subCommRankId, &config, &sub);
            return py::make_tuple(ret, comm, sub);
        },
        "comm"_a, "rankIds"_a, "subCommId"_a, "subCommRankId"_a, "config"_a);
    m.def("HcclGetRootInfo",
          [](void)
          {
              HcclRootInfo info{};
              HcclResult ret = HcclGetRootInfo(&info);
              return py::make_tuple(ret, info);
          });
    m.def(
        "HcclCommInitRootInfo",
        [](uint32_t nRanks, const HcclRootInfo& rootInfo, uint32_t rank)
        {
            HcclComm comm = 0;
            HcclResult ret = HcclCommInitRootInfo(nRanks, &rootInfo, rank, &comm);
            return py::make_tuple(ret, comm);
        },
        "nRanks"_a, "rootInfo"_a, "rank"_a);
    m.def(
        "HcclCommInitRootInfoConfig",
        [](uint32_t nRanks, const HcclRootInfo& rootInfo, uint32_t rank, const HcclCommConfig& config)
        {
            HcclComm comm = 0;
            HcclResult ret = HcclCommInitRootInfoConfig(nRanks, &rootInfo, rank, &config, &comm);
            return py::make_tuple(ret, comm);
        },
        "nRanks"_a, "rootInfo"_a, "rank"_a, "config"_a);
    m.def("HcclGetRootInfoScalable",
          [](void)
          {
              HcclRootInfo info{};
              HcclResult ret = HcclGetRootInfoScalable(&info);
              return py::make_tuple(ret, info);
          });
    m.def(
        "HcclCommInitRootInfoScalable",
        [](uint32_t nRanks, uint32_t nRoot, py::handle rootInfoList, uint32_t rank, uint32_t nExtRoot,
           const HcclCommConfig& config)
        {
            std::vector<HcclRootInfo> infos;
            for (auto item : rootInfoList)
            {
                infos.push_back(item.cast<HcclRootInfo>());
            }
            HcclComm comm = 0;
            HcclResult ret = HcclCommInitRootInfoScalable(nRanks, nRoot, infos.data(), rank, nExtRoot, &config, &comm);
            return py::make_tuple(ret, comm);
        },
        "nRanks"_a, "nRoot"_a, "rootInfoList"_a, "rank"_a, "nExtRoot"_a, "config"_a);
    m.def("HcclSetConfig", &HcclSetConfig, "config"_a, "configValue"_a);
    m.def(
        "HcclGetConfig",
        [](HcclConfig config)
        {
            HcclConfigValue v{};
            HcclResult ret = HcclGetConfig(config, &v);
            return py::make_tuple(ret, v.value);
        },
        "config"_a);
    m.def(
        "HcclGetCommName",
        [](HcclComm comm)
        {
            char name[COMM_NAME_MAX_LENGTH] = {0};
            HcclResult ret = HcclGetCommName(comm, name);
            return py::make_tuple(ret, std::string(name));
        },
        "comm"_a);
    m.def(
        "HcclCommGetHandleWithName",
        [](const std::string& commName)
        {
            HcclComm comm = 0;
            HcclResult ret = HcclCommGetHandleWithName(commName.c_str(), &comm);
            return py::make_tuple(ret, comm);
        },
        "commName"_a);
    m.def(
        "HcclGetRankSize",
        [](HcclComm comm)
        {
            uint32_t n = 0;
            HcclResult ret = HcclGetRankSize(comm, &n);
            return py::make_tuple(ret, n);
        },
        "comm"_a);
    m.def(
        "HcclGetRankId",
        [](HcclComm comm)
        {
            uint32_t n = 0;
            HcclResult ret = HcclGetRankId(comm, &n);
            return py::make_tuple(ret, n);
        },
        "comm"_a);
    m.def("HcclBarrier", &HcclBarrier, "comm"_a, "stream"_a);
    m.def("HcclCommDestroy", &HcclCommDestroy, "comm"_a);
    m.def(
        "HcclCommInitAll",
        [](py::handle devices)
        {
            auto devs = to_i32(devices);
            std::vector<HcclComm> comms(devs.size(), 0);
            HcclResult ret = HcclCommInitAll(static_cast<uint32_t>(devs.size()), devs.data(), comms.data());
            py::list out;
            for (auto c : comms)
            {
                out.append(c);
            }
            return py::make_tuple(ret, out);
        },
        "devices"_a);
    m.def(
        "HcclGetCommAsyncError",
        [](HcclComm comm)
        {
            HcclResult err = HCCL_SUCCESS;
            HcclResult ret = HcclGetCommAsyncError(comm, &err);
            return py::make_tuple(ret, err);
        },
        "comm"_a);
    m.def("HcclGetErrorString", &HcclGetErrorString, "code"_a);
    m.def("HcclGetCommConfigCapability", &HcclGetCommConfigCapability);
    m.def("HcclCommSuspend", &HcclCommSuspend, "comm"_a);
    m.def("HcclCommRegCommStateCallback", &HcclCommRegCommStateCallback, "regName"_a, "cb"_a, "args"_a);
    m.def("HcclCommResume", &HcclCommResume, "comm"_a);
    m.def(
        "HcclCommGetStatus",
        [](const std::string& commId)
        {
            HcclCommStatus st = HCCL_COMM_STATUS_INVALID;
            HcclResult ret = HcclCommGetStatus(commId.c_str(), &st);
            return py::make_tuple(ret, static_cast<int32_t>(st));
        },
        "commId"_a);
    m.def("HcclCommSetMemoryRange", &HcclCommSetMemoryRange, "comm"_a, "baseVirPtr"_a, "size"_a, "alignment"_a,
          "flags"_a);
    m.def("HcclCommUnsetMemoryRange", &HcclCommUnsetMemoryRange, "comm"_a, "baseVirPtr"_a);
    m.def("HcclCommActivateCommMemory", &HcclCommActivateCommMemory, "comm"_a, "virPtr"_a, "size"_a, "offset"_a,
          "handle"_a, "flags"_a);
    m.def("HcclCommDeactivateCommMemory", &HcclCommDeactivateCommMemory, "comm"_a, "virPtr"_a);
    m.def(
        "HcclCommWorkingDevNicSet",
        [](HcclComm comm, py::handle ranks, py::handle useBackup)
        {
            auto r = to_u32(ranks);
            auto b = to_bool_bytes(useBackup);
            return HcclCommWorkingDevNicSet(comm, r.data(), reinterpret_cast<bool*>(b.data()),
                                            static_cast<uint32_t>(r.size()));
        },
        "comm"_a, "ranks"_a, "useBackup"_a);
    m.def("HcclGroupStart", &HcclGroupStart);
    m.def("HcclGroupEnd", &HcclGroupEnd);
    m.def("HcclGroupStatusGet",
          [](void)
          {
              bool en = false;
              HcclResult ret = HcclGroupStatusGet(&en);
              return py::make_tuple(ret, en);
          });
    m.def(
        "HcclGroupEnqueueOp", [](HcclComm comm, const HcclOpDesc& opDesc, aclrtStream stream, uintptr_t func)
        { return HcclGroupEnqueueOp(comm, &opDesc, stream, func); }, "comm"_a, "opDesc"_a, "stream"_a, "func"_a);
    m.def(
        "HcclCommSymWinRegister",
        [](HcclComm comm, uintptr_t addr, uint64_t size, uint32_t flag)
        {
            HcclCommSymWindow win = 0;
            HcclResult ret = HcclCommSymWinRegister(comm, addr, size, &win, flag);
            return py::make_tuple(ret, win);
        },
        "comm"_a, "addr"_a, "size"_a, "flag"_a);
    m.def("HcclCommSymWinDeregister", &HcclCommSymWinDeregister, "winHandle"_a);
    m.def(
        "HcclCommSymWinGet",
        [](HcclComm comm, uintptr_t ptr, size_t size)
        {
            HcclCommSymWindow win = 0;
            size_t offset = 0;
            HcclResult ret = HcclCommSymWinGet(comm, ptr, size, &win, &offset);
            return py::make_tuple(ret, win, offset);
        },
        "comm"_a, "ptr"_a, "size"_a);
    m.def(
        "HcclConfigGetInfo",
        [](HcclComm comm, HcclConfigType cfgType, uint32_t infoLen)
        {
            std::vector<uint8_t> info(infoLen, 0);
            HcclResult ret = HcclConfigGetInfo(comm, cfgType, infoLen, reinterpret_cast<uintptr_t>(info.data()));
            return py::make_tuple(ret, py::bytes(reinterpret_cast<const char*>(info.data()), info.size()));
        },
        "comm"_a, "cfgType"_a, "infoLen"_a);

    // ---------- 通信资源与上下文（hccl_res.h） ----------
    m.def(
        "HcclGetHcclBuffer",
        [](HcclComm comm)
        {
            uintptr_t buf = 0;
            uint64_t size = 0;
            HcclResult ret = HcclGetHcclBuffer(comm, &buf, &size);
            return py::make_tuple(ret, buf, size);
        },
        "comm"_a);
    m.def(
        "HcclGetHcclBufferCleared",
        [](HcclComm comm)
        {
            uintptr_t buf = 0;
            uint64_t size = 0;
            HcclResult ret = HcclGetHcclBufferCleared(comm, &buf, &size);
            return py::make_tuple(ret, buf, size);
        },
        "comm"_a);
    m.def(
        "HcclThreadAcquire",
        [](HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread)
        {
            std::vector<ThreadHandle> threads(threadNum, 0);
            HcclResult ret = HcclThreadAcquire(comm, engine, threadNum, notifyNumPerThread, threads.data());
            py::list out;
            for (auto t : threads)
            {
                out.append(t);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a, "engine"_a, "threadNum"_a, "notifyNumPerThread"_a);
    m.def(
        "HcclThreadAcquireWithConfig",
        [](HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, py::handle config)
        {
            std::vector<ThreadConfig> cfgs;
            for (auto item : config)
            {
                cfgs.push_back(item.cast<ThreadConfig>());
            }
            std::vector<ThreadHandle> threads(threadNum, 0);
            HcclResult ret = HcclThreadAcquireWithConfig(comm, engine, threadNum, type, cfgs.data(), threads.data());
            py::list out;
            for (auto t : threads)
            {
                out.append(t);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a, "engine"_a, "threadNum"_a, "type"_a, "config"_a);
    m.def(
        "HcclThreadAcquireWithStream",
        [](HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum)
        {
            ThreadHandle t = 0;
            HcclResult ret = HcclThreadAcquireWithStream(comm, engine, stream, notifyNum, &t);
            return py::make_tuple(ret, t);
        },
        "comm"_a, "engine"_a, "stream"_a, "notifyNum"_a);
    m.def(
        "HcclDedicatedThreadAcquire",
        [](HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread)
        {
            ThreadHandle t = 0;
            HcclResult ret = HcclDedicatedThreadAcquire(comm, useType, notifyNumPerThread, &t);
            return py::make_tuple(ret, t);
        },
        "comm"_a, "useType"_a, "notifyNumPerThread"_a);
    m.def(
        "HcclEngineCtxCreate",
        [](HcclComm comm, const std::string& ctxTag, CommEngine engine, uint64_t size)
        {
            uintptr_t ctx = 0;
            HcclResult ret = HcclEngineCtxCreate(comm, ctxTag.c_str(), engine, size, &ctx);
            return py::make_tuple(ret, ctx);
        },
        "comm"_a, "ctxTag"_a, "engine"_a, "size"_a);
    m.def(
        "HcclEngineCtxGet",
        [](HcclComm comm, const std::string& ctxTag, CommEngine engine)
        {
            uintptr_t ctx = 0;
            uint64_t size = 0;
            HcclResult ret = HcclEngineCtxGet(comm, ctxTag.c_str(), engine, &ctx, &size);
            return py::make_tuple(ret, ctx, size);
        },
        "comm"_a, "ctxTag"_a, "engine"_a);
    m.def("HcclEngineCtxCopy", &HcclEngineCtxCopy, "comm"_a, "engine"_a, "ctxTag"_a, "srcCtx"_a, "size"_a,
          "dstCtxOffset"_a);
    m.def("HcclEngineCtxDestroy", &HcclEngineCtxDestroy, "comm"_a, "ctxTag"_a, "engine"_a);
    m.def(
        "HcclCommMemReg",
        [](HcclComm comm, const std::string& memTag, const CommMem& mem)
        {
            HcclMemHandle h = 0;
            HcclResult ret = HcclCommMemReg(comm, memTag.c_str(), &mem, &h);
            return py::make_tuple(ret, h);
        },
        "comm"_a, "memTag"_a, "mem"_a);
    m.def(
        "HcclThreadResGetInfo",
        [](HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen)
        {
            uintptr_t info = 0;
            HcclResult ret = HcclThreadResGetInfo(comm, thread, resType, infoLen, &info);
            return py::make_tuple(ret, info);
        },
        "comm"_a, "thread"_a, "resType"_a, "infoLen"_a);

    // ---------- 通信通道管理（hccl_channel.h） ----------
    m.def(
        "HcclChannelAcquire",
        [](HcclComm comm, CommEngine engine, py::handle channelDescs)
        {
            std::vector<HcclChannelDesc> descs;
            for (auto item : channelDescs)
            {
                descs.push_back(item.cast<HcclChannelDesc>());
            }
            std::vector<ChannelHandle> channels(descs.size(), 0);
            HcclResult ret =
                HcclChannelAcquire(comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data());
            py::list out;
            for (auto c : channels)
            {
                out.append(c);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a, "engine"_a, "channelDescs"_a);
    m.def("HcclChannelConfigCreate",
          [](void)
          {
              HcclChannelConfig c = 0;
              HcclResult ret = HcclChannelConfigCreate(&c);
              return py::make_tuple(ret, c);
          });
    m.def("HcclChannelConfigDestroy", &HcclChannelConfigDestroy, "config"_a);
    m.def("HcclChannelConfigSetInt", &HcclChannelConfigSetInt, "config"_a, "type"_a, "value"_a);
    m.def("HcclChannelConfigSetStr", &HcclChannelConfigSetStr, "config"_a, "type"_a, "value"_a);
    m.def(
        "HcclChannelAcquireWithConfig",
        [](HcclComm comm, CommEngine engine, py::handle channelDescs, HcclChannelConfig config)
        {
            std::vector<HcclChannelDesc> descs;
            for (auto item : channelDescs)
            {
                descs.push_back(item.cast<HcclChannelDesc>());
            }
            std::vector<ChannelHandle> channels(descs.size(), 0);
            HcclResult ret = HcclChannelAcquireWithConfig(comm, engine, descs.data(),
                                                          static_cast<uint32_t>(descs.size()), config, channels.data());
            py::list out;
            for (auto c : channels)
            {
                out.append(c);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a, "engine"_a, "channelDescs"_a, "config"_a);
    m.def(
        "HcclChannelQuery",
        [](HcclComm comm, CommEngine engine, py::handle channelDescs)
        {
            std::vector<HcclChannelDesc> descs;
            for (auto item : channelDescs)
            {
                descs.push_back(item.cast<HcclChannelDesc>());
            }
            std::vector<ChannelHandle> channels(descs.size(), 0);
            HcclResult ret =
                HcclChannelQuery(comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data());
            py::list out;
            for (auto c : channels)
            {
                out.append(c);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a, "engine"_a, "channelDescs"_a);
    m.def(
        "HcclChannelDestroy",
        [](HcclComm comm, py::handle channels)
        {
            auto ch = to_u64(channels);
            return HcclChannelDestroy(comm, ch.data(), static_cast<uint32_t>(ch.size()));
        },
        "comm"_a, "channels"_a);
    m.def(
        "HcclChannelGetHcclBuffer",
        [](HcclComm comm, ChannelHandle channel)
        {
            uintptr_t buf = 0;
            uint64_t size = 0;
            HcclResult ret = HcclChannelGetHcclBuffer(comm, channel, &buf, &size);
            return py::make_tuple(ret, buf, size);
        },
        "comm"_a, "channel"_a);
    m.def(
        "HcclChannelGetRemoteMems",
        [](HcclComm comm, ChannelHandle channel)
        {
            uint32_t n = 0;
            CommMem* mems = nullptr;
            char** tags = nullptr;
            HcclResult ret = HcclChannelGetRemoteMems(comm, channel, &n, &mems, &tags);
            py::list ml;
            py::list tl;
            if (ret == HCCL_SUCCESS)
            {
                for (uint32_t i = 0; i < n; ++i)
                {
                    ml.append(mems[i]);
                    tl.append(tags && tags[i] ? py::str(tags[i]) : py::str(""));
                }
            }
            return py::make_tuple(ret, n, ml, tl);
        },
        "comm"_a, "channel"_a);

    // ---------- Kernel 启动（hccl_launch.h） ----------
    m.def(
        "HcclAicpuKernelLaunch",
        [](HcclComm comm, const HcclOpDesc& opInfo, const HcclKernelFuncInfo& funcInfo, ThreadHandle aicpuThreadHandle,
           aclrtStream userStream, const HcclKernelLaunchCfg& cfg)
        { return HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &cfg); },
        "comm"_a, "opInfo"_a, "funcInfo"_a, "aicpuThreadHandle"_a, "userStream"_a, "kernelLaunchCfg"_a);

    // ---------- 通信拓扑查询（hccl_rank_graph.h） ----------
    m.def(
        "HcclRankGraphGetLayers",
        [](HcclComm comm)
        {
            uint32_t* p = nullptr;
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetLayers(comm, &p, &n);
            return py::make_tuple(ret, u32list(p, n));
        },
        "comm"_a);
    m.def(
        "HcclRankGraphGetRanksByLayer",
        [](HcclComm comm, uint32_t netLayer)
        {
            uint32_t* p = nullptr;
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetRanksByLayer(comm, netLayer, &p, &n);
            return py::make_tuple(ret, u32list(p, n));
        },
        "comm"_a, "netLayer"_a);
    m.def(
        "HcclRankGraphGetRankSizeByLayer",
        [](HcclComm comm, uint32_t netLayer)
        {
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetRankSizeByLayer(comm, netLayer, &n);
            return py::make_tuple(ret, n);
        },
        "comm"_a, "netLayer"_a);
    m.def(
        "HcclRankGraphGetTopoTypeByLayer",
        [](HcclComm comm, uint32_t netLayer)
        {
            CommTopo t = COMM_TOPO_RESERVED;
            HcclResult ret = HcclRankGraphGetTopoTypeByLayer(comm, netLayer, &t);
            return py::make_tuple(ret, static_cast<int32_t>(t));
        },
        "comm"_a, "netLayer"_a);
    m.def(
        "HcclRankGraphGetInstSizeListByLayer",
        [](HcclComm comm, uint32_t netLayer)
        {
            uint32_t* p = nullptr;
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetInstSizeListByLayer(comm, netLayer, &p, &n);
            return py::make_tuple(ret, u32list(p, n));
        },
        "comm"_a, "netLayer"_a);
    m.def(
        "HcclRankGraphGetLinks",
        [](HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank)
        {
            CommLink* links = nullptr;
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &n);
            py::list out;
            if (ret == HCCL_SUCCESS)
            {
                for (uint32_t i = 0; i < n; ++i)
                {
                    out.append(links[i]);
                }
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a, "netLayer"_a, "srcRank"_a, "dstRank"_a);
    m.def(
        "HcclRankGraphGetTopoInstsByLayer",
        [](HcclComm comm, uint32_t netLayer)
        {
            uint32_t* p = nullptr;
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &p, &n);
            return py::make_tuple(ret, u32list(p, n));
        },
        "comm"_a, "netLayer"_a);
    m.def(
        "HcclRankGraphGetTopoType",
        [](HcclComm comm, uint32_t netLayer, uint32_t topoInstId)
        {
            CommTopo t = COMM_TOPO_RESERVED;
            HcclResult ret = HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &t);
            return py::make_tuple(ret, static_cast<int32_t>(t));
        },
        "comm"_a, "netLayer"_a, "topoInstId"_a);
    m.def(
        "HcclRankGraphGetRanksByTopoInst",
        [](HcclComm comm, uint32_t netLayer, uint32_t topoInstId)
        {
            uint32_t* p = nullptr;
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &p, &n);
            return py::make_tuple(ret, u32list(p, n));
        },
        "comm"_a, "netLayer"_a, "topoInstId"_a);
    m.def(
        "HcclGetHeterogMode",
        [](HcclComm comm)
        {
            HcclHeterogMode mode = HCCL_HETEROG_MODE_INVALID;
            HcclResult ret = HcclGetHeterogMode(comm, &mode);
            return py::make_tuple(ret, static_cast<int32_t>(mode));
        },
        "comm"_a);
    m.def(
        "HcclRankGraphGetEndpointNum",
        [](HcclComm comm, uint32_t layer, uint32_t topoInstId)
        {
            uint32_t n = 0;
            HcclResult ret = HcclRankGraphGetEndpointNum(comm, layer, topoInstId, &n);
            return py::make_tuple(ret, n);
        },
        "comm"_a, "layer"_a, "topoInstId"_a);
    m.def(
        "HcclRankGraphGetEndpointDesc",
        [](HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t descNum)
        {
            std::vector<EndpointDesc> descs(descNum);
            uint32_t n = descNum;
            HcclResult ret = HcclRankGraphGetEndpointDesc(comm, layer, topoInstId, &n, descs.data());
            py::list out;
            for (uint32_t i = 0; i < n; ++i)
            {
                out.append(descs[i]);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a, "layer"_a, "topoInstId"_a, "descNum"_a);
    m.def(
        "HcclRankGraphGetEndpointInfo",
        [](HcclComm comm, uint32_t rankId, const EndpointDesc& endpointDesc, EndpointAttr endpointAttr,
           uint32_t infoLen)
        {
            // 调用方无需自行分配缓冲区，这里分配 infoLen 字节并返回其内容。
            std::vector<uint8_t> info(infoLen, 0);
            HcclResult ret = HcclRankGraphGetEndpointInfo(comm, rankId, &endpointDesc, endpointAttr, infoLen,
                                                          reinterpret_cast<uintptr_t>(info.data()));
            return py::make_tuple(ret, py::bytes(reinterpret_cast<const char*>(info.data()), info.size()));
        },
        "comm"_a, "rankId"_a, "endpointDesc"_a, "endpointAttr"_a, "infoLen"_a);

    // ---------- 通信团队（hccl_team.h） ----------
    m.def(
        "HcclTeamCreate",
        [](HcclComm comm, const HcclTeamCreateDesc& desc)
        {
            HcommTeamHandle team = 0;
            HcclResult ret = HcclTeamCreate(comm, &desc, &team);
            return py::make_tuple(ret, team);
        },
        "comm"_a, "desc"_a);
    m.def("HcclTeamDestroy", &HcclTeamDestroy, "team"_a);
    m.def(
        "HcclTeamGetLsaTeam",
        [](HcclComm comm)
        {
            HcommTeamHandle team = 0;
            HcclResult ret = HcclTeamGetLsaTeam(comm, &team);
            return py::make_tuple(ret, team);
        },
        "comm"_a);
    m.def(
        "HcclTeamMemberToRank",
        [](HcclComm comm, HcommTeamHandle team, uint32_t memberId)
        {
            uint32_t rankId = 0;
            HcclResult ret = HcclTeamMemberToRank(comm, team, memberId, &rankId);
            return py::make_tuple(ret, rankId);
        },
        "comm"_a, "team"_a, "memberId"_a);
    m.def(
        "HcclTeamRankToMember",
        [](HcclComm comm, HcommTeamHandle team, uint32_t rankId)
        {
            uint32_t memberId = 0;
            HcclResult ret = HcclTeamRankToMember(comm, team, rankId, &memberId);
            return py::make_tuple(ret, memberId);
        },
        "comm"_a, "team"_a, "rankId"_a);

    // ---------- CCU 实例绑定（hccl_ccu_res.h） ----------
    m.def("HcclCommAssignCcuIns", &HcclCommAssignCcuIns, "comm"_a, "insHandle"_a);
    m.def(
        "HcclCommQueryCcuIns",
        [](HcclComm comm)
        {
            CcuInsHandle h = 0;
            uint32_t n = 0;
            HcclResult ret = HcclCommQueryCcuIns(comm, &h, &n);
            py::list out;
            for (uint32_t i = 0; i < n; ++i)
            {
                out.append(h);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a);
    m.def(
        "HcclCommQueryAssignedCcuIns",
        [](HcclComm comm)
        {
            CcuInsHandle h = 0;
            uint32_t n = 0;
            HcclResult ret = HcclCommQueryAssignedCcuIns(comm, &h, &n);
            py::list out;
            for (uint32_t i = 0; i < n; ++i)
            {
                out.append(h);
            }
            return py::make_tuple(ret, out);
        },
        "comm"_a);

    // ---------- 对称内存窗口（hccl_sym_win.h） ----------
    m.def(
        "HcclSymWinGetPeerPointer",
        [](HcclCommSymWindow win, size_t offset, uint32_t peerRank)
        {
            uintptr_t ptr = 0;
            HcclResult ret = HcclSymWinGetPeerPointer(win, offset, peerRank, &ptr);
            return py::make_tuple(ret, ptr);
        },
        "winHandle"_a, "offset"_a, "peerRank"_a);
    m.def(
        "HcclSymWinGetRemoteAddr",
        [](HcclCommSymWindow win, size_t offset, uint32_t peerRank)
        {
            uintptr_t ptr = 0;
            HcclResult ret = HcclSymWinGetRemoteAddr(win, offset, peerRank, &ptr);
            return py::make_tuple(ret, ptr);
        },
        "winHandle"_a, "offset"_a, "peerRank"_a);

    // ---------- HCOMM 基础资源（hcomm_res.h） ----------
    m.def(
        "HcommEndpointCreate",
        [](const EndpointDesc& endpoint)
        {
            EndpointHandle h = 0;
            HcommResult ret = HcommEndpointCreate(&endpoint, &h);
            return py::make_tuple(ret, h);
        },
        "endpoint"_a);
    m.def("HcommEndpointDestroy", &HcommEndpointDestroy, "endpointHandle"_a);
    m.def(
        "HcommEndpointGetDescNum",
        [](int32_t deviceLogicId)
        {
            uint32_t n = 0;
            HcommResult ret = HcommEndpointGetDescNum(deviceLogicId, &n);
            return py::make_tuple(ret, n);
        },
        "deviceLogicId"_a);
    m.def(
        "HcommEndpointGetDescs",
        [](int32_t deviceLogicId, uint32_t descNum)
        {
            std::vector<EndpointDesc> descs(descNum);
            uint32_t n = descNum;
            HcommResult ret = HcommEndpointGetDescs(deviceLogicId, &n, descs.data());
            py::list out;
            for (uint32_t i = 0; i < n; ++i)
            {
                out.append(descs[i]);
            }
            return py::make_tuple(ret, out);
        },
        "deviceLogicId"_a, "descNum"_a);
    m.def(
        "HcommMemReg",
        [](EndpointHandle ep, const std::string& memTag, const CommMem& mem)
        {
            HcommMemHandle h = 0;
            HcommResult ret = HcommMemReg(ep, memTag.c_str(), &mem, &h);
            return py::make_tuple(ret, h);
        },
        "endpointHandle"_a, "memTag"_a, "mem"_a);
    m.def("HcommMemUnreg", &HcommMemUnreg, "endpointHandle"_a, "memHandle"_a);
    m.def(
        "HcommMemExport",
        [](EndpointHandle ep, HcommMemHandle memHandle)
        {
            uintptr_t desc = 0;
            uint32_t len = 0;
            HcommResult ret = HcommMemExport(ep, memHandle, &desc, &len);
            return py::make_tuple(ret, desc, len);
        },
        "endpointHandle"_a, "memHandle"_a);
    m.def(
        "HcommMemImport",
        [](EndpointHandle ep, uintptr_t memDesc, uint32_t descLen)
        {
            CommMem mem{};
            HcommResult ret = HcommMemImport(ep, memDesc, descLen, &mem);
            return py::make_tuple(ret, mem);
        },
        "endpointHandle"_a, "memDesc"_a, "descLen"_a);
    m.def("HcommMemUnimport", &HcommMemUnimport, "endpointHandle"_a, "memDesc"_a, "descLen"_a);
    m.def(
        "HcommThreadAlloc",
        [](CommEngine engine, py::handle notifyNumPerThread)
        {
            auto n = to_u32(notifyNumPerThread);
            std::vector<ThreadHandle> threads(n.size(), 0);
            HcommResult ret = HcommThreadAlloc(engine, static_cast<uint32_t>(n.size()), n.data(), threads.data());
            py::list out;
            for (auto t : threads)
            {
                out.append(t);
            }
            return py::make_tuple(ret, out);
        },
        "engine"_a, "notifyNumPerThread"_a);
    m.def(
        "HcommThreadFree",
        [](py::handle threads)
        {
            auto t = to_u64(threads);
            return HcommThreadFree(t.data(), static_cast<uint32_t>(t.size()));
        },
        "threads"_a);
    m.def(
        "HcommEndpointGetListenPort",
        [](EndpointHandle ep)
        {
            uint32_t port = 0;
            HcommResult ret = HcommEndpointGetListenPort(ep, &port);
            return py::make_tuple(ret, port);
        },
        "endpointHandle"_a);
    m.def(
        "HcommThreadResGetInfo",
        [](ThreadHandle thread, ThreadResType resType, uint32_t infoLen)
        {
            uintptr_t info = 0;
            HcommResult ret = HcommThreadResGetInfo(thread, resType, infoLen, &info);
            return py::make_tuple(ret, info);
        },
        "thread"_a, "resType"_a, "infoLen"_a);
    m.def(
        "HcommEndpointCheckFeature",
        [](HcommEndpointFeatureType featureType, const EndpointDesc& endpointDesc)
        {
            bool value = false;
            HcommResult ret = HcommEndpointCheckFeature(featureType, &endpointDesc, &value);
            return py::make_tuple(ret, value);
        },
        "featureType"_a, "endpointDesc"_a);
    m.def(
        "HcommMemAlloc",
        [](size_t size)
        {
            uintptr_t ptr = 0;
            HcommResult ret = HcommMemAlloc(&ptr, size);
            return py::make_tuple(ret, ptr);
        },
        "size"_a);
    m.def("HcommMemFree", &HcommMemFree, "ptr"_a);

    // ---------- HCOMM 通道（hcomm_channel.h） ----------
    m.def(
        "HcommChannelCreate",
        [](EndpointHandle ep, CommEngine engine, py::handle channelDescs)
        {
            std::vector<HcommChannelDesc> descs;
            for (auto item : channelDescs)
            {
                descs.push_back(item.cast<HcommChannelDesc>());
            }
            std::vector<ChannelHandle> channels(descs.size(), 0);
            HcommResult ret =
                HcommChannelCreate(ep, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data());
            py::list out;
            for (auto c : channels)
            {
                out.append(c);
            }
            return py::make_tuple(ret, out);
        },
        "endpointHandle"_a, "engine"_a, "channelDescs"_a);
    m.def("HcommChannelConfigCreate",
          [](void)
          {
              HcommChannelConfig c = 0;
              HcommResult ret = HcommChannelConfigCreate(&c);
              return py::make_tuple(ret, c);
          });
    m.def("HcommChannelConfigDestroy", &HcommChannelConfigDestroy, "config"_a);
    m.def("HcommChannelConfigSetInt", &HcommChannelConfigSetInt, "config"_a, "type"_a, "value"_a);
    m.def(
        "HcommChannelCreateWithConfig",
        [](EndpointHandle ep, CommEngine engine, py::handle channelDescs, HcommChannelConfig config)
        {
            std::vector<HcommChannelDesc> descs;
            for (auto item : channelDescs)
            {
                descs.push_back(item.cast<HcommChannelDesc>());
            }
            std::vector<ChannelHandle> channels(descs.size(), 0);
            HcommResult ret = HcommChannelCreateWithConfig(
                ep, engine, descs.data(), static_cast<uint32_t>(descs.size()), config, channels.data());
            py::list out;
            for (auto c : channels)
            {
                out.append(c);
            }
            return py::make_tuple(ret, out);
        },
        "endpointHandle"_a, "engine"_a, "channelDescs"_a, "config"_a);
    m.def(
        "HcommChannelGetStatus",
        [](py::handle channelList)
        {
            auto ch = to_u64(channelList);
            std::vector<int32_t> status(ch.size(), 0);
            HcommResult ret = HcommChannelGetStatus(ch.data(), static_cast<uint32_t>(ch.size()), status.data());
            py::list out;
            for (auto s : status)
            {
                out.append(s);
            }
            return py::make_tuple(ret, out);
        },
        "channelList"_a);
    m.def(
        "HcommChannelDestroy",
        [](py::handle channels)
        {
            auto ch = to_u64(channels);
            return HcommChannelDestroy(ch.data(), static_cast<uint32_t>(ch.size()));
        },
        "channels"_a);

    // ---------- HCOMM 数据面原语（hcomm_primitives.h） ----------
    m.def("HcommLocalCopyOnThread", &HcommLocalCopyOnThread, "thread"_a, "dst"_a, "src"_a, "len"_a);
    m.def("HcommLocalReduceOnThread", &HcommLocalReduceOnThread, "thread"_a, "dst"_a, "src"_a, "count"_a, "dataType"_a,
          "reduceOp"_a);
    m.def("HcommThreadNotifyRecordOnThread", &HcommThreadNotifyRecordOnThread, "thread"_a, "dstThread"_a,
          "dstNotifyIdx"_a);
    m.def("HcommThreadNotifyWaitOnThread", &HcommThreadNotifyWaitOnThread, "thread"_a, "notifyIdx"_a, "timeOut"_a);
    m.def("HcommAclrtNotifyRecordOnThread", &HcommAclrtNotifyRecordOnThread, "thread"_a, "dstNotifyId"_a);
    m.def("HcommAclrtNotifyWaitOnThread", &HcommAclrtNotifyWaitOnThread, "thread"_a, "notifyId"_a, "timeOut"_a);
    m.def("HcommWriteOnThread", &HcommWriteOnThread, "thread"_a, "channel"_a, "dst"_a, "src"_a, "len"_a);
    m.def("HcommWriteReduceOnThread", &HcommWriteReduceOnThread, "thread"_a, "channel"_a, "dst"_a, "src"_a, "count"_a,
          "dataType"_a, "reduceOp"_a);
    m.def("HcommWriteWithNotifyOnThread", &HcommWriteWithNotifyOnThread, "thread"_a, "channel"_a, "dst"_a, "src"_a,
          "len"_a, "remoteNotifyIdx"_a);
    m.def("HcommWriteReduceWithNotifyOnThread", &HcommWriteReduceWithNotifyOnThread, "thread"_a, "channel"_a, "dst"_a,
          "src"_a, "count"_a, "dataType"_a, "reduceOp"_a, "remoteNotifyIdx"_a);
    m.def("HcommReadOnThread", &HcommReadOnThread, "thread"_a, "channel"_a, "dst"_a, "src"_a, "len"_a);
    m.def("HcommReadReduceOnThread", &HcommReadReduceOnThread, "thread"_a, "channel"_a, "dst"_a, "src"_a, "count"_a,
          "dataType"_a, "reduceOp"_a);
    m.def("HcommWriteNbiOnThread", &HcommWriteNbiOnThread, "thread"_a, "channel"_a, "dst"_a, "src"_a, "len"_a);
    m.def("HcommWriteNbi", &HcommWriteNbi, "channel"_a, "dst"_a, "src"_a, "len"_a);
    m.def("HcommWriteWithNotifyNbiOnThread", &HcommWriteWithNotifyNbiOnThread, "thread"_a, "channel"_a, "dst"_a,
          "src"_a, "len"_a, "remoteNotifyIdx"_a);
    m.def("HcommWriteWithNotifyNbi", &HcommWriteWithNotifyNbi, "channel"_a, "dst"_a, "src"_a, "len"_a,
          "remoteNotifyIdx"_a);
    m.def("HcommReadNbiOnThread", &HcommReadNbiOnThread, "thread"_a, "channel"_a, "dst"_a, "src"_a, "len"_a);
    m.def("HcommReadNbi", &HcommReadNbi, "channel"_a, "dst"_a, "src"_a, "len"_a);
    m.def(
        "HcommBatchTransferOnThread",
        [](ThreadHandle thread, ChannelHandle channel, py::handle transferDescs)
        {
            std::vector<HcommBatchTransferDesc> descs;
            for (auto item : transferDescs)
            {
                descs.push_back(item.cast<HcommBatchTransferDesc>());
            }
            return HcommBatchTransferOnThread(thread, channel, descs.data(), static_cast<uint32_t>(descs.size()));
        },
        "thread"_a, "channel"_a, "transferDescs"_a);
    m.def("HcommChannelNotifyRecordOnThread", &HcommChannelNotifyRecordOnThread, "thread"_a, "channel"_a,
          "remoteNotifyIdx"_a);
    m.def("HcommChannelNotifyRecord", &HcommChannelNotifyRecord, "channel"_a, "remoteNotifyIdx"_a);
    m.def("HcommChannelNotifyWaitOnThread", &HcommChannelNotifyWaitOnThread, "thread"_a, "channel"_a,
          "localNotifyIdx"_a, "timeOut"_a);
    m.def("HcommChannelNotifyWait", &HcommChannelNotifyWait, "channel"_a, "localNotifyIdx"_a, "timeOut"_a);
    m.def(
        "HcommAicpuTsTaskCacheLookup",
        [](const std::string& tag)
        {
            bool hit = false;
            HcommResult ret = HcommAicpuTsTaskCacheLookup(tag.c_str(), &hit);
            return py::make_tuple(ret, hit);
        },
        "tag"_a);
    m.def(
        "HcommAicpuTsTaskCacheStart",
        [](const std::string& tag, py::handle addrs, py::handle sizes)
        {
            auto a = to_ptr(addrs);
            auto s = to_u64(sizes);
            return HcommAicpuTsTaskCacheStart(tag.c_str(), a.data(), s.data(), a.size());
        },
        "tag"_a, "addrs"_a, "sizes"_a);
    m.def("HcommAicpuTsTaskCacheEnd", &HcommAicpuTsTaskCacheEnd, "tag"_a);
    m.def(
        "HcommAicpuTsTaskCacheExecute",
        [](const std::string& tag, py::handle addrs, py::handle sizes)
        {
            auto a = to_ptr(addrs);
            auto s = to_u64(sizes);
            return HcommAicpuTsTaskCacheExecute(tag.c_str(), a.data(), s.data(), a.size());
        },
        "tag"_a, "addrs"_a, "sizes"_a);
    m.def("HcommAicpuTsTaskCacheClear", &HcommAicpuTsTaskCacheClear, "tag"_a);
    m.def("HcommBatchModeStart", &HcommBatchModeStart, "batchTag"_a);
    m.def("HcommBatchModeEnd", &HcommBatchModeEnd, "batchTag"_a);
    m.def("HcommAcquireComm", &HcommAcquireComm, "commId"_a);
    m.def("HcommReleaseComm", &HcommReleaseComm, "commId"_a);
    m.def("HcommFenceOnThread", &HcommFenceOnThread, "thread"_a);
    m.def("HcommChannelFenceOnThread", &HcommChannelFenceOnThread, "thread"_a, "channel"_a);
    m.def("HcommChannelFence", &HcommChannelFence, "channel"_a);
    m.def("HcommSetNotifyWaitTimeOut", &HcommSetNotifyWaitTimeOut, "timeOut"_a);
    m.def("HcommThreadResAcquireTimeOut", &HcommThreadResAcquireTimeOut, "timeOut"_a);
    m.def("HcommChannelNotifyWaitOnThreadWithDefaultTimeout", &HcommChannelNotifyWaitOnThreadWithDefaultTimeout,
          "thread"_a, "channel"_a, "localNotifyIdx"_a);
    m.def("HcommThreadNotifyWaitOnThreadWithDefaultTimeout", &HcommThreadNotifyWaitOnThreadWithDefaultTimeout,
          "thread"_a, "notifyIdx"_a);
    m.def("HcommChannelDrainOnThread", &HcommChannelDrainOnThread, "thread"_a, "channel"_a);

    // ---------- CCU 资源（ccu_res.h） ----------
    m.def(
        "HcommCcuVariableAlloc",
        [](CcuInsHandle ins, uint8_t dieId, uint32_t num)
        {
            CcuVariableHandle h = 0;
            CcuResult ret = HcommCcuVariableAlloc(ins, dieId, num, &h);
            return py::make_tuple(ret, h);
        },
        "insHandle"_a, "dieId"_a, "num"_a);
    m.def(
        "HcommCcuEventAlloc",
        [](CcuInsHandle ins, uint8_t dieId, uint32_t num)
        {
            CcuEventHandle h = 0;
            CcuResult ret = HcommCcuEventAlloc(ins, dieId, num, &h);
            return py::make_tuple(ret, h);
        },
        "insHandle"_a, "dieId"_a, "num"_a);
    m.def(
        "HcommCcuVariableGetAddr",
        [](CcuVariableHandle h, uint32_t index)
        {
            uint64_t va = 0;
            CcuResult ret = HcommCcuVariableGetAddr(h, index, &va);
            return py::make_tuple(ret, va);
        },
        "varHandle"_a, "index"_a);
    m.def(
        "HcommCcuEventGetAddr",
        [](CcuEventHandle h, uint32_t index)
        {
            uint64_t va = 0;
            CcuResult ret = HcommCcuEventGetAddr(h, index, &va);
            return py::make_tuple(ret, va);
        },
        "eventHandle"_a, "index"_a);
    m.def(
        "HcommCcuInsResDescCreate",
        [](uint32_t dieId)
        {
            HcommCcuResDescHandle h = 0;
            CcuResult ret = HcommCcuInsResDescCreate(dieId, &h);
            return py::make_tuple(ret, h);
        },
        "dieId"_a);
    m.def("HcommCcuInsResDescDestroy", &HcommCcuInsResDescDestroy, "resDesc"_a);
    m.def(
        "HcommCcuInsResDescQueryDieId",
        [](HcommCcuResDescHandle resDesc)
        {
            uint32_t dieId = 0;
            CcuResult ret = HcommCcuInsResDescQueryDieId(resDesc, &dieId);
            return py::make_tuple(ret, dieId);
        },
        "resDesc"_a);
    m.def("HcommCcuInsResDescSetNum", &HcommCcuInsResDescSetNum, "resDesc"_a, "resType"_a, "resNum"_a);
    m.def(
        "HcommCcuInsResDescQueryNum",
        [](HcommCcuResDescHandle resDesc, HcommCcuResType resType)
        {
            uint32_t num = 0;
            CcuResult ret = HcommCcuInsResDescQueryNum(resDesc, resType, &num);
            return py::make_tuple(ret, num);
        },
        "resDesc"_a, "resType"_a);
    m.def(
        "HcommCcuInsCreate",
        [](py::handle resDescs)
        {
            auto r = to_u64(resDescs);
            CcuInsHandle ins = 0;
            CcuResult ret = HcommCcuInsCreate(r.data(), static_cast<uint32_t>(r.size()), &ins);
            return py::make_tuple(ret, ins);
        },
        "resDescs"_a);
    m.def(
        "HcommCcuInsCreateDefault",
        [](py::handle dieIds)
        {
            auto d = to_u32(dieIds);
            CcuInsHandle ins = 0;
            CcuResult ret = HcommCcuInsCreateDefault(d.data(), static_cast<uint32_t>(d.size()), &ins);
            return py::make_tuple(ret, ins);
        },
        "dieIds"_a);
    m.def("HcommCcuInsDestroy", &HcommCcuInsDestroy, "ccuInsHandle"_a);
    m.def("HcommCcuInsQueryResDesc", &HcommCcuInsQueryResDesc, "ccuInsHandle"_a, "resDesc"_a);
    m.def("HcommCcuQueryRemainResDesc", &HcommCcuQueryRemainResDesc, "resDesc"_a);
    m.def(
        "HcommCcuKernelQueryResReq",
        [](uintptr_t kernelFunc, py::handle kernelArgs, HcommCcuResDescHandle resDesc)
        {
            auto a = to_ptr(kernelArgs);
            return HcommCcuKernelQueryResReq(kernelFunc, a.data(), static_cast<uint32_t>(a.size()), resDesc);
        },
        "kernelFunc"_a, "kernelArgs"_a, "resDesc"_a);
    m.def(
        "HcommCcuGetMemToken",
        [](uint64_t srcVa, uint64_t size)
        {
            uint64_t token = 0;
            CcuResult ret = HcommCcuGetMemToken(srcVa, size, &token);
            return py::make_tuple(ret, token);
        },
        "srcVa"_a, "size"_a);
    m.def(
        "HcommCcuCascCntAlloc",
        [](CcuInsHandle ins, uint8_t dieId)
        {
            HcommCcuCascCntHandle h = 0;
            CcuResult ret = HcommCcuCascCntAlloc(ins, dieId, &h);
            return py::make_tuple(ret, h);
        },
        "ccuInsHandle"_a, "dieId"_a);
    m.def(
        "HcommCcuCascCntGetMem",
        [](HcommCcuCascCntHandle handle)
        {
            CommMem mem{};
            CcuResult ret = HcommCcuCascCntGetMem(handle, &mem);
            return py::make_tuple(ret, mem);
        },
        "handle"_a);
    m.def(
        "HcommCcuGetRmtMemToken",
        [](ChannelHandle channel, uint64_t srcVa)
        {
            uint64_t token = 0;
            CcuResult ret = HcommCcuGetRmtMemToken(channel, srcVa, &token);
            return py::make_tuple(ret, token);
        },
        "channelHandle"_a, "srcVa"_a);

    // ---------- CCU Kernel 注册与启动（ccu_launch.h） ----------
    m.def("HcommCcuKernelRegisterStart", &HcommCcuKernelRegisterStart, "insHandle"_a);
    m.def(
        "HcommCcuKernelRegister",
        [](CcuInsHandle ins, uint32_t dieId, const std::string& kernelFuncName, uintptr_t kernelFunc,
           py::handle kernelArgs, uint32_t argNum)
        {
            auto a = to_ptr(kernelArgs);
            CcuKernelHandle h = 0;
            CcuResult ret =
                HcommCcuKernelRegister(ins, dieId, kernelFuncName.c_str(), kernelFunc, a.data(), argNum, &h);
            return py::make_tuple(ret, h);
        },
        "insHandle"_a, "dieId"_a, "kernelFuncName"_a, "kernelFunc"_a, "kernelArgs"_a, "argNum"_a);
    m.def("HcommCcuKernelRegisterEnd", &HcommCcuKernelRegisterEnd, "insHandle"_a);
    m.def(
        "HcommCcuGetTaskArgsNum",
        [](CcuKernelHandle kernelHandle)
        {
            uint32_t n = 0;
            CcuResult ret = HcommCcuGetTaskArgsNum(kernelHandle, &n);
            return py::make_tuple(ret, n);
        },
        "kernelHandle"_a);
    m.def("HcommCcuKernelLaunch", &HcommCcuKernelLaunch, "threadHandle"_a, "kernelHandle"_a, "taskArgs"_a, "argNum"_a);

    // ---------- 结构体初始化辅助函数 ----------
    m.def("HcclCommConfigInit", [](HcclCommConfig& c) { HcclCommConfigInit(&c); }, "config"_a);
    m.def("HcclOpDescInit", [](HcclOpDesc& d) { return HcclOpDescInit(&d); }, "opDesc"_a);
    m.def(
        "HcclKernelLaunchCfgInit", [](HcclKernelLaunchCfg& c) { return HcclKernelLaunchCfgInit(&c); },
        "kernelLaunchCfg"_a);
    // 注：以下 Init 函数均就地修改传入的结构体对象（按引用传入）。
    m.def("HcclChannelDescInit", [](HcclChannelDesc& d) { return HcclChannelDescInit(&d, 1); }, "channelDesc"_a);
    m.def("HcclTeamCreateDescInit", [](HcclTeamCreateDesc& d) { return HcclTeamCreateDescInit(&d); }, "desc"_a);
    m.def("HcommChannelDescInit", [](HcommChannelDesc& d) { return HcommChannelDescInit(&d, 1); }, "channelDesc"_a);
    m.def("ThreadConfigInit", [](ThreadConfig& c) { return ThreadConfigInit(&c, 1); }, "config"_a);
    m.def("EndpointDescInit", [](EndpointDesc& e) { return EndpointDescInit(&e, 1); }, "endpoint"_a);
    m.def("CommLinkInit", [](CommLink& l) { return CommLinkInit(&l, 1); }, "commLink"_a);
    m.def("CcuLoopCfgInit", [](CcuLoopCfg& c) { CcuLoopCfgInit(&c); }, "cfg"_a);
    m.def("CcuLoopGroupCfgInit", [](CcuLoopGroupCfg& c) { CcuLoopGroupCfgInit(&c); }, "cfg"_a);
}

// =============================================================================
// 模块入口
// =============================================================================
PYBIND11_MODULE(_C, m)
{
    m.doc() = "pyhccl native extension (ABI bindings for HCCL/HCOMM/CCU)";

    // 运行库加载辅助。
    m.def(
        "load_library", []() { return load_native_libraries(); },
        "Load the HCCL/HCOMM native libraries (dlopen). Returns the number of loaded libraries.");
    m.def("has_native_library", &has_native_library, "Return True if at least one native library has been loaded.");
    m.def(
        "resolve_symbol",
        [](const std::string& name)
        {
            void* p = resolve_symbol(name.c_str());
            return reinterpret_cast<uintptr_t>(p);
        },
        py::arg("name"), "Resolve a native symbol by name; return its address or 0.");

    // 常用常量。
    m.attr("HCCL_ROOT_INFO_BYTES") = HCCL_ROOT_INFO_BYTES;
    m.attr("COMM_NAME_MAX_LENGTH") = COMM_NAME_MAX_LENGTH;
    m.attr("BUFFER_NAME_MAX_LENGTH") = BUFFER_NAME_MAX_LENGTH;
    m.attr("UDI_MAX_LENGTH") = UDI_MAX_LENGTH;
    m.attr("HCCL_COMM_ALGO_MAX_LENGTH") = HCCL_COMM_ALGO_MAX_LENGTH;
    m.attr("HCCL_COMM_DEFAULT_BUFFSIZE") = HCCL_COMM_DEFAULT_BUFFSIZE;
    m.attr("HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE") = HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE;

    bind_enums(m);
    bind_structs(m);
    bind_functions(m);
}

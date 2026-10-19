// -*- coding: utf-8 -*-
// HCCL/HCOMM/CCU 全部 host 侧 C 接口的转发器。
//
// 原理：南向库在运行时才通过 dlopen 加载，这里的每一个“转发器”都在首次调用时
// 通过 dlsym 解析同名符号并缓存函数指针。这样 pyhccl 模块本身不依赖任何编译期
// 链接，也不要求导入时原生库必须存在。
//
// 签名与 /mnt/workspace/cann/{hccl,hcomm}/include 下的公开头文件保持一致：
//   - 指针类型（句柄/缓冲区/stream）按 uintptr_t 传值；
//   - 数组指针与出参指针保持 T* 形式，由 pybind11 层负责编组。

#pragma once

#include <cstring>
#include <stdexcept>
#include <string>

#include "abi.hpp"
#include "loader.hpp"

namespace pyhccl
{

// 生成一个“延迟解析 + 缓存”的转发器函数。
#define PYHCCL_FUNC(ret, name, params, args)                                                                           \
    using name##_fn = ret(*) params;                                                                                   \
    inline ret name params                                                                                             \
    {                                                                                                                  \
        static name##_fn fn = nullptr;                                                                                 \
        if (fn == nullptr)                                                                                             \
        {                                                                                                              \
            load_native_libraries();                                                                                   \
            fn = reinterpret_cast<name##_fn>(resolve_symbol(#name));                                                   \
        }                                                                                                              \
        if (fn == nullptr)                                                                                             \
        {                                                                                                              \
            throw std::runtime_error(std::string(#name) +                                                              \
                                     " is not available: HCCL/HCOMM native library is not loaded");                    \
        }                                                                                                              \
        return fn args;                                                                                                \
    }

// =============================================================================
// 内联的“结构体初始化”函数（原始头文件中是 static inline，不属于动态库导出符号，
// 因此直接在绑定层按原始逻辑重新实现）。
// =============================================================================
// 说明：原始实现使用 securec 的 memset_s，这里用标准库 std::memset 等价替代，
// 语义一致（0xFF 填充表示“无效”，0 填充表示“零初始化”）。

inline void HcclCommConfigInit(HcclCommConfig* config)
{
    if (config == nullptr)
    {
        return;
    }
    std::memset(config, 0, sizeof(*config));
    // 前 24 字节 ABI 头（size/magicWord/version/reserved）。
    struct CommConfigInfo
    {
        uint64_t size;
        uint32_t magicWord;
        uint32_t version;
        uint64_t reserved;
    };
    static_assert(sizeof(CommConfigInfo) == HCCL_COMM_CONFIG_INFO_BYTES, "ABI 头大小错误");
    CommConfigInfo info{sizeof(HcclCommConfig), HCCL_COMM_CONFIG_MAGIC_WORD, HCCL_COMM_CONFIG_VERSION, 0};
    std::memcpy(config->reserved, &info, sizeof(info));
    config->hcclBufferSize = HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET;
    config->hcclDeterministic = HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET;
    config->hcclCommName[0] = '\0';
    config->hcclUdi[0] = '\0';
    config->hcclOpExpansionMode = HCCL_COMM_DEFAULT_OP_EXPANSION_MODE;
    config->hcclRdmaTrafficClass = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET;
    config->hcclRdmaServiceLevel = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET;
    config->hcclWorldRankID = 0;
    config->hcclJobID = 0;
    config->aclGraphZeroCopyEnable = 0;
    config->hcclExecTimeOut = HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET;
    config->hcclAlgo[0] = '\0';
    config->hcclRetryEnable[0] = '\0';
    config->hcclRetryParams[0] = '\0';
    config->hcclBufferName[0] = '\0';
    config->hcclQos = HCCL_COMM_QOS_CONFIG_NOT_SET;
    config->hcclSymWinMaxMemSizePerRank = HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE;
    config->hcclChannelSqDepth = HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET;
}

inline HcclResult HcclOpDescInit(HcclOpDesc* opDesc)
{
    if (opDesc == nullptr)
    {
        return HCCL_E_PTR;
    }
    std::memset(opDesc, 0xFF, sizeof(*opDesc));
    opDesc->header.version = HCCL_OPDESC_VERSION;
    opDesc->header.magicWord = HCCL_OPDESC_MAGIC_WORD;
    opDesc->header.size = sizeof(HcclOpDesc);
    opDesc->header.reserved = 0;
    return HCCL_SUCCESS;
}

inline HcclResult HcclKernelLaunchCfgInit(HcclKernelLaunchCfg* cfg)
{
    if (cfg == nullptr)
    {
        return HCCL_E_PTR;
    }
    std::memset(cfg, 0xFF, sizeof(*cfg));
    cfg->header.version = HCCL_KERNEL_LAUNCH_CFG_VERSION;
    cfg->header.magicWord = HCCL_KERNEL_LAUNCH_CFG_MAGIC_WORD;
    cfg->header.size = sizeof(HcclKernelLaunchCfg);
    cfg->header.reserved = 0;
    return HCCL_SUCCESS;
}

inline HcclResult EndpointDescInit(EndpointDesc* endpoint, uint32_t num)
{
    for (uint32_t i = 0; i < num; ++i)
    {
        if (endpoint == nullptr)
        {
            return HCCL_E_PTR;
        }
        std::memset(endpoint, 0xFF, sizeof(*endpoint));
        endpoint->protocol = COMM_PROTOCOL_RESERVED;
        endpoint->commAddr.type = COMM_ADDR_TYPE_RESERVED;
        endpoint->loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
        ++endpoint;
    }
    return HCCL_SUCCESS;
}

inline HcclResult HcclChannelDescInit(HcclChannelDesc* desc, uint32_t descNum)
{
    for (uint32_t i = 0; i < descNum; ++i)
    {
        if (desc == nullptr)
        {
            return HCCL_E_PTR;
        }
        std::memset(desc, 0xFF, sizeof(*desc));
        desc->header.version = HCCL_CHANNEL_VERSION;
        desc->header.magicWord = HCCL_CHANNEL_MAGIC_WORD;
        desc->header.size = sizeof(HcclChannelDesc);
        desc->header.reserved = 0;
        desc->remoteRank = ~0U;
        desc->channelProtocol = COMM_PROTOCOL_RESERVED;
        desc->notifyNum = 0;
        desc->memHandles = 0;
        desc->memHandleNum = 0;
        if (EndpointDescInit(&desc->localEndpoint, 1) != HCCL_SUCCESS ||
            EndpointDescInit(&desc->remoteEndpoint, 1) != HCCL_SUCCESS)
        {
            return HCCL_E_INTERNAL;
        }
        ++desc;
    }
    return HCCL_SUCCESS;
}

inline HcclResult HcclTeamCreateDescInit(HcclTeamCreateDesc* desc)
{
    if (desc == nullptr)
    {
        return HCCL_E_PTR;
    }
    std::memset(desc, 0xFF, sizeof(*desc));
    desc->header.version = HCCL_TEAM_CREATE_DESC_VERSION;
    desc->header.magicWord = HCCL_TEAM_CREATE_DESC_MAGIC_WORD;
    desc->header.size = sizeof(HcclTeamCreateDesc);
    desc->header.reserved = 0;
    desc->rankIds = 0;
    desc->rankNum = 0;
    desc->selfRankId = 0;
    desc->netLayer = 0;
    desc->protocol = COMM_PROTOCOL_RESERVED;
    desc->requirement.signalCount = 0;
    desc->requirement.counterCount = 0;
    desc->requirement.barrierCount = 0;
    desc->engine = COMM_ENGINE_RESERVED;
    desc->notifyNum = 0;
    desc->channelCnt = 0;
    desc->isSharedQueue = 0;
    desc->sharedQueueTag[0] = '\0';
    for (uint32_t i = 0; i < 8; ++i)
    {
        desc->reserved[i] = 0;
    }
    return HCCL_SUCCESS;
}

inline HcommResult HcommChannelDescInit(HcommChannelDesc* desc, uint32_t descNum)
{
    for (uint32_t i = 0; i < descNum; ++i)
    {
        if (desc == nullptr)
        {
            return HCCL_E_PTR;
        }
        std::memset(desc, 0xFF, sizeof(*desc));
        desc->header.version = HCOMM_CHANNEL_VERSION;
        desc->header.magicWord = HCOMM_CHANNEL_MAGIC_WORD;
        desc->header.size = sizeof(HcommChannelDesc);
        desc->header.reserved = 0;
        desc->notifyNum = 0;
        desc->exchangeAllMems = false;
        desc->memHandles = 0;
        desc->memHandleNum = 0;
        desc->socket = 0;
        desc->role = HCOMM_SOCKET_ROLE_RESERVED;
        desc->port = 0;
        desc->roceAttr.srcPortList = 0;
        desc->channelName = 0;
        if (EndpointDescInit(&desc->remoteEndpoint, 1) != HCCL_SUCCESS)
        {
            return HCCL_E_INTERNAL;
        }
        ++desc;
    }
    return HCCL_SUCCESS;
}

inline HcommResult ThreadConfigInit(ThreadConfig* config, uint32_t num)
{
    for (uint32_t i = 0; i < num; ++i)
    {
        if (config == nullptr)
        {
            return HCCL_E_PTR;
        }
        std::memset(config, 0, sizeof(*config));
        config->header.version = HCOMM_THREAD_CONFIG_VERSION;
        config->header.magicWord = HCOMM_THREAD_CONFIG_MAGIC_WORD;
        config->header.size = sizeof(ThreadConfig);
        config->header.reserved = 0;
        config->notifyNumPerThread = 0;
        ++config;
    }
    return HCCL_SUCCESS;
}

inline HcclResult CommLinkInit(CommLink* link, uint32_t linkNum)
{
    for (uint32_t i = 0; i < linkNum; ++i)
    {
        if (link == nullptr)
        {
            return HCCL_E_PTR;
        }
        std::memset(link, 0xFF, sizeof(*link));
        link->header.version = COMM_LINK_VERSION;
        link->header.magicWord = COMM_LINK_MAGIC_WORD;
        link->header.size = sizeof(CommLink);
        link->header.reserved = 0;
        if (EndpointDescInit(&link->srcEndpointDesc, 1) != HCCL_SUCCESS ||
            EndpointDescInit(&link->dstEndpointDesc, 1) != HCCL_SUCCESS)
        {
            return HCCL_E_INTERNAL;
        }
        link->linkAttr.linkProtocol = COMM_PROTOCOL_RESERVED;
        link->linkAttr.hop = 1;
        ++link;
    }
    return HCCL_SUCCESS;
}

inline void CcuLoopCfgInit(CcuLoopCfg* cfg)
{
    cfg->header.magic = CCU_CFG_MAGIC_WORD;
    cfg->header.version = CCU_LOOP_CFG_VERSION;
    cfg->header.size = sizeof(CcuLoopCfg);
    cfg->header.reserved = 0;
}

inline void CcuLoopGroupCfgInit(CcuLoopGroupCfg* cfg)
{
    cfg->header.magic = CCU_CFG_MAGIC_WORD;
    cfg->header.version = CCU_LOOPGROUP_CFG_VERSION;
    cfg->header.size = sizeof(CcuLoopGroupCfg);
    cfg->header.reserved = 0;
}

// =============================================================================
// 集合通信与点对点算子（hccl.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclAllReduce,
            (uintptr_t sendBuf, uintptr_t recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
             HcclComm comm, aclrtStream stream),
            (sendBuf, recvBuf, count, dataType, op, comm, stream))

PYHCCL_FUNC(HcclResult, HcclBroadcast,
            (uintptr_t buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream),
            (buf, count, dataType, root, comm, stream))

PYHCCL_FUNC(HcclResult, HcclReduceScatter,
            (uintptr_t sendBuf, uintptr_t recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
             HcclComm comm, aclrtStream stream),
            (sendBuf, recvBuf, recvCount, dataType, op, comm, stream))

PYHCCL_FUNC(HcclResult, HcclReduceScatterV,
            (uintptr_t sendBuf, const uint64_t* sendCounts, const uint64_t* sendDispls, uintptr_t recvBuf,
             uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream),
            (sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream))

PYHCCL_FUNC(HcclResult, HcclScatter,
            (uintptr_t sendBuf, uintptr_t recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
             HcclComm comm, aclrtStream stream),
            (sendBuf, recvBuf, recvCount, dataType, root, comm, stream))

PYHCCL_FUNC(HcclResult, HcclAllGather,
            (uintptr_t sendBuf, uintptr_t recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
             aclrtStream stream),
            (sendBuf, recvBuf, sendCount, dataType, comm, stream))

PYHCCL_FUNC(HcclResult, HcclAllGatherV,
            (uintptr_t sendBuf, uint64_t sendCount, uintptr_t recvBuf, const uint64_t* recvCounts,
             const uint64_t* recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream),
            (sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream))

PYHCCL_FUNC(HcclResult, HcclSend,
            (uintptr_t sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm,
             aclrtStream stream),
            (sendBuf, count, dataType, destRank, comm, stream))

PYHCCL_FUNC(HcclResult, HcclRecv,
            (uintptr_t recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm,
             aclrtStream stream),
            (recvBuf, count, dataType, srcRank, comm, stream))

PYHCCL_FUNC(HcclResult, HcclAlltoAllVC,
            (uintptr_t sendBuf, const uint64_t* sendCountMatrix, HcclDataType sendType, uintptr_t recvBuf,
             HcclDataType recvType, HcclComm comm, aclrtStream stream),
            (sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream))

PYHCCL_FUNC(HcclResult, HcclAlltoAllV,
            (uintptr_t sendBuf, const uint64_t* sendCounts, const uint64_t* sdispls, HcclDataType sendType,
             uintptr_t recvBuf, const uint64_t* recvCounts, const uint64_t* rdispls, HcclDataType recvType,
             HcclComm comm, aclrtStream stream),
            (sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream))

PYHCCL_FUNC(HcclResult, HcclAlltoAll,
            (uintptr_t sendBuf, uint64_t sendCount, HcclDataType sendType, uintptr_t recvBuf, uint64_t recvCount,
             HcclDataType recvType, HcclComm comm, aclrtStream stream),
            (sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream))

PYHCCL_FUNC(HcclResult, HcclReduce,
            (uintptr_t sendBuf, uintptr_t recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
             uint32_t root, HcclComm comm, aclrtStream stream),
            (sendBuf, recvBuf, count, dataType, op, root, comm, stream))

PYHCCL_FUNC(HcclResult, HcclBatchSendRecv,
            (HcclSendRecvItem * sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream),
            (sendRecvInfo, itemNum, comm, stream))

// =============================================================================
// mc2 算子参数（hccl_mc2.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclKfcAllocOpArgs, (uintptr_t* opArgs), (opArgs))
PYHCCL_FUNC(HcclResult, HcclKfcFreeOpArgs, (uintptr_t opArgs), (opArgs))
PYHCCL_FUNC(HcclResult, HcclKfcOpArgsSetSrcDataType, (uintptr_t opArgs, uint8_t srcDataType), (opArgs, srcDataType))
PYHCCL_FUNC(HcclResult, HcclKfcOpArgsSetDstDataType, (uintptr_t opArgs, uint8_t dstDataType), (opArgs, dstDataType))
PYHCCL_FUNC(HcclResult, HcclKfcOpArgsSetReduceType, (uintptr_t opArgs, uint32_t reduceType), (opArgs, reduceType))
PYHCCL_FUNC(HcclResult, HcclKfcOpArgsSetCount, (uintptr_t opArgs, uint64_t count), (opArgs, count))
PYHCCL_FUNC(HcclResult, HcclKfcOpArgsSetAlgConfig, (uintptr_t opArgs, const char* algConfig), (opArgs, algConfig))
PYHCCL_FUNC(HcclResult, HcclKfcOpArgsSetCommEngine, (uintptr_t opArgs, uint8_t commEngine), (opArgs, commEngine))
PYHCCL_FUNC(HcclResult, HcclCreateOpResCtx, (HcclComm comm, uint8_t opType, uintptr_t opArgs, uintptr_t* opResCtx),
            (comm, opType, opArgs, opResCtx))

// =============================================================================
// 通信域初始化与管理（hccl_comm.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclCommInitClusterInfo, (const char* clusterInfo, uint32_t rank, HcclComm* comm),
            (clusterInfo, rank, comm))
PYHCCL_FUNC(HcclResult, HcclCommInitClusterInfoConfig,
            (const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm),
            (clusterInfo, rank, config, comm))
PYHCCL_FUNC(HcclResult, HcclCreateSubCommConfig,
            (HcclComm * comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId,
             HcclCommConfig* config, HcclComm* subComm),
            (comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm))
PYHCCL_FUNC(HcclResult, HcclGetRootInfo, (HcclRootInfo * rootInfo), (rootInfo))
PYHCCL_FUNC(HcclResult, HcclCommInitRootInfo,
            (uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm),
            (nRanks, rootInfo, rank, comm))
PYHCCL_FUNC(HcclResult, HcclCommInitRootInfoConfig,
            (uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config,
             HcclComm* comm),
            (nRanks, rootInfo, rank, config, comm))
PYHCCL_FUNC(HcclResult, HcclGetRootInfoScalable, (HcclRootInfo * rootInfo), (rootInfo))
PYHCCL_FUNC(HcclResult, HcclCommInitRootInfoScalable,
            (uint32_t nRanks, uint32_t nRoot, const HcclRootInfo* rootInfoList, uint32_t rank, uint32_t nExtRoot,
             const HcclCommConfig* config, HcclComm* comm),
            (nRanks, nRoot, rootInfoList, rank, nExtRoot, config, comm))
PYHCCL_FUNC(HcclResult, HcclSetConfig, (HcclConfig config, int32_t configValue), (config, configValue))
PYHCCL_FUNC(HcclResult, HcclGetConfig, (HcclConfig config, HcclConfigValue* configValue), (config, configValue))
PYHCCL_FUNC(HcclResult, HcclGetCommName, (HcclComm comm, char* commName), (comm, commName))
PYHCCL_FUNC(HcclResult, HcclCommGetHandleWithName, (const char* commName, HcclComm* comm), (commName, comm))
PYHCCL_FUNC(HcclResult, HcclGetRankSize, (HcclComm comm, uint32_t* rankSize), (comm, rankSize))
PYHCCL_FUNC(HcclResult, HcclGetRankId, (HcclComm comm, uint32_t* rank), (comm, rank))
PYHCCL_FUNC(HcclResult, HcclBarrier, (HcclComm comm, aclrtStream stream), (comm, stream))
PYHCCL_FUNC(HcclResult, HcclCommDestroy, (HcclComm comm), (comm))
PYHCCL_FUNC(HcclResult, HcclCommInitAll, (uint32_t ndev, int32_t* devices, HcclComm* comms), (ndev, devices, comms))
PYHCCL_FUNC(HcclResult, HcclGetCommAsyncError, (HcclComm comm, HcclResult* asyncError), (comm, asyncError))
PYHCCL_FUNC(const char*, HcclGetErrorString, (HcclResult code), (code))
PYHCCL_FUNC(uint32_t, HcclGetCommConfigCapability, (), ())
PYHCCL_FUNC(HcclResult, HcclCommSuspend, (HcclComm comm), (comm))
PYHCCL_FUNC(HcclResult, HcclCommRegCommStateCallback, (const char* regName, uintptr_t cb, uintptr_t args),
            (regName, cb, args))
PYHCCL_FUNC(HcclResult, HcclCommResume, (HcclComm comm), (comm))
PYHCCL_FUNC(HcclResult, HcclCommGetStatus, (const char* commId, HcclCommStatus* status), (commId, status))
PYHCCL_FUNC(HcclResult, HcclCommSetMemoryRange,
            (HcclComm comm, uintptr_t baseVirPtr, size_t size, size_t alignment, uint64_t flags),
            (comm, baseVirPtr, size, alignment, flags))
PYHCCL_FUNC(HcclResult, HcclCommUnsetMemoryRange, (HcclComm comm, uintptr_t baseVirPtr), (comm, baseVirPtr))
PYHCCL_FUNC(HcclResult, HcclCommActivateCommMemory,
            (HcclComm comm, uintptr_t virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags),
            (comm, virPtr, size, offset, handle, flags))
PYHCCL_FUNC(HcclResult, HcclCommDeactivateCommMemory, (HcclComm comm, uintptr_t virPtr), (comm, virPtr))
PYHCCL_FUNC(HcclResult, HcclCommWorkingDevNicSet, (HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks),
            (comm, ranks, useBackup, nRanks))
PYHCCL_FUNC(HcclResult, HcclGroupStart, (), ())
PYHCCL_FUNC(HcclResult, HcclGroupEnd, (), ())
PYHCCL_FUNC(HcclResult, HcclGroupStatusGet, (bool* isGroupEnabled), (isGroupEnabled))
PYHCCL_FUNC(HcclResult, HcclGroupEnqueueOp,
            (HcclComm comm, const HcclOpDesc* opDesc, aclrtStream stream, uintptr_t func), (comm, opDesc, stream, func))
PYHCCL_FUNC(HcclResult, HcclCommSymWinRegister,
            (HcclComm comm, uintptr_t addr, uint64_t size, HcclCommSymWindow* winHandle, uint32_t flag),
            (comm, addr, size, winHandle, flag))
PYHCCL_FUNC(HcclResult, HcclCommSymWinDeregister, (HcclCommSymWindow winHandle), (winHandle))
PYHCCL_FUNC(HcclResult, HcclCommSymWinGet,
            (HcclComm comm, uintptr_t ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset),
            (comm, ptr, size, winHandle, offset))
PYHCCL_FUNC(HcclResult, HcclConfigGetInfo, (HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, uintptr_t info),
            (comm, cfgType, infoLen, info))

// =============================================================================
// 通信资源与上下文（hccl_res.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclGetHcclBuffer, (HcclComm comm, uintptr_t* buffer, uint64_t* size), (comm, buffer, size))
PYHCCL_FUNC(HcclResult, HcclGetHcclBufferCleared, (HcclComm comm, uintptr_t* buffer, uint64_t* size),
            (comm, buffer, size))
PYHCCL_FUNC(HcclResult, HcclThreadAcquire,
            (HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads),
            (comm, engine, threadNum, notifyNumPerThread, threads))
PYHCCL_FUNC(HcclResult, HcclThreadAcquireWithConfig,
            (HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
             ThreadHandle* threads),
            (comm, engine, threadNum, type, config, threads))
PYHCCL_FUNC(HcclResult, HcclThreadAcquireWithStream,
            (HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread),
            (comm, engine, stream, notifyNum, thread))
PYHCCL_FUNC(HcclResult, HcclDedicatedThreadAcquire,
            (HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread),
            (comm, useType, notifyNumPerThread, thread))
PYHCCL_FUNC(HcclResult, HcclEngineCtxCreate,
            (HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, uintptr_t* ctx),
            (comm, ctxTag, engine, size, ctx))
PYHCCL_FUNC(HcclResult, HcclEngineCtxGet,
            (HcclComm comm, const char* ctxTag, CommEngine engine, uintptr_t* ctx, uint64_t* size),
            (comm, ctxTag, engine, ctx, size))
PYHCCL_FUNC(HcclResult, HcclEngineCtxCopy,
            (HcclComm comm, CommEngine engine, const char* ctxTag, uintptr_t srcCtx, uint64_t size,
             uint64_t dstCtxOffset),
            (comm, engine, ctxTag, srcCtx, size, dstCtxOffset))
PYHCCL_FUNC(HcclResult, HcclEngineCtxDestroy, (HcclComm comm, const char* ctxTag, CommEngine engine),
            (comm, ctxTag, engine))
PYHCCL_FUNC(HcclResult, HcclCommMemReg,
            (HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle),
            (comm, memTag, mem, memHandle))
PYHCCL_FUNC(HcclResult, HcclThreadResGetInfo,
            (HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, uintptr_t* info),
            (comm, thread, resType, infoLen, info))

// =============================================================================
// 通信通道管理（hccl_channel.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclChannelAcquire,
            (HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
             ChannelHandle* channels),
            (comm, engine, channelDescs, channelNum, channels))
PYHCCL_FUNC(HcclResult, HcclChannelConfigCreate, (HcclChannelConfig * config), (config))
PYHCCL_FUNC(HcclResult, HcclChannelConfigDestroy, (HcclChannelConfig config), (config))
PYHCCL_FUNC(HcclResult, HcclChannelConfigSetInt, (HcclChannelConfig config, HcclChannelConfigType type, uint32_t value),
            (config, type, value))
PYHCCL_FUNC(HcclResult, HcclChannelConfigSetStr,
            (HcclChannelConfig config, HcclChannelConfigType type, const char* value), (config, type, value))
PYHCCL_FUNC(HcclResult, HcclChannelAcquireWithConfig,
            (HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
             HcclChannelConfig config, ChannelHandle* channels),
            (comm, engine, channelDescs, channelNum, config, channels))
PYHCCL_FUNC(HcclResult, HcclChannelQuery,
            (HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
             ChannelHandle* channels),
            (comm, engine, channelDescs, channelNum, channels))
PYHCCL_FUNC(HcclResult, HcclChannelDestroy, (HcclComm comm, const ChannelHandle* channels, uint32_t channelNum),
            (comm, channels, channelNum))
PYHCCL_FUNC(HcclResult, HcclChannelGetHcclBuffer,
            (HcclComm comm, ChannelHandle channel, uintptr_t* buffer, uint64_t* size), (comm, channel, buffer, size))
PYHCCL_FUNC(HcclResult, HcclChannelGetRemoteMems,
            (HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags),
            (comm, channel, memNum, remoteMems, memTags))

// =============================================================================
// Kernel 启动（hccl_launch.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclAicpuKernelLaunch,
            (HcclComm comm, const HcclOpDesc* opInfo, const HcclKernelFuncInfo* funcInfo,
             ThreadHandle aicpuThreadHandle, aclrtStream userStream, const HcclKernelLaunchCfg* kernelLaunchCfg),
            (comm, opInfo, funcInfo, aicpuThreadHandle, userStream, kernelLaunchCfg))

// =============================================================================
// 通信拓扑查询（hccl_rank_graph.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclRankGraphGetLayers, (HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum),
            (comm, netLayers, netLayerNum))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetRanksByLayer,
            (HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum), (comm, netLayer, ranks, rankNum))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetRankSizeByLayer, (HcclComm comm, uint32_t netLayer, uint32_t* rankNum),
            (comm, netLayer, rankNum))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetTopoTypeByLayer, (HcclComm comm, uint32_t netLayer, CommTopo* topoType),
            (comm, netLayer, topoType))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetInstSizeListByLayer,
            (HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize),
            (comm, netLayer, instSizeList, listSize))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetLinks,
            (HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum),
            (comm, netLayer, srcRank, dstRank, links, linkNum))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetTopoInstsByLayer,
            (HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum),
            (comm, netLayer, topoInsts, topoInstNum))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetTopoType,
            (HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType),
            (comm, netLayer, topoInstId, topoType))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetRanksByTopoInst,
            (HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum),
            (comm, netLayer, topoInstId, ranks, rankNum))
PYHCCL_FUNC(HcclResult, HcclGetHeterogMode, (HcclComm comm, HcclHeterogMode* mode), (comm, mode))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetEndpointNum,
            (HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num), (comm, layer, topoInstId, num))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetEndpointDesc,
            (HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc),
            (comm, layer, topoInstId, descNum, endpointDesc))
PYHCCL_FUNC(HcclResult, HcclRankGraphGetEndpointInfo,
            (HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc, EndpointAttr endpointAttr,
             uint32_t infoLen, uintptr_t info),
            (comm, rankId, endpointDesc, endpointAttr, infoLen, info))

// =============================================================================
// 通信团队（hccl_team.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclTeamCreate, (HcclComm comm, const HcclTeamCreateDesc* desc, HcommTeamHandle* team),
            (comm, desc, team))
PYHCCL_FUNC(HcclResult, HcclTeamDestroy, (HcommTeamHandle team), (team))
PYHCCL_FUNC(HcclResult, HcclTeamGetLsaTeam, (HcclComm comm, HcommTeamHandle* lsaTeam), (comm, lsaTeam))
PYHCCL_FUNC(HcclResult, HcclTeamMemberToRank,
            (HcclComm comm, HcommTeamHandle team, uint32_t memberId, uint32_t* rankId), (comm, team, memberId, rankId))
PYHCCL_FUNC(HcclResult, HcclTeamRankToMember,
            (HcclComm comm, HcommTeamHandle team, uint32_t rankId, uint32_t* memberId), (comm, team, rankId, memberId))

// =============================================================================
// CCU 实例绑定（hccl_ccu_res.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclCommAssignCcuIns, (HcclComm comm, CcuInsHandle insHandle), (comm, insHandle))
PYHCCL_FUNC(HcclResult, HcclCommQueryCcuIns, (HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum),
            (comm, insHandles, insNum))
PYHCCL_FUNC(HcclResult, HcclCommQueryAssignedCcuIns, (HcclComm comm, CcuInsHandle* insHandles, uint32_t* insNum),
            (comm, insHandles, insNum))

// =============================================================================
// 对称内存窗口（hccl_sym_win.h）
// =============================================================================
PYHCCL_FUNC(HcclResult, HcclSymWinGetPeerPointer,
            (HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, uintptr_t* ptr),
            (winHandle, offset, peerRank, ptr))
PYHCCL_FUNC(HcclResult, HcclSymWinGetRemoteAddr,
            (HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, uintptr_t* ptr),
            (winHandle, offset, peerRank, ptr))

// =============================================================================
// HCOMM 基础资源（hcomm_res.h）
// =============================================================================
PYHCCL_FUNC(HcommResult, HcommEndpointCreate, (const EndpointDesc* endpoint, EndpointHandle* endpointHandle),
            (endpoint, endpointHandle))
PYHCCL_FUNC(HcommResult, HcommEndpointDestroy, (EndpointHandle endpointHandle), (endpointHandle))
PYHCCL_FUNC(HcommResult, HcommEndpointGetDescNum, (int32_t deviceLogicId, uint32_t* descNum), (deviceLogicId, descNum))
PYHCCL_FUNC(HcommResult, HcommEndpointGetDescs, (int32_t deviceLogicId, uint32_t* descNum, EndpointDesc* endpointDescs),
            (deviceLogicId, descNum, endpointDescs))
PYHCCL_FUNC(HcommResult, HcommMemReg,
            (EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle),
            (endpointHandle, memTag, mem, memHandle))
PYHCCL_FUNC(HcommResult, HcommMemUnreg, (EndpointHandle endpointHandle, HcommMemHandle memHandle),
            (endpointHandle, memHandle))
PYHCCL_FUNC(HcommResult, HcommMemExport,
            (EndpointHandle endpointHandle, HcommMemHandle memHandle, uintptr_t* memDesc, uint32_t* memDescLen),
            (endpointHandle, memHandle, memDesc, memDescLen))
PYHCCL_FUNC(HcommResult, HcommMemImport,
            (EndpointHandle endpointHandle, uintptr_t memDesc, uint32_t descLen, CommMem* outMem),
            (endpointHandle, memDesc, descLen, outMem))
PYHCCL_FUNC(HcommResult, HcommMemUnimport, (EndpointHandle endpointHandle, uintptr_t memDesc, uint32_t descLen),
            (endpointHandle, memDesc, descLen))
PYHCCL_FUNC(HcommResult, HcommThreadAlloc,
            (CommEngine engine, uint32_t threadNum, const uint32_t* notifyNumPerThread, ThreadHandle* threads),
            (engine, threadNum, notifyNumPerThread, threads))
PYHCCL_FUNC(HcommResult, HcommThreadFree, (const ThreadHandle* threads, uint32_t threadNum), (threads, threadNum))
PYHCCL_FUNC(HcommResult, HcommEndpointGetListenPort, (EndpointHandle endpointHandle, uint32_t* port),
            (endpointHandle, port))
PYHCCL_FUNC(HcommResult, HcommThreadResGetInfo,
            (ThreadHandle thread, ThreadResType resType, uint32_t infoLen, uintptr_t* info),
            (thread, resType, infoLen, info))
PYHCCL_FUNC(HcommResult, HcommEndpointCheckFeature,
            (HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc, bool* value),
            (featureType, endpointDesc, value))
PYHCCL_FUNC(HcommResult, HcommMemAlloc, (uintptr_t* ptr, size_t size), (ptr, size))
PYHCCL_FUNC(HcommResult, HcommMemFree, (uintptr_t ptr), (ptr))

// =============================================================================
// HCOMM 通道（hcomm_channel.h）
// =============================================================================
PYHCCL_FUNC(HcommResult, HcommChannelCreate,
            (EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
             ChannelHandle* channels),
            (endpointHandle, engine, channelDescs, channelNum, channels))
PYHCCL_FUNC(HcommResult, HcommChannelConfigCreate, (HcommChannelConfig * config), (config))
PYHCCL_FUNC(HcommResult, HcommChannelConfigDestroy, (HcommChannelConfig config), (config))
PYHCCL_FUNC(HcommResult, HcommChannelConfigSetInt,
            (HcommChannelConfig config, HcommChannelConfigType type, uint32_t value), (config, type, value))
PYHCCL_FUNC(HcommResult, HcommChannelCreateWithConfig,
            (EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
             HcommChannelConfig config, ChannelHandle* channels),
            (endpointHandle, engine, channelDescs, channelNum, config, channels))
PYHCCL_FUNC(HcommResult, HcommChannelGetStatus,
            (const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList),
            (channelList, listNum, statusList))
PYHCCL_FUNC(HcommResult, HcommChannelDestroy, (const ChannelHandle* channels, uint32_t channelNum),
            (channels, channelNum))

// =============================================================================
// HCOMM 数据面原语（hcomm_primitives.h）
// =============================================================================
PYHCCL_FUNC(int32_t, HcommLocalCopyOnThread, (ThreadHandle thread, uintptr_t dst, const uintptr_t src, uint64_t len),
            (thread, dst, src, len))
PYHCCL_FUNC(int32_t, HcommLocalReduceOnThread,
            (ThreadHandle thread, uintptr_t dst, const uintptr_t src, uint64_t count, HcommDataType dataType,
             HcommReduceOp reduceOp),
            (thread, dst, src, count, dataType, reduceOp))
PYHCCL_FUNC(int32_t, HcommThreadNotifyRecordOnThread,
            (ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx), (thread, dstThread, dstNotifyIdx))
PYHCCL_FUNC(int32_t, HcommThreadNotifyWaitOnThread, (ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut),
            (thread, notifyIdx, timeOut))
PYHCCL_FUNC(int32_t, HcommAclrtNotifyRecordOnThread, (ThreadHandle thread, uint64_t dstNotifyId), (thread, dstNotifyId))
PYHCCL_FUNC(int32_t, HcommAclrtNotifyWaitOnThread, (ThreadHandle thread, uint64_t notifyId, uint32_t timeOut),
            (thread, notifyId, timeOut))
PYHCCL_FUNC(int32_t, HcommWriteOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len),
            (thread, channel, dst, src, len))
PYHCCL_FUNC(int32_t, HcommWriteReduceOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t count,
             HcommDataType dataType, HcommReduceOp reduceOp),
            (thread, channel, dst, src, count, dataType, reduceOp))
PYHCCL_FUNC(int32_t, HcommWriteWithNotifyOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len,
             uint32_t remoteNotifyIdx),
            (thread, channel, dst, src, len, remoteNotifyIdx))
PYHCCL_FUNC(int32_t, HcommWriteReduceWithNotifyOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t count,
             HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx),
            (thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx))
PYHCCL_FUNC(int32_t, HcommReadOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len),
            (thread, channel, dst, src, len))
PYHCCL_FUNC(int32_t, HcommReadReduceOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t count,
             HcommDataType dataType, HcommReduceOp reduceOp),
            (thread, channel, dst, src, count, dataType, reduceOp))
PYHCCL_FUNC(int32_t, HcommWriteNbiOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len),
            (thread, channel, dst, src, len))
PYHCCL_FUNC(int32_t, HcommWriteNbi, (ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len),
            (channel, dst, src, len))
PYHCCL_FUNC(int32_t, HcommWriteWithNotifyNbiOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len,
             uint32_t remoteNotifyIdx),
            (thread, channel, dst, src, len, remoteNotifyIdx))
PYHCCL_FUNC(int32_t, HcommWriteWithNotifyNbi,
            (ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len, uint32_t remoteNotifyIdx),
            (channel, dst, src, len, remoteNotifyIdx))
PYHCCL_FUNC(int32_t, HcommReadNbiOnThread,
            (ThreadHandle thread, ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len),
            (thread, channel, dst, src, len))
PYHCCL_FUNC(int32_t, HcommReadNbi, (ChannelHandle channel, uintptr_t dst, const uintptr_t src, uint64_t len),
            (channel, dst, src, len))
PYHCCL_FUNC(int32_t, HcommBatchTransferOnThread,
            (ThreadHandle thread, ChannelHandle channel, const HcommBatchTransferDesc* transferDescs,
             uint32_t transferDescNum),
            (thread, channel, transferDescs, transferDescNum))
PYHCCL_FUNC(int32_t, HcommChannelNotifyRecordOnThread,
            (ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx), (thread, channel, remoteNotifyIdx))
PYHCCL_FUNC(int32_t, HcommChannelNotifyRecord, (ChannelHandle channel, uint32_t remoteNotifyIdx),
            (channel, remoteNotifyIdx))
PYHCCL_FUNC(int32_t, HcommChannelNotifyWaitOnThread,
            (ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut),
            (thread, channel, localNotifyIdx, timeOut))
PYHCCL_FUNC(int32_t, HcommChannelNotifyWait, (ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut),
            (channel, localNotifyIdx, timeOut))
PYHCCL_FUNC(HcommResult, HcommAicpuTsTaskCacheLookup, (const char* tag, bool* isHit), (tag, isHit))
PYHCCL_FUNC(HcommResult, HcommAicpuTsTaskCacheStart,
            (const char* tag, const uintptr_t* addrs, uint64_t* sizes, uint64_t count), (tag, addrs, sizes, count))
PYHCCL_FUNC(HcommResult, HcommAicpuTsTaskCacheEnd, (const char* tag), (tag))
PYHCCL_FUNC(HcommResult, HcommAicpuTsTaskCacheExecute,
            (const char* tag, const uintptr_t* addrs, uint64_t* sizes, uint64_t count), (tag, addrs, sizes, count))
PYHCCL_FUNC(HcommResult, HcommAicpuTsTaskCacheClear, (const char* tag), (tag))
PYHCCL_FUNC(int32_t, HcommBatchModeStart, (const char* batchTag), (batchTag))
PYHCCL_FUNC(int32_t, HcommBatchModeEnd, (const char* batchTag), (batchTag))
PYHCCL_FUNC(int32_t, HcommAcquireComm, (const char* commId), (commId))
PYHCCL_FUNC(int32_t, HcommReleaseComm, (const char* commId), (commId))
PYHCCL_FUNC(int32_t, HcommFenceOnThread, (ThreadHandle thread), (thread))
PYHCCL_FUNC(int32_t, HcommChannelFenceOnThread, (ThreadHandle thread, ChannelHandle channel), (thread, channel))
PYHCCL_FUNC(int32_t, HcommChannelFence, (ChannelHandle channel), (channel))
PYHCCL_FUNC(int32_t, HcommSetNotifyWaitTimeOut, (float timeOut), (timeOut))
PYHCCL_FUNC(int32_t, HcommThreadResAcquireTimeOut, (float timeOut), (timeOut))
PYHCCL_FUNC(int32_t, HcommChannelNotifyWaitOnThreadWithDefaultTimeout,
            (ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx), (thread, channel, localNotifyIdx))
PYHCCL_FUNC(int32_t, HcommThreadNotifyWaitOnThreadWithDefaultTimeout, (ThreadHandle thread, uint32_t notifyIdx),
            (thread, notifyIdx))
PYHCCL_FUNC(int32_t, HcommChannelDrainOnThread, (ThreadHandle thread, ChannelHandle channel), (thread, channel))

// =============================================================================
// CCU 资源（ccu_res.h）
// =============================================================================
PYHCCL_FUNC(CcuResult, HcommCcuVariableAlloc,
            (CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuVariableHandle* varHandle),
            (insHandle, dieId, num, varHandle))
PYHCCL_FUNC(CcuResult, HcommCcuEventAlloc,
            (CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuEventHandle* eventHandle),
            (insHandle, dieId, num, eventHandle))
PYHCCL_FUNC(CcuResult, HcommCcuVariableGetAddr, (CcuVariableHandle varHandle, uint32_t index, uint64_t* va),
            (varHandle, index, va))
PYHCCL_FUNC(CcuResult, HcommCcuEventGetAddr, (CcuEventHandle eventHandle, uint32_t index, uint64_t* va),
            (eventHandle, index, va))
PYHCCL_FUNC(CcuResult, HcommCcuInsResDescCreate, (uint32_t dieId, HcommCcuResDescHandle* resDesc), (dieId, resDesc))
PYHCCL_FUNC(CcuResult, HcommCcuInsResDescDestroy, (HcommCcuResDescHandle resDesc), (resDesc))
PYHCCL_FUNC(CcuResult, HcommCcuInsResDescQueryDieId, (HcommCcuResDescHandle resDesc, uint32_t* dieId), (resDesc, dieId))
PYHCCL_FUNC(CcuResult, HcommCcuInsResDescSetNum,
            (HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t resNum), (resDesc, resType, resNum))
PYHCCL_FUNC(CcuResult, HcommCcuInsResDescQueryNum,
            (HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t* resNum), (resDesc, resType, resNum))
PYHCCL_FUNC(CcuResult, HcommCcuInsCreate,
            (const HcommCcuResDescHandle* resDescs, uint32_t resDescNum, CcuInsHandle* ccuInsHandle),
            (resDescs, resDescNum, ccuInsHandle))
PYHCCL_FUNC(CcuResult, HcommCcuInsCreateDefault, (const uint32_t* dieIds, uint32_t dieNum, CcuInsHandle* ccuInsHandle),
            (dieIds, dieNum, ccuInsHandle))
PYHCCL_FUNC(CcuResult, HcommCcuInsDestroy, (CcuInsHandle ccuInsHandle), (ccuInsHandle))
PYHCCL_FUNC(CcuResult, HcommCcuInsQueryResDesc, (CcuInsHandle ccuInsHandle, HcommCcuResDescHandle resDesc),
            (ccuInsHandle, resDesc))
PYHCCL_FUNC(CcuResult, HcommCcuQueryRemainResDesc, (HcommCcuResDescHandle resDesc), (resDesc))
PYHCCL_FUNC(CcuResult, HcommCcuKernelQueryResReq,
            (uintptr_t kernelFunc, const uintptr_t* kernelArgs, uint32_t argNum, HcommCcuResDescHandle resDesc),
            (kernelFunc, kernelArgs, argNum, resDesc))
PYHCCL_FUNC(CcuResult, HcommCcuGetMemToken, (uint64_t srcVa, uint64_t size, uint64_t* tokenInfo),
            (srcVa, size, tokenInfo))
PYHCCL_FUNC(CcuResult, HcommCcuCascCntAlloc, (CcuInsHandle ccuInsHandle, uint8_t dieId, HcommCcuCascCntHandle* handle),
            (ccuInsHandle, dieId, handle))
PYHCCL_FUNC(CcuResult, HcommCcuCascCntGetMem, (HcommCcuCascCntHandle handle, CommMem* commMem), (handle, commMem))
PYHCCL_FUNC(CcuResult, HcommCcuGetRmtMemToken, (ChannelHandle channelHandle, uint64_t srcVa, uint64_t* tokenInfo),
            (channelHandle, srcVa, tokenInfo))

// =============================================================================
// CCU Kernel 注册与启动（ccu_launch.h）
// =============================================================================
PYHCCL_FUNC(CcuResult, HcommCcuKernelRegisterStart, (CcuInsHandle insHandle), (insHandle))
PYHCCL_FUNC(CcuResult, HcommCcuKernelRegister,
            (CcuInsHandle insHandle, uint32_t dieId, const char* kernelFuncName, uintptr_t kernelFunc,
             const uintptr_t* kernelArgs, uint32_t argNum, CcuKernelHandle* kernelHandle),
            (insHandle, dieId, kernelFuncName, kernelFunc, kernelArgs, argNum, kernelHandle))
PYHCCL_FUNC(CcuResult, HcommCcuKernelRegisterEnd, (CcuInsHandle insHandle), (insHandle))
PYHCCL_FUNC(CcuResult, HcommCcuGetTaskArgsNum, (CcuKernelHandle kernelHandle, uint32_t* taskArgsNum),
            (kernelHandle, taskArgsNum))
PYHCCL_FUNC(CcuResult, HcommCcuKernelLaunch,
            (ThreadHandle threadHandle, CcuKernelHandle kernelHandle, uintptr_t taskArgs, uint32_t argNum),
            (threadHandle, kernelHandle, taskArgs, argNum))

} // namespace pyhccl

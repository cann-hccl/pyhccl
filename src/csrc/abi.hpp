// -*- coding: utf-8 -*-
// pyhccl ABI 镜像头文件
//
// 本文件手工镜像 /mnt/workspace/cann/{hccl,hcomm}/include 下公开头文件中的类型与
// 结构体布局（枚举取值、字段顺序、sizeof 均保持一致）。之所以不直接 include 原始
// 头文件，是因为原始头文件依赖 acl/acl.h、securec.h、ccu_primitives_impl.h 等
// CANN 内部头文件，在无 CANN SDK 的 CI 环境里无法编译。
//
// 约定：
//   1. 所有南向的指针类型（含句柄、缓冲区、stream 等）统一用 uintptr_t 表示，
//      在 x86_64 / aarch64 上它与 void* 按值传递的 ABI 完全一致；Python 侧统一
//      用 int 传入传出。
//   2. 枚举统一用 `enum : int`，保证与 C 侧 int 枚举的 ABI 一致。
//   3. 结构体字段顺序、类型、对齐必须与原始头文件逐字节一致，否则会破坏 ABI。

#pragma once

#include <cstdint>
#include <cstddef>

namespace pyhccl
{

// =============================================================================
// 结果码与基础类型
// =============================================================================

using HcclResult = int32_t;  // typedef enum HcclResult
using HcommResult = int32_t; // typedef int32_t HcommResult
using CcuResult = int32_t;   // typedef enum CcuResult

// -----------------------------------------------------------------------------
// 句柄类型（原始均为指针或 64 位整数，此处统一镜像为 64 位无符号整数）
// -----------------------------------------------------------------------------
using HcclComm = uintptr_t;           // typedef void* HcclComm
using HcclConn = uintptr_t;           // typedef void* HcclConn
using HcclCommSymWindow = uintptr_t;  // typedef void* HcclCommSymWindow
using HcclMemHandle = uintptr_t;      // typedef void* HcclMemHandle
using HcommMemHandle = uintptr_t;     // typedef void* HcommMemHandle
using EndpointHandle = uintptr_t;     // typedef void* EndpointHandle
using HcommSocket = uintptr_t;        // typedef void* HcommSocket
using HcclChannelConfig = uintptr_t;  // typedef void* HcclChannelConfig
using HcommChannelConfig = uintptr_t; // typedef void* HcommChannelConfig
using CcuKernelArg = uintptr_t;       // typedef void* CcuKernelArg

// ACL 类型（实际均为 void* 句柄）
using aclrtStream = uintptr_t;       // typedef void* aclrtStream
using aclrtDrvMemHandle = uintptr_t; // typedef void* aclrtDrvMemHandle

// 64 位整型句柄
using ChannelHandle = uint64_t;
using ThreadHandle = uint64_t;
using CcuInsHandle = uint64_t;
using CcuKernelHandle = uint64_t;
using CcuVariableHandle = uint64_t;
using CcuAddressHandle = uint64_t;
using CcuEventHandle = uint64_t;
using CcuBufferHandle = uint64_t;
using CcuLocalAddrHandle = uint64_t;
using CcuRemoteAddrHandle = uint64_t;
using HcommCcuCascCntHandle = uint64_t;
using HcommCcuResDescHandle = uint64_t;
using CcuLoop = uint64_t;
using CcuLoopGroup = uint64_t;
using CcuLoopExecutors = uint64_t;

// 团队/窗口句柄（原始 typedef uint64_t* / uint64_t*，以指向的地址整数表示）
using HcommTeamHandle = uint64_t;
using HcommWindowHandle = uint64_t;

// =============================================================================
// 常量
// =============================================================================

constexpr uint32_t HCCL_ROOT_INFO_BYTES = 4108;
constexpr uint32_t COMM_NAME_MAX_LENGTH = 128;
constexpr uint32_t BUFFER_NAME_MAX_LENGTH = 128;
constexpr uint32_t UDI_MAX_LENGTH = 128;
constexpr uint32_t HCCL_COMM_ALGO_MAX_LENGTH = 1600;
constexpr uint32_t HCCL_COMM_RETRY_ENABLE_MAX_LENGTH = 50;
constexpr uint32_t HCCL_COMM_RETRY_PARAMS_MAX_LENGTH = 128;

constexpr uint32_t HCCL_COMM_CONFIG_INFO_BYTES = 24;
constexpr uint32_t HCCL_COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0;
constexpr uint32_t HCCL_COMM_CONFIG_VERSION = 11;
constexpr uint32_t HCCL_COMM_DEFAULT_BUFFSIZE = 200;
constexpr uint32_t HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xffffffff;
constexpr uint32_t HCCL_COMM_DEFAULT_DETERMINISTIC = 0;
constexpr uint32_t HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET = 0xffffffff;
constexpr uint32_t HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0;
constexpr uint32_t HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xffffffff;
constexpr uint32_t HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xffffffff;
constexpr int32_t HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = static_cast<int32_t>(0xffffffff);
constexpr uint32_t HCCL_COMM_QOS_CONFIG_NOT_SET = 0xffffffff;
constexpr uint32_t HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET = 0xffffffff;
constexpr uint64_t HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE = 16ULL;

constexpr uint32_t HCCL_RES_TAG_MAX_LEN = 255;
constexpr uint32_t P2P_MAX_ARG_SIZE = 8192U;
constexpr uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;
constexpr uint32_t HCCL_OPDESC_MAGIC_WORD = 0x0f0f0f0f;
constexpr uint32_t HCCL_OPDESC_VERSION = 1;
constexpr uint32_t HCCL_KERNEL_SO_NAME_MAX_LEN = 256;
constexpr uint32_t HCCL_KERNEL_FUNC_NAME_MAX_LEN = 256;
constexpr uint32_t HCCL_KERNEL_LAUNCH_CFG_MAGIC_WORD = 0x0f0f0f0f;
constexpr uint32_t HCCL_KERNEL_LAUNCH_CFG_VERSION = 1;

constexpr uint32_t HCCL_CHANNEL_MAGIC_WORD = 0x0f0f0f0fU;
constexpr uint32_t HCCL_CHANNEL_VERSION = 1U;
constexpr uint32_t HCCL_CHANNEL_DESC_RAW_MAX_LEN = 128U;
constexpr uint32_t HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN = 255U;
constexpr uint32_t HCCL_TEAM_CREATE_DESC_MAGIC_WORD = 0x0fcf0f14U;
constexpr uint32_t HCCL_TEAM_CREATE_DESC_VERSION = 2U;

constexpr uint32_t COMM_LINK_MAGIC_WORD = 0x0f0e0f0f;
constexpr uint32_t COMM_LINK_VERSION = 1;
constexpr uint32_t COMM_ADDR_EID_LEN = 16U;

constexpr uint32_t HCOMM_THREAD_CONFIG_MAGIC_WORD = 0x0f0f0f2f;
constexpr uint32_t HCOMM_THREAD_CONFIG_VERSION = 1;
constexpr uint32_t HCOMM_CHANNEL_MAGIC_WORD = 0x0fcf0f0fU;
constexpr uint32_t HCOMM_CHANNEL_VERSION_ONE = 1U;
constexpr uint32_t HCOMM_CHANNEL_VERSION = 4U;
constexpr uint32_t HCOMM_CHANNEL_NAME_MAX_LEN = 191U;
constexpr uint32_t HCOMM_CHANNEL_DESC_RAW_MAX_LEN = 128U;

constexpr uint32_t CCU_CFG_MAGIC_WORD = 0xCC0CF000u;
constexpr uint32_t CCU_LOOP_CFG_VERSION = 1u;
constexpr uint32_t CCU_LOOPGROUP_CFG_VERSION = 1u;

// =============================================================================
// 枚举定义（取值与原始头文件保持一致）
// =============================================================================

// --- hccl_types.h ---
enum HcclResultEnum : int32_t
{
    HCCL_SUCCESS = 0,
    HCCL_E_PARA = 1,
    HCCL_E_PTR = 2,
    HCCL_E_MEMORY = 3,
    HCCL_E_INTERNAL = 4,
    HCCL_E_NOT_SUPPORT = 5,
    HCCL_E_NOT_FOUND = 6,
    HCCL_E_UNAVAIL = 7,
    HCCL_E_SYSCALL = 8,
    HCCL_E_TIMEOUT = 9,
    HCCL_E_OPEN_FILE_FAILURE = 10,
    HCCL_E_TCP_CONNECT = 11,
    HCCL_E_ROCE_CONNECT = 12,
    HCCL_E_TCP_TRANSFER = 13,
    HCCL_E_ROCE_TRANSFER = 14,
    HCCL_E_RUNTIME = 15,
    HCCL_E_DRV = 16,
    HCCL_E_PROFILING = 17,
    HCCL_E_CCE = 18,
    HCCL_E_NETWORK = 19,
    HCCL_E_AGAIN = 20,
    HCCL_E_REMOTE = 21,
    HCCL_E_SUSPENDING = 22,
    HCCL_E_OPRETRY_FAIL = 23,
    HCCL_E_OOM = 24,
    HCCL_E_IN_STATUS = 1041,
    HCCL_E_RESERVED = 1042,
};

enum HcclReduceOp : int32_t
{
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_PROD = 1,
    HCCL_REDUCE_MAX = 2,
    HCCL_REDUCE_MIN = 3,
    HCCL_REDUCE_RESERVED = 255,
};

enum HcclDataType : int32_t
{
    HCCL_DATA_TYPE_INT8 = 0,
    HCCL_DATA_TYPE_INT16 = 1,
    HCCL_DATA_TYPE_INT32 = 2,
    HCCL_DATA_TYPE_FP16 = 3,
    HCCL_DATA_TYPE_FP32 = 4,
    HCCL_DATA_TYPE_INT64 = 5,
    HCCL_DATA_TYPE_UINT64 = 6,
    HCCL_DATA_TYPE_UINT8 = 7,
    HCCL_DATA_TYPE_UINT16 = 8,
    HCCL_DATA_TYPE_UINT32 = 9,
    HCCL_DATA_TYPE_FP64 = 10,
    HCCL_DATA_TYPE_BFP16 = 11,
    HCCL_DATA_TYPE_INT128 = 12,
    HCCL_DATA_TYPE_HIF8 = 14,
    HCCL_DATA_TYPE_FP8E4M3 = 15,
    HCCL_DATA_TYPE_FP8E5M2 = 16,
    HCCL_DATA_TYPE_FP8E8M0 = 17,
    HCCL_DATA_TYPE_RESERVED = 255,
};

enum HcclConfig : int32_t
{
    HCCL_DETERMINISTIC = 0,
    HCCL_CONFIG_RESERVED = 1,
};

enum HcclCommConfigCapability : int32_t
{
    HCCL_COMM_CONFIG_BUFFER_SIZE = 0,
    HCCL_COMM_CONFIG_DETERMINISTIC = 1,
    HCCL_COMM_CONFIG_COMM_NAME = 2,
    HCCL_COMM_CONFIG_OP_EXPANSION_MODE = 3,
    HCCL_COMM_CONFIG_SUPPORT_INIT_BY_ENV = 4,
    HCCL_COMM_CONFIG_WORLD_RANKID = 5,
    HCCL_COMM_CONFIG_JOBID = 6,
    HCCL_COMM_CONFIG_ACLGRAPH_ZEROCOPY_ENABLE = 7,
    HCCL_COMM_CONFIG_EXEC_TIMEOUT = 8,
    HCCL_COMM_CONFIG_ALGO = 9,
    HCCL_COMM_CONFIG_RETRY = 10,
    HCCL_COMM_CONFIG_BUFFER_NAME = 11,
    HCCL_COMM_CONFIG_RESERVED = 12,
};

enum HcclSendRecvType : int32_t
{
    HCCL_SEND = 0,
    HCCL_RECV = 1,
    HCCL_SEND_RECV_RESERVED = 2,
};

enum HcclCMDType : int32_t
{
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE = 2,
    HCCL_CMD_REDUCE = 3,
    HCCL_CMD_SEND = 4,
    HCCL_CMD_RECEIVE = 5,
    HCCL_CMD_ALLGATHER = 6,
    HCCL_CMD_REDUCE_SCATTER = 7,
    HCCL_CMD_ALLTOALLV = 8,
    HCCL_CMD_ALLTOALLVC = 9,
    HCCL_CMD_ALLTOALL = 10,
    HCCL_CMD_GATHER = 11,
    HCCL_CMD_SCATTER = 12,
    HCCL_CMD_BATCH_SEND_RECV = 13,
    HCCL_CMD_BATCH_PUT = 14,
    HCCL_CMD_BATCH_GET = 15,
    HCCL_CMD_ALLGATHER_V = 16,
    HCCL_CMD_REDUCE_SCATTER_V = 17,
    HCCL_CMD_BATCH_WRITE = 18,
    HCCL_CMD_HALF_ALLTOALLV = 20,
    HCCL_CMD_ALL = 21,
    HCCL_CMD_FINALIZE = 100,
    HCCL_CMD_INTER_GROUP_SYNC = 101,
    HCCL_CMD_INIT = 102,
    HCCL_CMD_BARRIER = 103,
    HCCL_CMD_MAX = 104,
};

enum HcclCommStatus : int32_t
{
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDING = 1,
    HCCL_COMM_STATUS_INVALID = 254,
    HCCL_COMM_STATUS_RESERVED = 255,
};

enum HcclCommStatePhase : int32_t
{
    HCCL_COMM_STATE_PHASE_INVALID = -1,
    HCCL_COMM_STATE_PHASE_DESTROY_PRE = 0,
    HCCL_COMM_STATE_PHASE_DESTROY_POST = 1,
    HCCL_COMM_STATE_PHASE_RESUME_PRE = 2,
    HCCL_COMM_STATE_PHASE_RESUME_POST = 3,
};

// --- hccl_comm.h ---
enum HcclOpExpansionMode : int32_t
{
    HCCL_OP_EXPANSION_MODE_INVALID = -1,
    HCCL_OP_EXPANSION_MODE_AI_CPU = 0,
    HCCL_OP_EXPANSION_MODE_AIV = 1,
    HCCL_OP_EXPANSION_MODE_HOST = 2,
    HCCL_OP_EXPANSION_MODE_HOST_TS = 3,
    HCCL_OP_EXPANSION_MODE_CCU_MS = 4,
    HCCL_OP_EXPANSION_MODE_CCU_SCHED = 5,
    HCCL_OP_EXPANSION_AIV_ONLY = 6,
};

enum HcclConfigType : int32_t
{
    HCCL_CONFIG_TYPE_INVALID = -1,
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE = 0,
    HCCL_CONFIG_TYPE_HCCL_ALGO = 1,
    HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM = 2,
    HCCL_CONFIG_TYPE_DETERMINISTIC = 3,
    HCCL_CONFIG_TYPE_UDI = 4,
};

// --- hccl_res.h ---
enum HcclDedicatedThreadType : int32_t
{
    HCCL_DED_THREAD_TYPE_INVALID = -1,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH = 0,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE = 1,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE = 2,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH = 3,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE = 4,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE = 5,
};

// --- hcomm_res_defs.h ---
enum ThreadType : int32_t
{
    THREAD_TYPE_INVALID = -1,
    THREAD_TYPE_TS = 0,
    THREAD_TYPE_CPU = 1,
};

enum CommEngine : int32_t
{
    COMM_ENGINE_RESERVED = -1,
    COMM_ENGINE_CPU = 0,
    COMM_ENGINE_CPU_TS = 1,
    COMM_ENGINE_AICPU = 2,
    COMM_ENGINE_AICPU_TS = 3,
    COMM_ENGINE_AIV = 4,
    COMM_ENGINE_CCU = 5,
};

enum CommProtocol : int32_t
{
    COMM_PROTOCOL_RESERVED = -1,
    COMM_PROTOCOL_HCCS = 0,
    COMM_PROTOCOL_ROCE = 1,
    COMM_PROTOCOL_PCIE = 2,
    COMM_PROTOCOL_SIO = 3,
    COMM_PROTOCOL_UB_CTP = 4,
    COMM_PROTOCOL_UBC_CTP = 4,
    COMM_PROTOCOL_UBC_TP = 5,
    COMM_PROTOCOL_UB_MEM = 6,
    COMM_PROTOCOL_UBOE = 7,
    COMM_PROTOCOL_HCCS_ONLY = 8,
    COMM_PROTOCOL_UB_RTP = 9,
    COMM_PROTOCOL_UBG = 9,
};

enum CommAddrType : int32_t
{
    COMM_ADDR_TYPE_RESERVED = -1,
    COMM_ADDR_TYPE_IP_V4 = 0,
    COMM_ADDR_TYPE_IP_V6 = 1,
    COMM_ADDR_TYPE_ID = 2,
    COMM_ADDR_TYPE_EID = 3,
};

enum EndpointLocType : int32_t
{
    ENDPOINT_LOC_TYPE_RESERVED = -1,
    ENDPOINT_LOC_TYPE_DEVICE = 0,
    ENDPOINT_LOC_TYPE_HOST = 1,
};

enum CommMemType : int32_t
{
    COMM_MEM_TYPE_INVALID = -1,
    COMM_MEM_TYPE_DEVICE = 0,
    COMM_MEM_TYPE_HOST = 1,
    COMM_MEM_TYPE_CCU = 2,
};

enum HcommSocketRole : int32_t
{
    HCOMM_SOCKET_ROLE_RESERVED = -1,
    HCOMM_SOCKET_ROLE_CLIENT = 0,
    HCOMM_SOCKET_ROLE_SERVER = 1,
};

enum HcommEndpointFeatureType : int32_t
{
    HCOMM_ENDPOINT_FEATURE_INVALID = -1,
    HCOMM_ENDPOINT_FEATURE_NDA = 0,
};

enum ThreadResType : int32_t
{
    THREAD_RES_TYPE_INVALID = -1,
    THREAD_RES_TYPE_STREAM = 0,
};

// --- hccl_channel.h ---
enum HcclChannelConfigType : int32_t
{
    HCCL_CHANNEL_CONFIG_TYPE_INVALID = -1,
    HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
    HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG = 1,
};

// --- hcomm_channel.h ---
enum HcommChannelConfigType : int32_t
{
    HCOMM_CHANNEL_CONFIG_TYPE_INVALID = -1,
    HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
};

enum HcommChannelStatus : int32_t
{
    HCOMM_CHANNEL_STATUS_READY = 0,
    HCOMM_CHANNEL_STATUS_CONNECTING = 1,
    HCOMM_CHANNEL_STATUS_FAILED_INTERNAL = 2,
};

// --- hcomm_primitives.h ---
enum HcommReduceOp : int32_t
{
    HCOMM_REDUCE_SUM = 0,
    HCOMM_REDUCE_PROD = 1,
    HCOMM_REDUCE_MAX = 2,
    HCOMM_REDUCE_MIN = 3,
    HCOMM_REDUCE_RESERVED = 255,
};

enum HcommDataType : int32_t
{
    HCOMM_DATA_TYPE_INT8 = 0,
    HCOMM_DATA_TYPE_INT16 = 1,
    HCOMM_DATA_TYPE_INT32 = 2,
    HCOMM_DATA_TYPE_FP16 = 3,
    HCOMM_DATA_TYPE_FP32 = 4,
    HCOMM_DATA_TYPE_INT64 = 5,
    HCOMM_DATA_TYPE_UINT64 = 6,
    HCOMM_DATA_TYPE_UINT8 = 7,
    HCOMM_DATA_TYPE_UINT16 = 8,
    HCOMM_DATA_TYPE_UINT32 = 9,
    HCOMM_DATA_TYPE_FP64 = 10,
    HCOMM_DATA_TYPE_BFP16 = 11,
    HCOMM_DATA_TYPE_INT128 = 12,
    HCOMM_DATA_TYPE_HIF8 = 14,
    HCOMM_DATA_TYPE_FP8E4M3 = 15,
    HCOMM_DATA_TYPE_FP8E5M2 = 16,
    HCOMM_DATA_TYPE_FP8E8M0 = 17,
    HCOMM_DATA_TYPE_RESERVED = 255,
};

enum HcommTransferType : int32_t
{
    HCOMM_TRANSFER_TYPE_INVALID = -1,
    HCOMM_TRANSFER_TYPE_WRITE = 0,
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE = 1,
    HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY = 2,
    HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY = 3,
    HCOMM_TRANSFER_TYPE_READ = 4,
    HCOMM_TRANSFER_TYPE_READ_REDUCE = 5,
    HCOMM_TRANSFER_TYPE_NOTIFY_RECORD = 6,
};

// --- hccl_rank_graph.h ---
enum EndpointAttr : int32_t
{
    ENDPOINT_ATTR_INVALID = -1,
    ENDPOINT_ATTR_BW_COEFF = 0,
    ENDPOINT_ATTR_DIE_ID = 1,
    ENDPOINT_ATTR_LOCATION = 2,
};

enum CommTopo : int32_t
{
    COMM_TOPO_RESERVED = -1,
    COMM_TOPO_CLOS = 0,
    COMM_TOPO_1DMESH = 1,
    COMM_TOPO_910_93 = 2,
    COMM_TOPO_310P = 3,
    COMM_TOPO_A2AXSERVER = 4,
    COMM_TOPO_CUSTOM = 5,
};

enum HcclHeterogMode : int32_t
{
    HCCL_HETEROG_MODE_INVALID = -1,
    HCCL_HETEROG_MODE_HOMOGENEOUS = 0,
    HCCL_HETEROG_MODE_MIX_A2_A3 = 1,
};

// --- ccu_types.h ---
enum CcuResultEnum : int32_t
{
    CCU_SUCCESS = 0,
    CCU_E_PARA = 1,
    CCU_E_PTR = 2,
    CCU_E_INTERNAL = 4,
    CCU_E_NOT_SUPPORT = 5,
    CCU_E_NOT_FOUND = 6,
    CCU_E_UNAVAIL = 7,
    CCU_E_RUNTIME = 15,
    CCU_E_DRV_START = 4096,
    CCU_E_DRV_INIT_FAILED = 4097,
    CCU_E_DRV_BUSY = 4098,
    CCU_E_DRV_END = 4224,
    CCU_E_RESERVED = 9216,
};

enum CcuConditionType : int32_t
{
    CCU_CONDITION_EQ = 0,
    CCU_CONDITION_NE = 1,
    CCU_CONDITION_LT = 2,
    CCU_CONDITION_LE = 3,
    CCU_CONDITION_GT = 4,
    CCU_CONDITION_GE = 5,
};

// --- ccu_res_defs.h ---
enum HcommCcuResType : int32_t
{
    HCOMM_CCU_RES_TYPE_INVALID = -1,
    HCOMM_CCU_RES_TYPE_LOOP = 0,
    HCOMM_CCU_RES_TYPE_CCU_BUF = 1,
    HCOMM_CCU_RES_TYPE_VARIABLE = 2,
    HCOMM_CCU_RES_TYPE_ADDRESS = 3,
    HCOMM_CCU_RES_TYPE_EVENT = 4,
    HCOMM_CCU_RES_TYPE_CCU_THREAD = 5,
    HCOMM_CCU_RES_TYPE_INSTRUCTION = 6,
    HCOMM_CCU_RES_TYPE_CASC_CNT = 7,
};

// =============================================================================
// 结构体定义（字段顺序/类型/对齐与原始头文件一致）
// =============================================================================

// hcomm_res_defs.h：CommAbiHeader
struct CommAbiHeader
{
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
};

// hcomm_res_defs.h：ThreadConfig
struct ThreadConfig
{
    CommAbiHeader header;
    uint16_t notifyNumPerThread;
    uint8_t reserved[14];
};

// hcomm_res_defs.h：CommAddr（union 以 raws/id/eid 镜像，尺寸 36 字节）
struct CommAddr
{
    CommAddrType type;
    union
    {
        uint8_t raws[36];
        uint32_t id;
        uint8_t eid[16];
    };
};

// hcomm_res_defs.h：EndpointLoc（union 以 raws/device/host 镜像，尺寸 60 字节）
struct EndpointLoc
{
    EndpointLocType locType;
    union
    {
        uint8_t raws[60];
        struct
        {
            uint32_t devPhyId;
            uint32_t superDevId;
            uint32_t serverIdx;
            uint32_t superPodIdx;
        } device;
        struct
        {
            uint32_t id;
        } host;
    };
};

// hcomm_res_defs.h：EndpointDesc
struct EndpointDesc
{
    CommProtocol protocol;
    CommAddr commAddr;
    EndpointLoc loc;
    union
    {
        uint8_t raws[52];
    };
};

// hcomm_res_defs.h：CommMem
struct CommMem
{
    CommMemType type;
    uintptr_t addr; // 原始为 void* addr
    uint64_t size;
};

// hcomm_team_defs.h：HcommTeamSyncMemRequirement
struct HcommTeamSyncMemRequirement
{
    uint32_t signalCount;
    uint32_t counterCount;
    uint32_t barrierCount;
    uint32_t reserved[5];
};

// hccl_types.h：HcclRootInfo
struct HcclRootInfo
{
    char internal[HCCL_ROOT_INFO_BYTES];
};

// hccl_types.h：HcclConfigValue
union HcclConfigValue
{
    int32_t value;
};

// hccl_types.h：HcclCommConfig
struct HcclCommConfig
{
    char reserved[HCCL_COMM_CONFIG_INFO_BYTES]; // ABI 头（size/magic/version/reserved）
    uint32_t hcclBufferSize;
    uint32_t hcclDeterministic;
    char hcclCommName[COMM_NAME_MAX_LENGTH];
    char hcclUdi[UDI_MAX_LENGTH];
    uint32_t hcclOpExpansionMode;
    uint32_t hcclRdmaTrafficClass;
    uint32_t hcclRdmaServiceLevel;
    uint32_t hcclWorldRankID;
    uint64_t hcclJobID;
    uint8_t aclGraphZeroCopyEnable;
    int32_t hcclExecTimeOut;
    char hcclAlgo[HCCL_COMM_ALGO_MAX_LENGTH];
    char hcclRetryEnable[HCCL_COMM_RETRY_ENABLE_MAX_LENGTH];
    char hcclRetryParams[HCCL_COMM_RETRY_PARAMS_MAX_LENGTH];
    char hcclBufferName[BUFFER_NAME_MAX_LENGTH];
    uint32_t hcclQos;
    uint64_t hcclSymWinMaxMemSizePerRank;
    uint32_t hcclChannelSqDepth;
};

// hccl_types.h：HcclSendRecvItem
struct HcclSendRecvItem
{
    HcclSendRecvType sendRecvType;
    uintptr_t buf; // 原始为 void* buf
    uint64_t count;
    HcclDataType dataType;
    uint32_t remoteRank;
};

// hccl_launch.h：HcclP2pKernelParam / HcclOpP2pDesc / HcclOpDesc
struct HcclP2pKernelParam
{
    ThreadHandle sendRecvThread;
    uint8_t opParams[P2P_MAX_ARG_SIZE];
};

struct HcclOpP2pDesc
{
    uintptr_t buffer; // void* buffer
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    uintptr_t unfoldStream; // void* unfoldStream
};

struct HcclOpDesc
{
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union
    {
        uint8_t raws[76];
        HcclOpP2pDesc p2p;
    };
};

// hccl_launch.h：HcclKernelFuncInfo / HcclKernelLaunchCfg
struct HcclKernelFuncInfo
{
    char kernelSoName[HCCL_KERNEL_SO_NAME_MAX_LEN];
    char kernelFuncName[HCCL_KERNEL_FUNC_NAME_MAX_LEN];
    uintptr_t args; // void* args
    uint32_t argSize;
};

struct HcclKernelLaunchCfg
{
    CommAbiHeader header;
    uint64_t timeOut;
    uint8_t reserved[104];
};

// hccl_channel.h：HcclChannelDesc（union 以 raws/roceAttr/ubMemAttr 镜像）
struct HcclChannelDesc
{
    CommAbiHeader header;
    uint32_t remoteRank;
    CommProtocol channelProtocol;
    EndpointDesc localEndpoint;
    EndpointDesc remoteEndpoint;
    uint32_t notifyNum;
    uintptr_t memHandles; // 原始为 HcclMemHandle* memHandles（数组首地址）
    uint32_t memHandleNum;
    union
    {
        uint8_t raws[HCCL_CHANNEL_DESC_RAW_MAX_LEN];
        struct
        {
            uint32_t queueNum;
            uint32_t retryCnt;
            uint32_t retryInterval;
            uint8_t tc;
            uint8_t sl;
        } roceAttr;
        struct
        {
            uint8_t pathMode;
        } ubMemAttr;
    };
};

// hcomm_channel.h：HcommChannelDesc（union 以 raws/roceAttr/hccsAttr/ubAttr/ubMemAttr 镜像）
struct HcommChannelDesc
{
    CommAbiHeader header;
    EndpointDesc remoteEndpoint;
    uint32_t notifyNum;
    bool exchangeAllMems;
    uintptr_t memHandles; // HcommMemHandle* memHandles
    uint32_t memHandleNum;
    HcommSocket socket;
    HcommSocketRole role;
    uint16_t port;
    union
    {
        uint8_t raws[HCOMM_CHANNEL_DESC_RAW_MAX_LEN];
        struct
        {
            uint32_t queueNum;
            uint32_t retryCnt;
            uint32_t retryInterval;
            uint8_t tc;
            uint8_t sl;
            uint32_t qpThreshold;
            uint32_t cqAttrFlags;
            uintptr_t srcPortList; // uint16_t* srcPortList
            uint32_t sqDepth;
            uint32_t scqDepth;
        } roceAttr;
        struct
        {
            uint32_t qos;
        } hccsAttr;
        struct
        {
            uint32_t sqDepth;
            uint32_t scqDepth;
        } ubAttr;
        struct
        {
            uint8_t pathMode;
        } ubMemAttr;
    };
    uint32_t qos;
    uintptr_t channelName; // const char* channelName
};

// hccl_team.h：HcclTeamCreateDesc
struct HcclTeamCreateDesc
{
    CommAbiHeader header;
    uintptr_t rankIds; // const uint32_t* rankIds
    uint32_t rankNum;
    uint32_t selfRankId;
    uint32_t netLayer;
    CommProtocol protocol;
    HcommTeamSyncMemRequirement requirement;
    CommEngine engine;
    uint32_t notifyNum;
    uint32_t channelCnt;
    uint32_t isSharedQueue;
    char sharedQueueTag[HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN];
    uint32_t reserved[8];
};

// hccl_rank_graph.h：CommLink
struct CommLink
{
    CommAbiHeader header;
    EndpointDesc srcEndpointDesc;
    EndpointDesc dstEndpointDesc;
    union
    {
        uint8_t raws[128];
        struct
        {
            CommProtocol linkProtocol;
            uint8_t hop;
        };
    } linkAttr;
};

// hcomm_primitives.h：HcommBatchTransferDesc
struct HcommBatchTransferDesc
{
    HcommTransferType transType;
    uint8_t reserved[4];
    union
    {
        uint8_t raws[56];
        struct
        {
            uint64_t len;
            uintptr_t dst;
            uintptr_t src;
        } write;
        struct
        {
            uint64_t len;
            uintptr_t dst;
            uintptr_t src;
        } read;
        struct
        {
            uint64_t count;
            uintptr_t dst;
            uintptr_t src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
        } reduce;
        struct
        {
            uint32_t notifyIdx;
        } notifyRecord;
        struct
        {
            uint64_t len;
            uintptr_t dst;
            uintptr_t src;
            uint32_t notifyIdx;
        } writeWithNotify;
        struct
        {
            uint64_t count;
            uintptr_t dst;
            uintptr_t src;
            HcommReduceOp reduceOp;
            HcommDataType dataType;
            uint32_t notifyIdx;
        } writeReduceWithNotify;
    } transferInfo;
};

// ccu_types.h：CcuLoopConfig / CcuLoopGroupConfig / CcuCfgHeader / CcuLoopCfg / CcuLoopGroupCfg
struct CcuLoopConfig
{
    uint64_t addrOffset;
    uint64_t iterNum;
};

struct CcuLoopGroupConfig
{
    uint32_t cloneNum;
    uint32_t cloneLoopOffset;
    uint32_t addrOffset;
    uint32_t ccuBufferOffset;
    uint32_t eventOffset;
};

struct CcuCfgHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t size;
    uint32_t reserved;
};

struct CcuLoopCfg
{
    CcuCfgHeader header;
    uint64_t addrOffset;
    uint64_t iterNum;
    uint64_t reserved[4];
};

struct CcuLoopGroupCfg
{
    CcuCfgHeader header;
    uint32_t cloneNum;
    uint32_t cloneLoopOffset;
    uint32_t addrOffset;
    uint32_t ccuBufferOffset;
    uint32_t eventOffset;
    uint32_t varOffset;
    uint32_t reserved[8];
};

} // namespace pyhccl

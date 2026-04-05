#!/usr/bin/env python3
"""
极简 MLP 张量并行示例：类名与 vLLM/sglang 的 Qwen2MoeMLP 对齐（ColumnParallel gate_up + RowParallel down + SiLU*gate）。

命令行示例（AMD 可把 CUDA_VISIBLE_DEVICES 换成 HIP_VISIBLE_DEVICES）:

  TP=1（单进程，无 torchrun，tp_size=1）:
    python3 mlp_tp_example.py

  TP=1（单进程，torchrun，WORLD_SIZE=1）:
    HIP_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 mlp_tp_example.py

  TP=2（两进程两卡，tp_size=2，gate_up 每卡 last_dim 为 2*inter/2）:
    HIP_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 mlp_tp_example.py

说明:
- TP 度数 = ``torchrun --nproc_per_node``（WORLD_SIZE）；单进程 ``python3 mlp_tp_example.py`` 等价 tp_size=1。
- 可见卡数须 ≥ ``nproc_per_node``（例如只暴露 2,3 两张卡则 nproc 最多为 2）。

各 rank 的 shape 打印由文件顶部宏 ``MLP_TP_LOG_SHAPES`` 控制（默认 True）。

"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# 宏：是否打印各 rank 上 x / gate_up / mid / y 的 shape（改成 False 即关）
MLP_TP_LOG_SHAPES = True


def _init_dist() -> tuple[int, int, torch.device]:
    """对齐 SGLang `init_distributed_environment` 的两种入口：torchrun(env) 与单进程(world_size=1 可不建组)。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ndev = torch.cuda.device_count()
    if local_rank >= ndev:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} 超出可见 GPU 数量 ({ndev})。"
            f"--nproc_per_node 不能大于 HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES 暴露的卡数。"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), device

    # torchrun / 外部已设好 RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT → 默认 env://
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), device

    # 单进程：SGLang 在仅 1 个 model worker 时也要通信组时可走 tcp://；本示例 tp=1 无需 all_reduce，不 init 即可
    return 0, 1, device


def _all_reduce_sum_(t: torch.Tensor) -> None:
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


class SiluAndMul(nn.Module):
    """与 sglang SiluAndMul 一致：沿最后一维对半拆成 gate / up。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class MergedColumnParallelLinear(nn.Module):
    """
    合并 gate_proj + up_proj，按 intermediate 维对齐切分输出（每卡持有 inter/tp 的 gate 与 up 行）。
    权重布局与整卡合并线性一致：前 inter 行为 gate，后 inter 行为 up。
    """

    def __init__(
        self,
        hidden_size: int,
        output_sizes: list[int],
        *,
        tp_rank: int,
        tp_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(output_sizes) != 2 or output_sizes[0] != output_sizes[1]:
            raise ValueError("本示例仅支持 [intermediate, intermediate] 的 merged gate_up。")
        inter = output_sizes[0]
        if inter % tp_size != 0:
            raise ValueError(f"intermediate_size={inter} 必须能被 tp_size={tp_size} 整除。")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.inter_local = inter // tp_size
        out_local = 2 * self.inter_local

        # 整权重形状 (2*inter, hidden)；每卡取 gate/up 各一段行
        w = torch.empty(2 * inter, hidden_size, device=device, dtype=dtype)
        nn.init.normal_(w, std=0.02)
        g0, g1 = tp_rank * self.inter_local, (tp_rank + 1) * self.inter_local
        u0, u1 = inter + g0, inter + g1
        self.weight = nn.Parameter(torch.cat([w[g0:g1], w[u0:u1]], dim=0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # x: [*, hidden] -> [*, 2*inter_local]
        return F.linear(x, self.weight), None


class RowParallelLinear(nn.Module):
    """down_proj：按 intermediate 输入维切分，前向后 all-reduce（求和）。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        tp_rank: int,
        tp_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if input_size % tp_size != 0:
            raise ValueError(f"input_size={input_size} 必须能被 tp_size={tp_size} 整除。")
        self.in_local = input_size // tp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        w = torch.empty(output_size, input_size, device=device, dtype=dtype)
        nn.init.normal_(w, std=0.02)
        c0, c1 = tp_rank * self.in_local, (tp_rank + 1) * self.in_local
        self.weight = nn.Parameter(w[:, c0:c1])

    def forward(
        self, x: torch.Tensor, skip_all_reduce: bool = False
    ) -> Tuple[torch.Tensor, None]:
        y = F.linear(x, self.weight)
        #print(f"RowParallelLinear forward: y.shape={y.shape},device={y.device}")
        if not skip_all_reduce and self.tp_size > 1:
            _all_reduce_sum_(y)
        return y, None


class Qwen2MoeMLP(nn.Module):
    """
    与 vLLM `Qwen2MoeMLP` 类似的构造/前向（去掉 quant、expert_gate、prefix 等与并行无关的细节）。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        reduce_results: bool = True,
        tp_rank: int = 0,
        tp_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        dev = device or torch.device("cuda")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.reduce_results = reduce_results
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            tp_rank=tp_rank,
            tp_size=tp_size,
            device=dev,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            device=dev,
            dtype=dtype,
        )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        """
        def forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            out = self.act_fn(gate_up)
            out, _ = self.down_proj(out)

            # if self.expert_gate is not None:
            #     out = F.sigmoid(self.expert_gate(x)[0]) * out

            return out
        """
        gr = dist.get_rank() if dist.is_initialized() else 0
        if MLP_TP_LOG_SHAPES:
            print(f"[rank={gr} tp={self.tp_rank}/{self.tp_size}] x {tuple(x.shape)}")
        gate_up, _ = self.gate_up_proj(x)
        if MLP_TP_LOG_SHAPES:
            print(
                f"[rank={gr} tp={self.tp_rank}/{self.tp_size}] "
                f"gate_up {tuple(gate_up.shape)}  # 列并行，last_dim=2*inter/tp"
            )
        x = self.act_fn(gate_up)
        if MLP_TP_LOG_SHAPES:
            print(
                f"[rank={gr} tp={self.tp_rank}/{self.tp_size}] "
                f"mid {tuple(x.shape)} | down_proj.weight {tuple(self.down_proj.weight.shape)} "
                f"(hidden×inter/tp)"
            )
        skip_ar = (should_allreduce_fusion or use_reduce_scatter) or not self.reduce_results
        x, _ = self.down_proj(x, skip_all_reduce=skip_ar)
        if MLP_TP_LOG_SHAPES:
            print(
                f"[rank={gr} tp={self.tp_rank}/{self.tp_size}] "
                f"y {tuple(x.shape)}  # 行并行局部 shape 已是 [B,hidden]，all_reduce 后数值对齐单卡"
            )
        return x



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    # 与 HuggingFace Qwen/Qwen3.5-27B 的 text_config 一致（config.json）
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--inter", type=int, default=17408)
    args = p.parse_args()

    rank, world_size, device = _init_dist()
    if dist.is_initialized():
        tp_size = world_size
        tp_rank = rank
    else:
        tp_size = 1
        tp_rank = 0

    torch.manual_seed(0)
    h, inter, b = args.hidden, args.inter, args.batch
    if inter % tp_size != 0:
        raise SystemExit(
            f"intermediate_size={inter} 必须能被 tp_size={tp_size} 整除。"
        )

    m = Qwen2MoeMLP(
        h,
        inter,
        "silu",
        tp_rank=tp_rank,
        tp_size=tp_size,
        device=device,
        dtype=torch.float32,
    ).to(device)

    x = torch.randn(b, h, device=device, dtype=torch.float32)
    y = m(x)
    if rank == 0:
        print(f"Qwen2MoeMLP 前向完成: x.shape={tuple(x.shape)} -> y.shape={tuple(y.shape)}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

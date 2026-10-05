import torch

def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    """Same layout as ``tests.utils.shuffle_weight``; copied to avoid ``from tests...`` (can import wrong ``tests``)."""
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    """
    x_.view(-1,
         N_dim // 16,    # 维1：沿 N 走了多少个「16 的块」
         16,             # 维2：在这 16 宽的条带里，是第几个 N
         K_dim // 32,    # 维3：沿 K 走了多少个「32 的块」
         4,              # 维4：一个 32 元素块再拆成 4 段，每段 8 个元素 (32/8=4)
         8)              # 维5：这 8 个 fp16 是内存里最细的一段
    (1, 256, 16, 128, 4, 8)
     ↑    ↑   ↑   ↑  ↑
   batch  N块 BN K块 子K 8元

    (1, 256, 128, 4, 16, 8)
    permute 之后 所以严格说：物理存储顺序往往还没变，变的是「多维下标怎么映射到线性下标」。
    contiguous() 会按 当前维度的逻辑顺序 把数据 重新排好并（通常）拷到新的一块连续存储。
所以 最终返回的 preshuffle_B：线性内存里的字节顺序 已经变了；这才是「preshuffle 真的生效」的那一步。
    """
    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_

def main():

    tmp = torch.tensor(16 * 32, dtype=torch.float16, device="cuda") # bf16在256-1024 精度不足 相邻间隔变小所以这里用float16

    # 生成从 0 到 (16*32 - 1) 的递增 tensor，保持 dtype 和 device 一致
    before_tensor = torch.arange(
        tmp.item(), 
        dtype=tmp.dtype, 
        device=tmp.device
    ).reshape(16,32)
    #print(tmp)
    torch.set_printoptions(linewidth=300)
    print(before_tensor)

    after_tensor = shuffle_weight(before_tensor)
    print(after_tensor)

if __name__ == "__main__":
    main()
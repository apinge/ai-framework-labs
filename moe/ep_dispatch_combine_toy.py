import torch


def fake_expert(expert_id: int, x: torch.Tensor) -> torch.Tensor:
    return x + expert_id * 100.0


def main() -> None:
    T = 4  # tokens 当前 rank 上进入 MoE 的 token 数
    H = 3  # hidden size, 每个 token hidden vector 的长度
    E = 3  # experts
    K = 2  # top-k experts per token

    # T 个 token hidden vectors
    hidden_states = torch.arange(T * H, dtype=torch.float32).view(T, H)

    # 这里是每个token选出的topk专家列表 [T,K]
    topk_ids = torch.tensor(
        [
            [0, 2],
            [1, 0],
            [2, 1],
            [0, 1],
        ],
        dtype=torch.long,
    )
    # 每个 token 对应 K 个 expert 的 routing weight, 就是topk softmax后面出来的那个weight, [T,K]
    topk_weights = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )

    print("hidden_states [T, H]")
    print(hidden_states)
    print("\ntopk_ids [T, K]")
    print(topk_ids)
    print("\ntopk_weights [T, K]")
    print(topk_weights)
    print("\ntopk_weights[t, k] is a scalar:", topk_weights[0, 0].item())

    assignments = []
    for t in range(T):
        for k in range(K):
            e = topk_ids[t, k].item()
            assignments.append((e, t, k))
            # e = expert_id
            # t = token index
            # k = topk slot index
    # Python tuple 默认按第 1 个元素、再第 2 个、再第 3 个排序，所以它的效果是：
    # 先按 expert_id 分组
    # 同一个 expert 里，再按 token_id / topk_slot 排一下
    # 真实 sglang 里，不一定有一个 Python 层显式 sort()，但“把 token 按 expert 分组/重排”这个动作仍然需要，只是可能藏在 DeepEP/Mori/AITER/DeepGEMM 的 kernel 里面。
    assignments.sort()

    dispatch_expert_id = torch.tensor([e for e, _, _ in assignments])
    dispatch_token_idx = torch.tensor([t for _, t, _ in assignments])
    dispatch_k_idx = torch.tensor([k for _, _, k in assignments])
    dispatch_x = hidden_states[dispatch_token_idx]
    dispatch_weight = topk_weights[dispatch_token_idx, dispatch_k_idx]

    print("\nAfter dispatch: rows are grouped by expert_id")
    print("dispatch_expert_id:", dispatch_expert_id.tolist())
    print("dispatch_token_idx: ", dispatch_token_idx.tolist())
    print("dispatch_k_idx:     ", dispatch_k_idx.tolist())
    print("dispatch_weight:    ", dispatch_weight.tolist())
    print("dispatch_x [T*K, H]")
    print(dispatch_x)

    expert_outputs = torch.empty_like(dispatch_x)
    for e in range(E):
        mask = dispatch_expert_id == e
        expert_outputs[mask] = fake_expert(e, dispatch_x[mask])

    print("\nAfter fake MoE: expert_outputs [T*K, H]")
    print(expert_outputs)

    final = torch.zeros(T, H)
    for row in range(dispatch_x.shape[0]):
        t = dispatch_token_idx[row]
        w = dispatch_weight[row]
        final[t] += w * expert_outputs[row]

    print("\nAfter combine: final_hidden_states [T, H]")
    print(final)


if __name__ == "__main__":
    main()

"""
hidden_states [T, H]
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])

topk_ids [T, K]
tensor([[0, 2],
        [1, 0],
        [2, 1],
        [0, 1]])

topk_weights [T, K]
tensor([[0.7000, 0.3000],
        [0.6000, 0.4000],
        [0.5000, 0.5000],
        [0.2000, 0.8000]])

topk_weights[t, k] is a scalar: 0.699999988079071

After dispatch: rows are grouped by expert_id
dispatch_expert_id: [0, 0, 0, 1, 1, 1, 2, 2]
dispatch_token_idx:  [0, 1, 3, 1, 2, 3, 0, 2]
dispatch_k_idx:      [0, 1, 0, 0, 1, 1, 1, 0]
dispatch_weight:     [0.699999988079071, 0.4000000059604645, 0.20000000298023224, 0.6000000238418579, 0.5, 0.800000011920929, 0.30000001192092896, 0.5]
dispatch_x [T*K, H]
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 9., 10., 11.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.],
        [ 0.,  1.,  2.],
        [ 6.,  7.,  8.]])

After fake MoE: expert_outputs [T*K, H]
tensor([[  0.,   1.,   2.],
        [  3.,   4.,   5.],
        [  9.,  10.,  11.],
        [103., 104., 105.],
        [106., 107., 108.],
        [109., 110., 111.],
        [200., 201., 202.],
        [206., 207., 208.]])

After combine: final_hidden_states [T, H]
tensor([[ 60.0000,  61.0000,  62.0000],
        [ 63.0000,  64.0000,  65.0000],
        [156.0000, 157.0000, 158.0000],
        [ 89.0000,  90.0000,  91.0000]])
"""
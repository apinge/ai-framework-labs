import torch

# ------------------------------------
# 原始 tensor
# ------------------------------------
x = torch.arange(8).reshape(2, 2, 2)

print("===== x =====")
print(x)
print("shape :", x.shape)
print("stride:", x.stride())
print("contiguous:", x.is_contiguous())

print("\nflatten storage:")
print(x.reshape(-1).tolist())

# ------------------------------------
# permute
# ------------------------------------
y = x.permute(1, 2, 0)

print("\n\n===== y = x.permute(1,2,0) =====")
print(y)
print("shape :", y.shape)
print("stride:", y.stride())
print("contiguous:", y.is_contiguous())

print("\nlogical iteration order of y:")
for a in range(y.shape[0]):
    for b in range(y.shape[1]):
        for c in range(y.shape[2]):
            print(
                f"y[{a},{b},{c}] = {y[a,b,c].item()}"
            )

# ------------------------------------
# contiguous
# ------------------------------------
z = y.contiguous()

print("\n\n===== z = y.contiguous() =====")
print(z)
print("shape :", z.shape)
print("stride:", z.stride())
print("contiguous:", z.is_contiguous())

print("\nnew storage after contiguous:")
print(z.reshape(-1).tolist())

# ------------------------------------
# 对比 storage
# ------------------------------------
print("\n\n===== Compare =====")

print("x storage:")
print(x.reshape(-1).tolist())

print("y logical order:")
print([v.item() for v in y.reshape(-1)])

print("z storage:")
print(z.reshape(-1).tolist())
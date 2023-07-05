import torch

free = 0
total = 0
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.mem_get_info(i)
    free += mem[0]/1024/1024/1024
    total += mem[1]/1024/1024/1024

print("Cuda devices:", torch.cuda.device_count())
print("Total VRAM usage: %0.4f GB / %0.4f GB (%0.4f GB free)" %(total-free, total, free))

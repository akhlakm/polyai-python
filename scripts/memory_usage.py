import torch

mem = torch.cuda.mem_get_info()
free = mem[0]/1024/1024/1024
total = mem[1]/1024/1024/1024
print("VRAM usage: %0.4f GB / %0.4f GB (%0.4f GB free)" %(total-free, total, free))

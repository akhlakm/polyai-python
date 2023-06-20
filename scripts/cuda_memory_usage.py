import torch

mem = torch.cuda.mem_get_info()
o = mem[0]/1024/1024/1024
t = mem[1]/1024/1024/1024

print("Usage: %0.4f GB / %0.4f GB" %(o, t))


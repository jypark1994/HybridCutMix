import torch

ngpu = torch.cuda.device_count()

N, M, K = 256, 512, 1024 # example matmul sizes

A = []
B = torch.randn(M, K, device='cuda:0')

# randomly initialize A
for i in range(ngpu):
    # each GPU has a slice of A
    A.append(torch.randn(N // ngpu, M, device='cuda:' + str(i)))

# now let's matmul

# Step 1: make a copy of B on each GPU
B_ = [B]
for i in range(ngpu):
    if i != 0:
        B_.append(B.to('cuda:' + str(i)))

# Step 2: issue the matmul on each GPU
C_ = []
for i in range(ngpu):
    C_.append(torch.matmul(A[i], B_[i]))

C = torch.empty(N, K)

for i in range(ngpu):
    start_index = i * (N//ngpu)
    C[start_index:start_index+(N//ngpu), :].copy_(C_[i])

# C is the final result gathered on GPU-0
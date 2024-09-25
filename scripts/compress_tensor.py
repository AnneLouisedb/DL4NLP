import torch

# Load torch tensor
tensor_ans = torch.load('/home/scur1744/DL4NLP/data/llama/test-answer.pt')

# Print the shape of the tensor
print(tensor_ans.shape)

# Make new tensor of shape (tensor_ans.shape[0], 256, 2)
tensor_ans_new = torch.zeros(tensor_ans.shape[0], 2, 256)

max_val, idx = torch.max(tensor_ans, dim=-1) # Shape of idx is (tensor_ans.shape[0], 256)
print(f"Max val shape: {max_val.shape}")
print(f"Idx shape: {idx.shape}")

tensor_ans_new[:, 0, :] = idx
tensor_ans_new[:, 1, :] = max_val
# Save tensor
torch.save(tensor_ans_new, '/home/scur1744/DL4NLP/data/llama/tensors/test-answer.pt')

tensor_follow_up = torch.load('/home/scur1744/DL4NLP/data/llama/test-follow-up.pt')

# Print the shape of the tensor
print(tensor_follow_up.shape)

# Make new tensor of shape (tensor_follow-up.shape[0], 256, 2)
tensor_follow_up_new = torch.zeros(tensor_follow_up.shape[0], 2, 64)

max_val, idx = torch.max(tensor_follow_up, dim=-1) # Shape of idx is (tensor_ans.shape[0], 256)

tensor_follow_up_new[:, 0, :] = idx
tensor_follow_up_new[:, 1, :] = max_val

print(f"Max val shape: {max_val.shape}")
print(f"Idx shape: {idx.shape}")

# Save tensor
torch.save(tensor_follow_up_new, '/home/scur1744/DL4NLP/data/llama/tensors/test-follow-up.pt')

import torch

# Load torch tensor
tensor_ans = torch.load('/home/scur1744/DL4NLP/data/llama/valid-answer.pt')

# Print the shape of the tensor
print(tensor_ans.shape)

# Make new tensor of shape (tensor_ans.shape[0], 256, 2)
tensor_ans_new = torch.zeros(tensor_ans.shape[0], 2, 256)

max_val, idx = torch.max(tensor_ans, dim=-1) # Shape of idx is (tensor_ans.shape[0], 256)
print(f"Max val shape: {max_val.shape}")
print(f"Idx shape: {idx.shape}")

tensor_ans_new[:, 0, :] = idx
tensor_ans_new[:, 1, :] = max_val

# Save tensor
torch.save(tensor_ans_new, '/home/scur1744/DL4NLP/data/llama/tensors/valid-answer.pt')

tensor_follow_up = torch.load('/home/scur1744/DL4NLP/data/llama/valid-follow-up.pt')

# Print the shape of the tensor
print(tensor_follow_up.shape)

# Make new tensor of shape (tensor_follow-up.shape[0], 256, 2)
tensor_follow_up_new = torch.zeros(tensor_follow_up.shape[0], 2, 64)

max_val, idx = torch.max(tensor_follow_up, dim=-1) # Shape of idx is (tensor_ans.shape[0], 256)

tensor_follow_up_new[:, 0, :] = idx
tensor_follow_up_new[:, 1, :] = max_val
print(f"Max val shape: {max_val.shape}")
print(f"Idx shape: {idx.shape}")

# Save tensor
torch.save(tensor_follow_up_new, '/home/scur1744/DL4NLP/data/llama/tensors/valid-follow-up.pt')
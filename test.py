import torch

A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = torch.triu(A, diagonal=1)
print(B)

# tensor([[0, 2, 3],
#         [0, 0, 6],
#         [0, 0, 0]])
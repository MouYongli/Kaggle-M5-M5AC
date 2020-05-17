# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# torch.manual_seed(1)
#
# batch_size = 4
# seq_length = 5
# in_dim = 10
# embedding_dim = 15
# hidden_dim = 20
# n_layer = 2
# output_dim = 3
# batch_first = True

# x = torch.randn(batch_size, seq_length, embedding_dim)
# h0 = torch.randn(n_layer, batch_size, hidden_dim)
# c0 = torch.randn(n_layer, batch_size, hidden_dim)
# t = torch.randn(batch_size, seq_length, output_dim)
#
# model = nn.LSTM(embedding_dim, hidden_dim, n_layer, batch_first=batch_first)
# y, (hn, cn) = model(x, (h0, c0))
# loss = nn.MSELoss()
# print(y.shape)
# print(hn.shape)
# print(cn.shape)

# embedding = nn.Embedding(10, 3)
# x = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# # shape of x: (batch_size, seq_length)
# print(x.shape)
# y = embedding(x)
# # shape of output of embedding: (batch_size, seq_length, embedding_dim)
# print(y.shape)

# x = torch.randn(batch_size, seq_length, 4).float()
# model = nn.Linear(4, embedding_dim)
# y = model(x)
# print(y.shape)

# y = Variable(torch.ones((3,4,1)))
# t = Variable(torch.ones((3,4,1))*2)
# f = nn.MSELoss(reduction="none")
# loss = f(y, t).data.mean(axis=1).view(3).numpy()
# print(loss)
# pid = ('HOBBIES_1_418', 'FOODS_2_357', 'HOUSEHOLD_1_335')
# for (p, l) in zip(pid, loss):
#     print(p, l)
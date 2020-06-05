import torch
import torch.nn as nn

class M5Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layer, output_dim=1, batch_first=True):
        super(M5Net, self).__init__()
        self.embedding_dim =embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.output_dim = output_dim
        self.batch_first = batch_first
        self.item_embedding = nn.Embedding(3049, embedding_dim)
        self.dept_embedding = nn.Embedding(7, embedding_dim)
        self.cat_embedding = nn.Embedding(3, embedding_dim)
        self.store_embedding = nn.Embedding(10, embedding_dim)
        self.state_embedding = nn.Embedding(3, embedding_dim)
        self.event_name_embedding = nn.Embedding(31, embedding_dim)
        self.event_type_embedding = nn.Embedding(5, embedding_dim)
        self.calendar_prices_embedding = nn.Linear(5, embedding_dim)
        self.encode = nn.Linear(4*embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layer, batch_first=batch_first)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, products, events, calendar_prices):
        item = self.item_embedding(products[:,:,0])
        dept = self.dept_embedding(products[:,:,1])
        cat = self.cat_embedding(products[:,:,2])
        store = self.store_embedding(products[:,:,3])
        state = self.state_embedding(products[:,:,4])
        en1 = self.event_name_embedding(events[:,:,0])
        et1 = self.event_type_embedding(events[:,:,1])
        en2 = self.event_name_embedding(events[:,:,2])
        et2 = self.event_type_embedding(events[:,:,3])
        i = item + dept + cat
        s = store + state
        e = en1 + en2 + et1 + et2
        c = self.calendar_prices_embedding(calendar_prices)
        x = self.encode(torch.cat((i, s, e, c), 2))
        y, (_, _) = self.lstm(x)
        out = self.classifier(y)
        return out

if __name__ == "__main__":
    import torch
    embedding_dim = 10
    hidden_dim = 15
    n_layer = 2
    output_dim = 1
    seq_length = 5
    batch_size = 4
    model = M5Net(embedding_dim, hidden_dim, n_layer, output_dim)
    products = torch.randint(3, (batch_size, seq_length, 5)).long()
    events = torch.randint(5, (batch_size, seq_length, 4)).long()
    calendar_prices = torch.randn(batch_size, seq_length, 5).float()
    h0 = torch.randn(n_layer, batch_size, hidden_dim)
    c0 = torch.randn(n_layer, batch_size, hidden_dim)
    y = model(products, events, calendar_prices)
    print(y.shape)


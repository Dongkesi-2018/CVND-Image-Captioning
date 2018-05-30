import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # trainable hidden state. refer to this paper https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        self.hidden_state = self.init_gru_hidden(batch_size, device)

        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, vocab_size)


    def init_lstm_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0

    def init_gru_hidden(self, batch_size, device):
        h0 = torch.nn.Parameter(torch.randn(self.num_layers, batch_size, self.hidden_size))
        return h0

    def forward(self, features, captions):
        batch_size, seq_len = captions.size()
        # [batch_size, seq_len, embed_size]
        word_embedding = self.embeddings(captions[:, :-1])
        # [batch_size, 1, embed_size]
        features = features.unsqueeze(1)
        # [batch_size, 1+seq_len, embed_size]
        inputs = torch.cat([features, word_embedding], 1)

        outputs, hidden = self.rnn(inputs, self.hidden_state)
        outputs = self.fc(outputs)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_ids = []

        for i in range(max_len):
            outputs, states = self.rnn(inputs, states)
            outputs = self.fc(outputs)                    # batch_size x 1 x vocab_size
            predict = torch.argmax(outputs, dim=2)            # batch_size x 1
            sample_ids.append(predict)
            inputs = self.embeddings(predict)                # batch_size x 1 x embed_size

        sample_ids = torch.cat(sample_ids, 1).squeeze()
        sample_ids = sample_ids.cpu().numpy()                 # batch_size x max_len

        return sample_ids.tolist()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    import numpy as np
    batch_size = 3
    hidden_size = 128
    embed_size = 256
    vocab_size = 9550
    seq_len = 20
    num_layers = 2

    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, batch_size, num_layers=num_layers)
    print(decoder)

    features = torch.randn(batch_size, embed_size)
    captions = torch.zeros(batch_size, seq_len).long()

    print("Captions--size:")
    print(features.size(), captions.size())

    lengths = [1, 2, 3, 2, 3, 2, 3, 20, 6, 4]

    outputs = decoder(features, captions)

    print('type(outputs):', type(outputs))
    print('outputs.shape:', outputs.shape)
    print(batch_size, captions.shape[1], vocab_size)

    print("decoder parameter start--------")
    for name, param in decoder.named_parameters():
        print (name, param.size(), param.requires_grad)

    print(decoder.count_parameters())
    print("decoder parameter end----------")

    features = features.unsqueeze(1)
    pred = decoder.sample(features)
    print(np.array(pred))
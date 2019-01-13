import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # resnet = models.resnet50(pretrained=True)
        resnet = models.resnet101(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batch_norm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # initialize the hidden state (see code below)
        self.hidden = self.init_hidden()

        self.linear = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, 10, self.hidden_dim),
                torch.zeros(1, 10, self.hidden_dim))
    
    def forward(self, features, captions):
        # create embedded word vectors for each word in a captions
        cap_embedding = self.word_embeddings(captions[:, :-1])

        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        # lstm_out, self.hidden = self.lstm(
        #     cap_embedding.view(len(captions), 1, -1), self.hidden)

        # LSTM 返回的第一个值是序列中的所有隐藏状态。
        # 第二个是最近的隐藏状态

        # 添加额外的第二个维度
        inputs = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        lstm_out, self.hidden = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens)
            _, predicted = outputs.max(2)
            sampled_ids.append(predicted.item())
            inputs = self.word_embeddings(predicted)

        return sampled_ids

        print('\n')
        print('Predicted tags: \n', predicted_sentence)
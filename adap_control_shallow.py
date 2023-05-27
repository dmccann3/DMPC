import torch
import torch.nn as nn
import torch.nn.init as init


torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, last_hidden_size, num_classes):
        super(DNN, self).__init__()
        # ADD batchnorm       
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, last_hidden_size)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(last_hidden_size, num_classes)

        # self.skip1 = nn.Linear(input_size, hidden_size)
        # self.skip2 = nn.Linear(hidden_size, last_hidden_size)

        # initialize weights
        init.xavier_normal_(self.fc1.weight)
        self.fc1.bias.data = torch.randn(hidden_size)
        init.xavier_normal_(self.fc2.weight)
        self.fc2.bias.data = torch.randn(hidden_size)
        init.xavier_normal_(self.fc3.weight)
        self.fc3.bias.data = torch.randn(last_hidden_size)
        self.fc4.weight.data.fill_(0)
        self.fc4.bias.data.fill_(0)

        # freeze outer layer from learning
        self.fc4.requires_grad_(False)

    def forward(self, x):
        # skip1 = self.skip1(x)
        # skip2 = self.skip2(self.relu2(skip1))

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out) 
        out = self.tanh3(out) 
        sigma = out
        out = self.fc4(out) 
        return out, sigma

    def adapt_outer_layer(self, K_a):
        K_a = torch.from_numpy(K_a)
        self.fc4.weight.data = K_a
        bias_ = torch.tensor([K_a[0, 0], K_a[0, 1], K_a[0, 2], K_a[0, 3]])
        bias = torch.nn.Parameter(bias_)
        self.fc4.bias = bias
        

def train_layers(model, optimizer, criterion, inputs, labels):
    for i in range(inputs.size(0)):
        model.train()
        optimizer.zero_grad()
        output, _ = model(inputs[i,:])
        loss = criterion(output, labels[i,:])
        loss.backward()
        optimizer.step()
    return output, loss
from abc import ABC
from torch import nn, optim


class ServerSegment(nn.Module, ABC):
    def __init__(self, hidden_sizes):
        """
        Args
        :param hidden_sizes: given the hidden size vector
        when initializing the server's model segment
        """
        super(ServerSegment, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            # nn.ReLU(),
            # nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.party_output = []

    def zero_grads(self):
        self.optimizer.zero_grad()

    def forward(self, party_cut_layer):
        """
        overload the forward function, by accepting parties'
        cut layer aggregated output and returning the prediction
        :param party_cut_layer: the aggregated output from parties' cut layer
        :return: the final model prediction
        """
        self.party_output = party_cut_layer.detach().requires_grad_()
        pred = self.model(self.party_output)
        return pred

    def backward(self):
        """
        backward computation until reaching party's output (i.e., the cut layer)
        :return: the gradients of the cut layer, to be returned to parties
        """
        server_grad = self.party_output.grad
        return server_grad

    def step(self):
        self.optimizer.step()


class PartySegment(nn.Module, ABC):
    def __init__(self, input_size, hidden_sizes):
        """
        Args
        :param input_size: party's input size (i.e., number of held features)
        :param hidden_sizes: given the hidden size vector
        when initializing the party's model segment
        """
        super(PartySegment, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.local_output = []

    def zero_grads(self):
        self.optimizer.zero_grad()

    def forward(self, x):
        """
        forward computation
        :param x: party's local input
        :return: party's output on the cut layer
        """
        self.local_output = self.model(x)
        return self.local_output

    def backward(self, server_grad):
        """
        backward computation
        :param server_grad: received gradients from the server
        :return:
        """
        return self.local_output.backward(server_grad)

    def step(self):
        self.optimizer.step()

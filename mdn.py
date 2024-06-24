import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as normal


class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, inp):
        inp = F.tanh(self.fc1(inp))
        inp = F.sigmoid(self.fc2(inp))
        return inp

    def predict(self, inp):
        with torch.no_grad():
            inp = self.forward(inp)
        return inp


class MDN(nn.Module):
    def __init__(self) -> None:
        super(MDN, self).__init__()
        self.in_features = 1
        self.num_gaussians = 3

        self.fc1 = nn.Linear(self.in_features, 20)
        self.pi = nn.Linear(20, self.num_gaussians)
        self.sigma = nn.Linear(20, self.num_gaussians)
        self.mu = nn.Linear(20, self.num_gaussians)

    def forward(self, inp):
        inp = F.tanh(self.fc1(inp))

        pis = F.softmax(self.pi(inp), dim=1)
        sigmas = torch.exp(self.sigma(inp))
        mus = self.mu(inp)

        out = torch.cat((pis, sigmas, mus), dim=1)
        return out

    def predict(self, inp):
        with torch.no_grad():
            out = self.forward(inp)
        return out


def _calc_simple_nn_loss(out, target):
    loss = torch.sum((target.view(-1) - out.view(-1)) ** 2) / target.shape[0]
    return loss


def _calc_mdn_loss(mdn, inp, target):
    out = mdn.forward(inp.view(-1, 1))

    prob = 0  # not actually probability though
    for i in range(3):
        pi, sigma, mu = out[:, i], out[:, i+3], out[:, i+6]
        gaussian = normal.Normal(mu, sigma)
        phi = torch.exp(gaussian.log_prob(target.view(-1)))
        prob += pi * phi

    loss = torch.sum(- torch.log(prob)) / prob.size()[0]

    return loss


def calc_loss(net, inp, out, target):
    if isinstance(net, SimpleNN):
        loss = _calc_simple_nn_loss(out, target)
    elif isinstance(net, MDN):
        loss = _calc_mdn_loss(net, inp, target)
    else:
        raise NotImplementedError

    return loss


def train(net, num_epochs, batch_size, optimizer, inp, target):
    # shuffling is importtant
    perm_idx = torch.randperm(len(target))
    inp = inp[perm_idx]
    target = target[perm_idx]

    n = len(inp)
    hist_loss = []

    for _ in range(num_epochs):

        for i in range(n // batch_size + 1):
            batch_inp = inp[i * batch_size: (i + 1) * batch_size].view(-1, 1)
            batch_target = target[i * batch_size: (i + 1) * batch_size]

            optimizer.zero_grad()
            batch_out = net.forward(batch_inp)
            loss = calc_loss(net, batch_inp, batch_out, batch_target)
            loss.backward()
            optimizer.step()

        hist_loss.append(
            calc_loss(
                net,
                inp,
                net.forward(inp.view(-1, 1)),
                target).detach().numpy()
        )

    return net, hist_loss

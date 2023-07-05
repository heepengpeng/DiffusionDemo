import time

import torch
from sklearn.datasets import make_s_curve

s_curve, _ = make_s_curve(10 ** 4, noise=0.1)
s_curve = s_curve[:, [0, 2]] / 10.0
data = s_curve.T

dataset = torch.Tensor(s_curve).float()

num_steps = 100

betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

import torch.nn as nn


class MLPDiffusion(nn.Module):

    def __init__(self, n_steps, num_uints=128):
        super(MLPDiffusion, self).__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_uints),
                nn.ReLU(),
                nn.Linear(num_uints, num_uints),
                nn.ReLU(),
                nn.Linear(num_uints, num_uints),
                nn.ReLU(),
                nn.Linear(num_uints, 2),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_uints),
                nn.Embedding(n_steps, num_uints),
                nn.Embedding(n_steps, num_uints),
            ]
        )

    def forward(self, x_0, t):
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)
        return x


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]

    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)

    a = alphas_bar_sqrt[t]

    am1 = one_minus_alphas_bar_sqrt[t]

    e = torch.randn_like(x_0)

    x = x_0 * a + e * am1
    x = x.to("cuda:0")
    output = model(x, t.squeeze(-1).to("cuda:0"))
    return (e - output).square().mean()


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)


def train(model):
    batch_size = 128
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoch = 4000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            if (t+1) % 100 == 0:
                print(f"{t+1} steps, loss:{loss}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()


def sample_100(model):
    for i in range(100):
        x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)


if __name__ == '__main__':
    model = MLPDiffusion(num_steps).to("cuda:0")

    print("start to train")
    start_time = time.time()
    train(model)
    end_time = time.time()
    print(f" train time consume: {end_time - start_time}")

    print("start to sample")
    start_time = time.time()
    sample_100(model)
    end_time = time.time()
    print(f" sample 100 time consume: {end_time - start_time}")

import time
import itertools
from itertools import chain
import numpy as np
import torch
from torch import nn, tensor
from util import OneHot
import matplotlib.pyplot as plt



#Define generator and discriminator network
class Generator(nn.Module):
    def __init__(self, batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, device):
        super().__init__()
        self.batch_size = batch_size
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
        self.device = device
        
        self.layer1 = nn.Sequential(nn.Linear(dim_z + dim_y, dim_W1, bias=False),
                                    nn.BatchNorm1d(dim_W1, eps=1e-8),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(dim_W1 + dim_y, dim_W2*6*6, bias=False),
                                    nn.BatchNorm1d(dim_W2*6*6, eps=1e-8),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(in_channels=dim_W2 + dim_y, out_channels=dim_W3, kernel_size=5, stride=2, padding=2, output_padding=1),
                                    nn.BatchNorm2d(dim_W3, eps=1e-8),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(in_channels=dim_W3 + dim_y, out_channels=dim_channel, kernel_size=5, stride=2, padding=2, output_padding=1),
                                    nn.BatchNorm2d(dim_channel, eps=1e-8))

    def forward(self, z, y):
        yb = torch.reshape(y, (self.batch_size, self.dim_y, 1, 1))
        z = torch.cat((z, y), 1)
        h1 = self.layer1(z)
        h1 = torch.cat((h1, y), 1)
        h2 = self.layer2(h1)
        h2 = torch.reshape(h2, (self.batch_size, self.dim_W2, 6, 6))
        h2 = torch.cat((h2, torch.mul(yb, torch.ones((self.batch_size, self.dim_y, 6, 6), dtype=torch.float32, device=self.device))), 1)
        n = torch.mul(yb, torch.ones((self.batch_size, self.dim_y, 6, 6), dtype=torch.float32, device=self.device))
        h3 = self.layer3(h2)
        h3 = torch.cat((h3, torch.mul(yb, torch.ones((self.batch_size, self.dim_y, 12, 12), dtype=torch.float32, device=self.device))), 1)
        h4 = self.layer4(h3)

        return h4


class Discriminator(nn.Module):
    def __init__(self, batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, device):
        super().__init__()
        self.batch_size = batch_size
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
        self.device = device
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=dim_channel + dim_y, out_channels=dim_W3, kernel_size=5, stride=2, padding=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=dim_W3 + dim_y, out_channels=dim_W2, kernel_size=5, stride=2, padding=2),
                                    nn.BatchNorm2d(dim_W2, eps=1e-8),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(dim_W2*6*6 + dim_y, dim_W1, bias=False),
                                    nn.BatchNorm1d(dim_W1, eps=1e-8),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, image, y):
        yb = torch.reshape(y, (self.batch_size, self.dim_y, 1, 1))
        x = torch.cat((image, torch.mul(yb, torch.ones((self.batch_size, self.dim_y, 24, 24), dtype=torch.float32, device=self.device))), 1)
        h1 = self.layer1(x)
        h1 = torch.cat((h1, torch.mul(yb, torch.ones((self.batch_size, self.dim_y, 12, 12), dtype=torch.float32, device=self.device))), 1)
        h2 = self.layer2(h1)
        h2 = torch.reshape(h2, (self.batch_size, -1))
        h2 = torch.cat((h2, y), 1)
        h3 = self.layer3(h2)

        return h3

#Define loss functions
def generator_cost(raw_gen2):
    return -torch.mean(raw_gen2)

def discriminator_cost(raw_real2, raw_gen2):
    return torch.sum(raw_gen2) - torch.sum(raw_real2)

#Define GAN architecture
class GAN(nn.Module):
    def __init__(self, epochs=10, batch_size=32, image_shape=[1, 24, 24], dim_y=6, dim_z=100, dim_W1=1024, dim_W2=128, dim_W3=64, dim_channel=1, learning_rate=1e-4, device='cpu'):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
        self.learning_rate = learning_rate
        self.device = device
        self.normal = (0, 0.1)

        self.generator = Generator(batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, device=device)
        self.discriminator = Discriminator(batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, device=device)

        self.optimizer_g = torch.optim.RMSprop(list(self.generator.parameters()), lr=learning_rate)
        self.optimizer_d = torch.optim.RMSprop(list(self.discriminator.parameters()), lr=learning_rate)

        #Initialization of weights
        for weights in chain(self.generator.parameters(), self.discriminator.parameters()):
            torch.nn.init.normal_(weights, mean=0.0, std=0.02)

        self.fitting_time = None

    
    def fit(self, x, y):
        self.fitting_time = time.time()
        iterations = 0
        #Control balance of training discriminator vs generator
        k = 4

        gen_loss_all = []
        p_real = []
        p_fake = []
        p_distri = []
        discrim_loss = []

        for epoch in range(self.epochs):
            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('Epoch:', epoch + 1)
                
            index = np.arange(len(y))
            np.random.shuffle(index)
            x = x[index]
            y = y[index]
            y2 = OneHot(y, n=self.dim_y)

            for start, end in zip(range(0, len(y), self.batch_size), range(self.batch_size, len(y), self.batch_size)):
                xs = x [start:end].reshape([-1, 1, 24, 24])
                ys = y2[start:end]

                zs = np.random.normal(self.normal[0], self.normal[1], size=(self.batch_size, self.dim_z)).astype(np.float32)

                xs = tensor(xs, dtype=torch.float32, device=self.device)
                ys = tensor(ys, dtype=torch.float32, device=self.device)
                zs = tensor(zs, dtype=torch.float32, device=self.device)

                self.train()
                if iterations % k == 0:
                    self.optimizer_g.zero_grad()

                    h4 = self.generator(zs, ys)
                    image_gen = nn.Sigmoid()(h4)
                    raw_gen2 = self.discriminator(image_gen, ys)
                    p_gen_val = torch.mean(raw_gen2)

                    gen_loss_val = generator_cost(raw_gen2)

                    raw_real2 = self.discriminator(xs, ys)
                    p_real_val = torch.mean(raw_real2)

                    discrim_loss_val = discriminator_cost(raw_real2, raw_gen2)

                    gen_loss_val.backward()
                    self.optimizer_g.step()
                else:
                    self.optimizer_d.zero_grad()

                    h4 = self.generator(zs, ys)
                    image_gen = nn.Sigmoid()(h4)
                    raw_gen2 = self.discriminator(image_gen, ys)
                    p_gen_val = torch.mean(raw_gen2)

                    gen_loss_val = generator_cost(raw_gen2)

                    raw_real2 = self.discriminator(xs, ys)
                    p_real_val = torch.mean(raw_real2)

                    discrim_loss_val = discriminator_cost(raw_real2, raw_gen2)

                    discrim_loss_val.backward()
                    self.optimizer_d.step()

                p_real.append(p_real_val.cpu().item())
                p_fake.append(p_gen_val.cpu().item())
                discrim_loss.append(discrim_loss_val.cpu().item())

                if iterations % 1000 == 0:
                    print('Iterations',
                          iterations,
                          '| Average P(real):', f'{p_real_val.cpu().item():12.9f}',
                          '| Average P(fake):', f'{p_gen_val.cpu().item():12.9f}',
                          '| Discriminator loss:', f'{discrim_loss_val.cpu().item():12.9f}')

                    # self.eval()
                    # with torch.no_grad():
                    #     y_np_sample = OneHot(np.random.randint(events_num, size=[self.batch_size]), n=events_num)
                    #     z_np_sample = np.random.normal(self.normal[0], self.normal[1], size=[self.batch_size, self.dim_z]).astype(np.float32)

                    #     y_np_sample = tensor(y_np_sample, dtype=torch.float32, device=device)
                    #     z_np_sample = tensor(z_np_sample, dtype=torch.float32, device=device)

                    #     generated_samples = nn.Sigmoid()(self.generator(z_np_sample, y_np_sample))

                    #     generated_samples = generated_samples.reshape([-1, 576])
                    
                    #     generated_samples = generated_samples * max_value
                    
                        # with open(data_saving_path + '%s.csv' %iterations, 'w') as csvfile:
                        #     # spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        #     writer = csv.writer(csvfile)
                        #     writer.writerows(generated_samples.cpu().detach().numpy())
                        #     csvfile.close()

                iterations += 1

        self.fitting_time = np.round(time.time() - self.fitting_time, 3)
        print('\nElapsed Training Time: ' + time.strftime('%Hh %Mmin %Ss', time.gmtime(self.fitting_time)))
      
        # print('P_real', p_real)
        # print('P_fake', p_fake)

        #Plotting
        fig, ax = plt.subplots()
        ax.plot(p_real, label='real')
        ax.plot(p_fake, label='fake')
        ax.legend()
        ax.set_xlim(0, len(p_real))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Wasserstein Distance')
        ax.grid(True)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(discrim_loss)
        ax.set_xlim(0, len(discrim_loss))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discriminator Loss')
        ax.grid(True)
        fig.show()

    def predict(self):
        self.eval()
        with torch.no_grad():
            y_np_sample = OneHot(np.random.randint(self.dim_y, size=[self.batch_size]), n=self.dim_y)
            zs = np.random.normal(self.normal[0], self.normal[1], size=[self.batch_size, self.dim_z]).astype(np.float32)

            y_np_sample = tensor(y_np_sample, dtype=torch.int8, device=self.device)
            zs = tensor(zs, dtype=torch.float32, device=self.device)

            generated_samples = nn.Sigmoid()(self.generator(zs, y_np_sample))

        #Image shape 24x24 = 576
        generated_samples = generated_samples.reshape([-1, 576])

        return generated_samples.cpu().detach().numpy(), y_np_sample.cpu().detach().numpy()

        # with open(f'{path_data}' if path_data.endswith('csv') else f'{path_data}.csv', 'w') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(generated_samples.cpu().detach().numpy())
        #     csvfile.close()

        # with open(f'{path_labels}' if path_labels.endswith('csv') else f'{path_labels}.csv', 'w') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(y_np_sample.cpu().detach().numpy())
        #     csvfile.close()
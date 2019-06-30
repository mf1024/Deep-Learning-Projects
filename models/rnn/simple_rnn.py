import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data

# New class. Replace LSTM with our own CELL.
#

TIMESTEPS_INPUT = 16
TIMESTEPS_EXT = 12
TIMESTEPS_PREDICT = 4

DATASET_WINDOW_OVERLAP = 1
DATASET_LEN = TIMESTEPS_INPUT * int(1e2)
DATASET_X_DELTA = 1e-1
DATASET_NOISE_PERCENT = 0

HIDDEN_SIZE = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = 'cpu'
NUM_EPOCHS = 10000
RNN_LAYERS = 1

y_data = []
for x in range(DATASET_LEN):
    y = np.sin(x * DATASET_X_DELTA)
    y_data.append(y)

y_data = np.array(y_data)
#plt.plot(np.arange(0, TIMESTEPS_INPUT), y_data[:TIMESTEPS_INPUT])
#plt.show()


class DatasetTimeseries(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.data_x = []
        self.data_y = []
        for idx in range(0, len(y_data) - 1 - TIMESTEPS_INPUT - TIMESTEPS_PREDICT, DATASET_WINDOW_OVERLAP):
            x_slice = torch.FloatTensor(y_data[idx:idx + TIMESTEPS_INPUT])
            x_slice = x_slice.unsqueeze(dim=1)

            x_noise = torch.zeros_like(x_slice).uniform_()
            x_slice = (1.0 - DATASET_NOISE_PERCENT) * x_slice + DATASET_NOISE_PERCENT * x_noise
            self.data_x.append(x_slice)

            y_slice = torch.FloatTensor(y_data[idx + 1:idx + 1 + TIMESTEPS_INPUT + TIMESTEPS_PREDICT])
            y_slice = y_slice.unsqueeze(dim=1)
            self.data_y.append(y_slice)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


dataset = DatasetTimeseries()
print(f'datapoints: {len(dataset)}')
data_loader = torch.utils.data.DataLoader(
    dataset,
    BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, batch_first):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.fc0 = torch.nn.Linear(in_features=1, out_features=self.hidden_size)
        self.fc = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

        # self.fc = []
        # for i in range(self.hidde_size):
        #     self.fc.append(torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))

    def forward(self, x, hidden): #

        # x = [batches, time_steps, input_size]
        # y = [batches, time_steps, input_size]

        y_list = []

        for i in range(x.shape[1]):
            inp = x[:,i,:]

            hidden = torch.tanh(self.fc0.forward(inp) + self.fc.forward(hidden))
            y_list.append(hidden)

        y = torch.cat(y_list).permute(1,0,2)

        return y, hidden


class ModelPytorch(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = VanillaRNN(input_size=1, num_layers=RNN_LAYERS, hidden_size=HIDDEN_SIZE, batch_first=True)


        self.decoder = torch.nn.Sequential(
            #torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=1)
        )

        self.hidden = None
        self.reset_hidden(BATCH_SIZE)



    def reset_hidden(self, batch_size):
        self.hidden = torch.zeros(( RNN_LAYERS, batch_size, HIDDEN_SIZE))

    def forward(self, input):

        # many to many
        out, self.hidden = self.rnn.forward(input, (self.hidden))
        y = self.decoder.forward(out)

        if self.training:

            last_output = y[:,-1:]

            for _ in range(TIMESTEPS_PREDICT):

                out_h, self.hidden = self.rnn.forward(last_output, (self.hidden))
                #print(torch.sum(self.hidden))

                last_output = self.decoder(out_h)
                y = torch.cat([y, last_output], dim = 1)

                #print(y.shape)

                # out, _ = self.rnn.forward(y[:, -TIMESTEPS_EXT:], (self.hidden))
                # y_last = self.decoder.forward(out)
                # y = torch.cat([y, y_last[:,-1:]], dim=1) #(batch, timesteps, features)

        return y

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# print(count_parameters(model))
#
#
# def layers_debug(optim):
#     layer_count = 0
#     for var_name in optim.state_dict():
#         shape = optim.state_dict()[var_name].shape
#         if len(optim.state_dict()[var_name].shape)>1:
#             layer_count += 1
#
#         print(f"{var_name}\t\t{optim.state_dict()[var_name].shape}")
#     print(layer_count)
#
#
# layers_debug(model)

model = ModelPytorch().to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, NUM_EPOCHS+1):

    losses = []
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        # np_x = batch_x.data.numpy()
        # np_y = batch_y.data.numpy()
        # np_y = np_y[:, TIMESTEPS_INPUT:]
        # for idx in range(batch_x.size(0)):
        #     plt.plot(np.arange(0, TIMESTEPS_INPUT), np_x[idx])
        #     plt.plot(np.arange(TIMESTEPS_INPUT, TIMESTEPS_INPUT + len(np_y[idx])), np_y[idx])
        #     plt.show()

        model.reset_hidden(batch_x.size(0))
        batch_y_prim = model.forward(batch_x)

        # if epoch % 10 == 0:
        #     plt.plot(batch_y_prim[0].squeeze().detach().numpy())
        #     plt.show()


        loss = torch.mean((batch_y_prim - batch_y)**2)
       # print(f"batch_pred {batch_y_prim}")
       # print(f"batch_y {batch_y}")
        #loss = torch.sum(torch.abs(batch_y_prim - batch_y))
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'epoch: {epoch} loss: {np.average(losses)}')

    # Simulation

    offset = 0
    sim_x = y_data[offset:TIMESTEPS_INPUT + offset]
    sim_out = sim_x.tolist()
    sim_x = torch.FloatTensor(sim_x)
    sim_x = sim_x.unsqueeze(dim=1)

    model = model.eval()
    with torch.no_grad():

        # inital memory

        if epoch % 1000 == 0:

            model.reset_hidden(batch_size=1)
            sim_x = sim_x.unsqueeze(dim=0)
            sim_y = model.forward(sim_x) # Input is TIMESTEPS_INPUT

            for idx_rollout in range(1000):

                sim_y_prim = model.forward(sim_y[:,-TIMESTEPS_EXT:])
                sim_y = torch.cat([sim_y, sim_y_prim[:,-1:]], dim=1)

                sim_y_last_timestep = sim_y[:,-1] # last timestep
                sim_out.append(sim_y_last_timestep[0].item())

            plt.title(f'epoch: {epoch}')
            plt.plot(np.arange(0, TIMESTEPS_INPUT), sim_out[:TIMESTEPS_INPUT])
            plt.plot(np.arange(TIMESTEPS_INPUT, len(sim_out)), sim_out[TIMESTEPS_INPUT:])
            plt.show()

    model = model.train()



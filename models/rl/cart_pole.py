import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

#Model definition
class RL_Model(nn.Module):
    def __init__(self, input_state_size, action_number):

        super(RL_Model, self).__init__()
        self.hidden_layer = nn.Linear(in_features=input_state_size, out_features=20)
        self.output_layer = nn.Linear(in_features=20, out_features=action_number)

    def forward(self, states_input):

        x = self.hidden_layer(states_input)
        x = F.relu(x)
        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x

def prep_input(state_tuple):
    return torch.Tensor(state_tuple).reshape(1,-1)

#Model training
num_episodes = 10000
max_steps_per_ep = 1000
LEARNING_RATE = 1e-3
env = gym.make('CartPole-v0')
env._max_episode_steps = 1000
model = RL_Model(4, 2)

test_period = 100
test_period_steps = 0

optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

for ep in range(num_episodes):

    s = env.reset()

    chosen_action_prob_seq = []
    reward_seq = []

    ep_steps = 0

    for step in range(max_steps_per_ep):

        if ep > 0 and ep % test_period == 0:
            env.render()

        a_dist = model.forward(prep_input(s))
        a_dist_np = a_dist.detach().to('cpu').numpy()
        chosen_action_idx = np.random.choice(range(a_dist_np.size),p=a_dist_np[0])

        chosen_action_prob_seq.append(a_dist[0][chosen_action_idx])

        s1, r, done, _ = env.step(chosen_action_idx)

        if done:
            reward_seq.append(0)
            break

        reward_seq.append(r)
        s = s1

        ep_steps+=1

    test_period_steps += ep_steps

    culm_reward = 0
    reward_discount = 0.998
    for i in reversed(range(len(reward_seq ))):
        culm_reward = culm_reward * reward_discount + reward_seq[i]
        reward_seq[i] = culm_reward

    if (ep % test_period == 0):
        avg_steps_per_test_period = test_period_steps/test_period
        print(f"Average number of steps in the last {test_period} episods is {avg_steps_per_test_period}")
        test_period_steps = 0

    reward = torch.Tensor(reward_seq).view(1,-1)
    chosen_action_prob = torch.stack(chosen_action_prob_seq).reshape(1,-1)

    loss = -torch.mean(torch.log(chosen_action_prob) * reward)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
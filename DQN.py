import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import gym
env = gym.make('CartPole-v1')
env = env.unwrapped
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
EPISODES = 10000
N_STATES = env.observation_space.shape[0]
N_ACTIONS =  env.action_space.n
MEMORY_CAPACITY = 2000
LR = 0.01
EPSILON = 0.1
Q_NETWORK_ITERATION =100
GAMMA = 0.9
BATCH_SIZE = 32
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128 )
        self.out = nn.Linear(128, N_ACTIONS)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.out(x)
        return x 
class Dqn():
    def __init__(self) :
        self.eval_net = Net().to(device)
        self.target_net = Net().to(device)
        self.learn_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        self.fig,self.ax = plt.subplots()
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        if np.random.randn() <= EPSILON:
            action = np.random.randint(0, N_ACTIONS)
        else:
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        return action
    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    def learn(self):
        # 100counter后更新target_net的参数
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_action = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)).to(device)
        b_reward = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]).to(device)
        b_next_state = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)
        
        q_eval=self.eval_net(b_state).gather(1,b_action)
        q_next=self.target_net(b_next_state).detach()
        q_target=b_reward+GAMMA*q_next.max(1)[0].view(BATCH_SIZE,1)
        
        loss=self.loss(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def plot(self,ax,step_counter_list):
        ax.cla()
        ax.set_xlabel('episode')
        ax.set_ylabel('reward')  
        ax.plot(step_counter_list)
        # plt.show()
        plt.pause(0.0000001)
        
            
if __name__ == '__main__':
    dqn = Dqn()
    step_counter_list = []
    for episode in range(EPISODES):
        state = env.reset()[0]
        step_counter=0
        while True:
            # env.render()
            action = dqn.choose_action(state)
            next_state, reward, done,truncated, _ = env.step(action)
            reward = reward*100 if reward>0 else reward*5
            dqn.store_transition(state, action, reward, next_state)
            
            # 不攒数据了，直接学习
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    step_counter_list.append(step_counter)
                    if episode % 100 == 0:
                        print('episode:', episode, 'step_counter:', step_counter)
                    # dqn.plot(dqn.ax,step_counter_list)
                    break
            state = next_state
            step_counter+=1
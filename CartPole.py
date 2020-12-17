import gym
import matplotlib.pyplot as plt
import random
import torch

BATCH_SIZE = 100
B_APPRENTISSAGE=1000
B_SIZE=100000
EPSILONE = 0.1
TAUX_APPRENTISAGE=0.01

class Buffer:
    def __init__(self):
        self.buffer = []
        self.position = 0

    def add(self, tab):
        if len(self.buffer) < B_SIZE:
            self.buffer.append(tab)
        else:
            self.buffer[self.position] = tab
            self.position = (self.position + 1) % B_SIZE

    def sample(self):
        return random.sample(self.buffer, BATCH_SIZE)

class Rn(torch.nn.Module):
    def __init__(self,nb_entrée,nb_action):
        super(Rn, self).__init__()
        self.l1 = torch.nn.Linear(nb_entrée, 16)
        self.l2 = torch.nn.Linear(16, nb_action)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

class agent:

    def __init__(self,reseau_neurone,nb_action,taille):
        self.buffer=Buffer(taille)
        self.nb_action=nb_action
        self.model = reseau_neurone
        self.loss_func = torch.nn.MSELoss(reduction='sum')
        self.optim = torch.optim.SGD(self.model.parameters(), lr=TAUX_APPRENTISAGE)

    def train(self, nb_ep,buffer):
        for i in range(nb_ep):
            ob = env.reset()
            while True:
                action = agent.act(ob, reward, done)
                ob, reward, done, _ = env.step(action)
                if done:
                    break

    def get_action(self,etat):
        res = self.model(etat)
        if random()>EPSILONE:
            return torch.argmax(res, 1)
        else:
            return random.randint(0, self.nb_action-1)

def train(env,agent,nb_ep):
    for i in range(nb_ep):
        ob = env.reset()
        while True:
            action = agent.get_action(ob)
            ob, reward, done, _ = env.step(action)
            if done:
                break


env = gym.make('CartPole-v1')
tab=[]
for i_episode in range(20):
    observation = env.reset()
    reward_total=0
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        reward_total+=reward
        if done:
            tab.append(reward_total/t+1)
            print("Episode finished after {} timesteps".format(t + 1))
            break


plt.plot(tab)
plt.ylabel('some numbers')
plt.show()
env.close()
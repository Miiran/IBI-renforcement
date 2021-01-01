import gym
import matplotlib.pyplot as plt
import random
import torch
import math

BATCH_SIZE = 100
B_SIZE=100000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
GAMMA = 0.9
NB_EP = 500
UPDATE = 15
TAUX_APPRENTISAGE=0.001
NB_ACTION=2
TAILLE_OB=4

class Buffer:
    def __init__(self):
        self.buffer = []
        self.position = 0
        self.taille = 0

    def add(self, step):
        if self.taille < B_SIZE:
            self.taille += 1
            self.buffer.append(step)
        else:
            self.buffer[self.position % B_SIZE]=step
            self.position = (self.position + 1) % B_SIZE

    def sample(self):
        sample = random.sample(self.buffer, BATCH_SIZE)
        ob = []
        action = []
        ob_next = []
        reward = []
        done = []
        for step in sample:
            ob.append(step[0])
            action.append(step[1])
            ob_next.append(step[2])
            reward.append(step[3])
            done.append(step[4])
        return ob,action,ob_next,reward,done

class Rn(torch.nn.Module):
    def __init__(self,nb_entrée,nb_action):
        super(Rn, self).__init__()
        self.l1 = torch.nn.Linear(nb_entrée, 32)
        self.l2 = torch.nn.Linear(32, 32)
        self.l3 = torch.nn.Linear(32, 32)
        self.l4 = torch.nn.Linear(32, nb_action)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        return x


class Agent:

    def __init__(self,rn1,rn2,nb_action):
        self.steps_done=0
        self.buffer=Buffer()
        self.nb_action=nb_action
        self.model = rn1
        self.modelExpect = rn2
        self.loss_func = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=TAUX_APPRENTISAGE,weight_decay=1e-5)

    def train(self):

        ob,action,ob_next,reward,done = self.buffer.sample()
        ob_batch=torch.FloatTensor(ob)
        ob_next_batch = torch.FloatTensor(ob_next)
        reward = torch.FloatTensor(reward)
        action_batch=torch.LongTensor(action)
        done=[not elem for elem in done]
        done = torch.FloatTensor(done)

        Q_net = self.model(ob_batch).gather(1, action_batch.unsqueeze(1))
        Q_next = self.modelExpect(ob_next_batch)
        Q_expect = reward + GAMMA * Q_next.max(1)[0] * done

        loss = self.loss_func(Q_net, Q_expect.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def add_buffer(self, step):
        self.buffer.add(step)

    def action(self,etat):
        self.steps_done+=1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        res = self.model(etat)
        if random.random()>eps_threshold:
            return torch.argmax(res).item()
        else:
            return random.randint(0, self.nb_action-1)




def train(env,agent,nb_ep):
    tab_reward = []
    tab_ep=[]
    for i in range(nb_ep):
        tab_ep.append(i)
        reward_total = 0
        observation = env.reset()
        done = False
        while not done:
            action = agent.action(torch.FloatTensor(observation))
            observation_next, reward, done, info = env.step(action)
            step=[observation, action, observation_next, reward, done]
            agent.add_buffer(step)
            reward_total += reward
            observation=observation_next
            if agent.buffer.taille > BATCH_SIZE:
                agent.train()
        tab_reward.append(reward_total)
        if i % UPDATE == 0:
            agent.modelExpect.load_state_dict(agent.model.state_dict())
    return tab_reward,tab_ep

def resultat(agent,env):
    for i in range(10):
        ob = env.reset()
        score = 0
        done = False
        while not done:
            env.render()
            action = agent.action(torch.FloatTensor(ob))
            ob, reward, done, _ = env.step(action)
            score += reward

env = gym.make('CartPole-v1')
net=Rn(TAILLE_OB,NB_ACTION)
net2=Rn(TAILLE_OB,NB_ACTION)
agent=Agent(net,net2,NB_ACTION)
taby,tabx=train(env,agent,NB_EP)
plt.scatter(tabx,taby)
plt.show()
resultat(agent,env)
env.close()
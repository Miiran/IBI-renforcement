import gym
import matplotlib.pyplot as plt
import random
import torch
import math
import vizdoomgym
from gym import wrappers
import skimage.color
import skimage.transform


BATCH_SIZE = 5 #taille du buffer
B_SIZE=100000 #epsilone au debut de l'apprentissage
EPS_START = 0.9 #epsilone a la fin de l'apprentissage
EPS_END = 0.05 #taux de diminution d'epsilone
EPS_DECAY = 200
GAMMA = 0.9
NB_EP = 100 #Nombre d'épisode
UPDATE = 2 #Nombre d'épisode avant d'update le reseau de neurone expect
TAUX_APPRENTISAGE=0.001
NB_ACTION=2
SAVE = False #True pour sauvegarder l'entrainement
USE_SAVE = True #True pour ne pas entrainer et l'agent et load le fichier
SAVE_NAME="saveVizdoom" #emplacement de la save




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



def preprocess(img,resolution):
    img = skimage.transform.resize(img, resolution)
    #passage en noir et blanc
    img = skimage.color.rgb2gray(img)
    #passage en format utilisable par pytorch
    img = img.reshape([1, 1, resolution[0], resolution[1]])
    return img



class ConvNet(torch.nn.Module):
    def __init__(self, h, w, outputs):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.layer = torch.nn.Linear(linear_input_size, outputs)
        self.relu=torch.nn.ReLU()


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.layer(x.view(x.size(0), -1))



class Agent:

    def __init__(self,rn1,rn2,nb_action):
        self.steps_done=0
        self.buffer=Buffer()
        self.nb_action=nb_action
        self.model = rn1
        self.modelExpect = rn2
        self.loss_func = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=TAUX_APPRENTISAGE,weight_decay=1e-3)

    def train(self):

        ob, action, ob_next, reward, done = self.buffer.sample()
        ob_batch = torch.cat(ob)
        ob_next_batch = torch.cat(ob_next)
        reward = torch.FloatTensor(reward)
        action_batch = torch.LongTensor(action)
        done = [not elem for elem in done]
        done = torch.BoolTensor(done)

        Q_net = self.model(ob_batch).gather(1, action_batch.unsqueeze(1))
        Q_next = self.modelExpect(ob_next_batch)
        Q_expect = reward + GAMMA * Q_next.max(1)[0] * done

        loss = self.loss_func(Q_net, Q_expect.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def add_buffer(self, step):
        self.buffer.add(step)

    def action(self,etat,result=False):
        self.steps_done += 1
        res = self.model(etat)
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        if random.random()>eps or result:
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
        observation = torch.FloatTensor(preprocess(observation[0], [112, 64]))
        done = False
        while not done:
            action = agent.action(observation)
            observation_next, reward, done, info = env.step(action)
            observation_next = torch.FloatTensor(preprocess(observation_next[0], [112, 64]))
            step=[observation, action, observation_next, reward, done]
            agent.add_buffer(step)
            reward_total += reward
            observation=observation_next
            if agent.buffer.taille > BATCH_SIZE:
                agent.train()
        tab_reward.append(reward_total)
        print("episode : {} et reward : {}".format(i, reward_total))
        if i % UPDATE == 0:
            agent.modelExpect.load_state_dict(agent.model.state_dict())
    if SAVE: torch.save({'model_state_dict': agent.model.state_dict()}, SAVE_NAME)
    return tab_reward,tab_ep



def resultat(agent,env):
    for i in range(5):
        ob = env.reset()
        ob=torch.FloatTensor(preprocess(ob[0], [112, 64]))
        score = 0
        done = False
        i=0
        while not done:
            env.render()
            action = agent.action(ob,True)
            i+=1
            ob, reward, done, _ = env.step(action)
            ob = torch.FloatTensor(preprocess(ob[0], [112, 64]))
            score += reward
        print(score)



env = gym.make('VizdoomBasic-v0', depth=True, labels=True,position=True, health=True)
NB_ACTION=env.action_space.n
net=ConvNet(112,64,NB_ACTION)
net2=ConvNet(112,64,NB_ACTION)
if USE_SAVE:
    load=torch.load("./"+SAVE_NAME)
    net.load_state_dict(load['model_state_dict'])
    agent = Agent(net, net2, NB_ACTION)
    resultat(agent, env)
else:
    agent = Agent(net, net2, NB_ACTION)
    taby,tabx=train(env,agent,NB_EP)
    plt.scatter(tabx,taby)
    plt.show()
    resultat(agent, env)
env.close()
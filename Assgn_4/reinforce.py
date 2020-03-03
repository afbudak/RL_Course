from typing import Iterable
import numpy as np
import torch
from torch import nn

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        n_in = state_dims
        n_h = 32
        n_out = num_actions
        self.model = nn.Sequential(nn.Linear(n_in, n_h),
                                 nn.ReLU(),
                                 nn.Linear(n_h, n_h),
                                 nn.ReLU(),
                                 nn.Linear(n_h, n_out),
                                 nn.Softmax())
        
        self.model = self.model.float()
        #self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> int:
        # TODO: implement this method
        action_probs = self.model(torch.tensor(s).float()).detach().numpy()
        action_space = np.arange(len(action_probs))
        action = np.random.choice(action_space, p=action_probs)
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        self.optimizer.zero_grad()
        #state_tensor = torch.FloatTensor(s)
        action_tensor = torch.LongTensor(a)
        logprob = torch.log(self.model(torch.tensor(s).float()))
        loss = -gamma_t * delta * logprob[a]
        # Calculate gradients
        loss.backward()
        # Apply gradients
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        n_in = state_dims
        n_h = 32
        n_out = 1
        self.model = nn.Sequential(nn.Linear(n_in, n_h),
                                 nn.ReLU(),
                                 nn.Linear(n_h, n_h),
                                 nn.ReLU(),
                                 nn.Linear(n_h, n_out))
        
        self.model = self.model.float()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        # TODO: implement this method
        output = self.model(torch.tensor(s).float()).detach().numpy()
        #return output.tolist()[0]
        return output.tolist()[0]

    def update(self,s,G):
        # TODO: implement this method
        s = torch.tensor(s).float()
        v_head = self(s)
        delta = G-v_head
        loss = -delta*self.model(s)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    G_0 = []
    for eps in range(num_episodes):
        print(eps)
        SAR = []
        #generate an episode using pi
        done = True
        observation = env.reset()
        act = pi(observation)
        observation_prime, reward, done, info = env.step(act)
        SAR.append([observation,act,reward])
        while not done:
            observation = observation_prime
            act = pi(observation)
            observation_prime, reward, done, info = env.step(act)
            SAR.append([observation,act,reward])
        
        G = np.zeros(len(SAR))
        gamma_vec = np.zeros(len(SAR))
        gamma_vec[0] = 1
        G_prev = 0
        for i,SAR_rev in enumerate(reversed(SAR)):
            G[-i-1] = gamma*G_prev+SAR_rev[2]
            obs = SAR_rev[0]
            act = SAR_rev[1]
            delta = G[-i-1] - V(obs)
            V.update(obs,G[-i-1])
            pi.update(obs,act,gamma,delta)
            G_prev = G[-i-1]    
        G_0.append(G[0])

    return G_0


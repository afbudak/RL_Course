from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    
    nS = env.spec.nS
    nA = env.spec.nA
    gamma = env.spec.gamma
    TD = env.TD
    R = env.R
    V = initV
    Q = np.zeros((nS,nA))
    
    while True:
        delta = 0;
        for i in range(nS):
            prevVal = V[i]
            action_sum_temp = 0
            for j in range(nA):
                act_pr = pi.action_prob(i,j)
                action_sum_temp = action_sum_temp + act_pr * sum(TD[i,j,:] * (R[i,j,:] + gamma*V))
            V[i] = action_sum_temp
            delta = max(delta, abs(V[i]-prevVal) )
            
        if delta<theta:
            break
    
    for s in range(nS):
        for a in range(nA):
            Q[s,a] = sum(TD[s,a,:] * (R[s,a,:] + gamma*V))
    
    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################
    
    class OptimalPolicy(Policy):
    
        
        def __init__(self, OptActionProb, OptAction):
            self.OptActionProb = OptActionProb;
            self.OptAction= OptAction;   
     
    
        def action_prob(self,state,action):
            return self.OptActionProb[state, action] 
     
    
        def action(self,state):
            return self.OptAction[state]
        

    nS = env.spec.nS
    nA = env.spec.nA
    gamma = env.spec.gamma
    TD = env.TD
    R = env.R
    V = initV
    Q = np.zeros((nS,nA))
    OptActionProb = np.zeros((nS,nA))
    OptAction = np.zeros(nS)
    
    while True:
        delta = 0
        for s in range(nS):
            prevVal = V[s]
            s_a_vals = np.zeros(nA)
            for a in range(nA):
                s_a_vals[a] = sum(TD[s,a,:] * (R[s,a,:] + gamma*V))
            V[s] = max(s_a_vals)
            OptAction[s] = s_a_vals.argmax()
            delta = max(delta,abs(V[s]-prevVal))
        if delta<theta:
            break
    
    for s in range(nS):
        best_action = OptAction[s].astype(int)
        OptActionProb[s,best_action] = 1
        
    pi = OptimalPolicy(OptActionProb,OptAction)

    return V, pi

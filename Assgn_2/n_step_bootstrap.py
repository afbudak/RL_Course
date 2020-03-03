from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    
    nS = env_spec.nS
    nA = env_spec.nA
    gamma = env_spec.gamma
    V = initV
    gamma_vec = np.zeros(n+1)
    gamma_vec[0] = 1
    for i in range(n):
        gamma_vec[i+1] =  gamma_vec[i] * gamma
    
    for eps_tr in trajs:
        T = len(eps_tr) - 1
        for tau, step_tr in enumerate(eps_tr):
            if tau+n<=T:
                rewards = np.asarray(eps_tr[tau:tau+n])[:,2]
                Vs_prime = V[eps_tr[tau+n-1][3]]
                G = sum(gamma_vec * np.append(rewards,Vs_prime))
            else:
                rewards = np.asarray(eps_tr[tau:tau+n])[:,2]
                rewlen = len(rewards)
                gamma_vec_modf = gamma_vec[0:rewlen]
                G = sum(gamma_vec_modf * rewards)
            
            V[step_tr[0]] = V[step_tr[0]] + alpha*(G - V[step_tr[0]])
                
    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    
    class OptimalPolicy(Policy):
        
        
        def __init__(self, OptActionProb, OptAction):
            self.OptActionProb = OptActionProb;
            self.OptAction= OptAction;   
     
    
        def action_prob(self,state,action):
            return self.OptActionProb[state, action] 
     
    
        def action(self,state):
            return self.OptAction[state]
        
    
    nS = env_spec.nS
    nA = env_spec.nA
    gamma = env_spec.gamma
    Q = initQ
    gamma_vec = np.zeros(n+1)
    gamma_vec[0] = 1
    OptActionProb = (1/nA)*np.ones((nS,nA))
    OptAction = np.zeros(nS)
    for i in range(n):
        gamma_vec[i+1] =  gamma_vec[i] * gamma
    
    for eps_tr in trajs:
        T = len(eps_tr) - 1
        for tau, step_tr in enumerate(eps_tr):
            rho = 1
            if tau+n<=T:
                rewards = np.asarray(eps_tr[tau:tau+n])[:,2]
                Qs_prime = Q[eps_tr[tau+n][0],eps_tr[tau+n][1]]
                G = sum(gamma_vec * np.append(rewards,Qs_prime))
                for i in range(n):
                    s = eps_tr[tau+i+1][0]
                    a = eps_tr[tau+i+1][1]
                    rho = rho * OptActionProb[s,a] / bpi.action_prob(s,a)
            else:
                rewards = np.asarray(eps_tr[tau:tau+n])[:,2]
                rewlen = len(rewards)
                gamma_vec_modf = gamma_vec[0:rewlen]
                G = sum(gamma_vec_modf * rewards)
                for i in range(rewlen-1):
                    s = eps_tr[tau+i+1][0]
                    a = eps_tr[tau+i+1][1]
                    rho = rho * OptActionProb[s,a] / bpi.action_prob(s,a)
            
            Q[step_tr[0],step_tr[1]] = Q[step_tr[0],step_tr[1]] + alpha*rho*(G - Q[step_tr[0],step_tr[1]])
            s_a_vals = Q[step_tr[0],:]
            OptAction[step_tr[0]] = s_a_vals.argmax()
            #best_action = OptAction[s].astype(int)
            OptActionProb[step_tr[0],:] = 0
            OptActionProb[step_tr[0],OptAction[step_tr[0]].astype(int)] = 1
            
    pi = OptimalPolicy(OptActionProb,OptAction)

    return Q, pi

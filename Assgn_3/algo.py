import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    gamma_vec = np.zeros(n+1)
    gamma_vec[0] = 1
    for i in range(n):
        gamma_vec[i+1] =  gamma_vec[i] * gamma
        
    for eps in range(num_episode):
        print(V(np.array([ 0.48690072,  0.04923175])))
        observation = env.reset()
        T = float('inf')
        t = 0
        R = [0] * n
        S = [0] * n
        while True:
            #env.render()
            if t<T:
                action = pi.action(observation)
                observation, reward, done, info = env.step(action)
                R.pop(0)
                R.append(reward)
                S.pop(0)
                S.append(observation)
                if done:
                    T = t+1
            tau = t-n+1
            if tau>0:
                if (tau+n)<T:
                    Vs_prime = V(S[n-1])
                    G  = sum(gamma_vec * np.append(R,Vs_prime))
                    s_tau = S[0]
                else:
                    rem_step = T - tau
                    R = R[-rem_step:]
                    gamma_vec_modf = gamma_vec[0:rem_step]
                    G = sum(gamma_vec_modf * R)
                    s_tau = S[-rem_step]
                    
                V.update(alpha,G,s_tau)
                
            if tau == T-1:
                break
            else:
                t = t+1
        
        
            
        


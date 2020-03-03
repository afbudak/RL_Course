import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here        
        hor_tile_num = (np.ceil((state_high[0]-state_low[0])/tile_width[0]) + 1).astype(int)
        ver_tile_num = (np.ceil((state_high[1]-state_low[1])/tile_width[1]) + 1).astype(int)
        num_tiles = (hor_tile_num * ver_tile_num).astype(int)       
        self.num_tiles = num_tiles
        
        xs = []
        ys = []
        #Store total tiling width
        total_width_x = (hor_tile_num) * tile_width[0]
        total_width_y = (ver_tile_num) * tile_width[1]
        for tile_idx in range(num_tilings):
            leftest_lowest_x = state_low[0] - tile_idx / num_tilings * tile_width[0]
            leftest_lowest_y = state_low[1] - tile_idx / num_tilings * tile_width[1]
            xs.append(np.arange(leftest_lowest_x, leftest_lowest_x + total_width_x, tile_width[0]))
            ys.append(np.arange(leftest_lowest_y, leftest_lowest_y + total_width_y, tile_width[1]))
            
        d = (num_actions * num_tilings * num_tiles).astype(int)
        self.hor_tile_num = hor_tile_num
        self.ver_tile_num = ver_tile_num
        self.num_tiles = num_tiles
        self.xs = xs
        self.ys = ys
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        self.d = d

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return self.d
        #raise NotImplementedError()

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        num_tiles = self.num_tiles
        hor_tile_num = self.hor_tile_num
        ver_tile_num = self.ver_tile_num
        xs = self.xs
        ys = self.ys 
        w = np.zeros(self.d)
        x = np.zeros(self.d)
        tilingidxlen = self.num_actions*self.num_tiles
        val = 0
        for tiling in range(self.num_tilings):
            active_idx = np.zeros(num_tiles)
            horidx = hor_tile_num
            veridx = ver_tile_num
            for i in range(hor_tile_num):
                if s[0]<xs[tiling][i]:
                    horidx = i
                    break
            for i in range(ver_tile_num):
                if s[1]<ys[tiling][i]:
                    veridx = i
                    break
            x[tiling*tilingidxlen+(horidx*ver_tile_num + veridx)*self.num_actions+a] = 1

        self.x = x
        if done:
            return np.zeros(self.d)
        else:
            return x

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    #TODO: implement this function
    for eps in range(num_episode):
        observation = env.reset()
        act = epsilon_greedy_policy(observation,False,w)
        x = X(observation, False, act)
        z = np.zeros(X.d)
        Qold = 0
        while True:
            observation_prime, reward, done, info = env.step(act)
            act_prime = epsilon_greedy_policy(observation_prime,False,w)
            x_prime = X(observation_prime, False, act_prime)
            Q = np.dot(w,x)
            Q_prime = np.dot(w,x_prime)
            delta = reward + gamma*Q_prime - Q
            z = gamma*lam*z + (1-alpha*gamma*lam*np.dot(z,x))*x
            w = w + alpha*(delta+Q-Qold)*z - alpha*(Q-Qold)*x
            Qold = Q
            x = x_prime
            act = act_prime
            if done:
                break
            
    return w

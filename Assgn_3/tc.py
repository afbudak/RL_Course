import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension   
        """
        # TODO: implement this method
        
        #Calculate required number of tiles per tiling
        hor_tile_num = (np.ceil((state_high[0]-state_low[0])/tile_width[0]) + 1).astype(int)
        ver_tile_num = (np.ceil((state_high[1]-state_low[1])/tile_width[1]) + 1).astype(int)
        tilepertiling = (hor_tile_num * ver_tile_num).astype(int)
        
        #Store tile location for each tiling --> only store left x and lower y per tile
        #left_xs = np.zeros((num_tilings, tilepertiling))
        #lower_ys = np.zeros((num_tilings, tilepertiling))
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
            #tile_ys = sample_tile_ys * ver_tile_num
            #tile_xs = []
            #for i in sample_tile_xs:
            #    tile_xs.extend([i]*hor_tile_num)
            #left_xs[tile_idx,:] = tile_xs
            #lower_ys[tile_idx,:] = tile_ys
            
        
        #Initialize weights
        self.w =  np.zeros((num_tilings, tilepertiling))
        self.hor_tile_num = hor_tile_num
        self.ver_tile_num = ver_tile_num
        self.tilepertiling = tilepertiling
        self.xs = xs
        self.ys = ys
        self.num_tilings = num_tilings
        
        

    def __call__(self,s):
        # TODO: implement this method
        
        num_tilings = self.num_tilings
        tilepertiling = self.tilepertiling
        hor_tile_num = self.hor_tile_num
        ver_tile_num = self.ver_tile_num
        xs = self.xs
        ys = self.ys 
        w = self.w
        features = np.zeros((num_tilings, tilepertiling))
        val = 0
        for tiling in range(self.num_tilings):
            active_idx = np.zeros(tilepertiling)
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
            active_idx[horidx*ver_tile_num + veridx] = 1
            features[tiling,:] = active_idx
                
            weights =  w[tiling,:]
            val = val + np.dot(weights,active_idx)
        self.features = features
        
        return val

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        Vs = self.__call__(s_tau)
        features = self.features
        #Calculate gradient
        for tiling in range(self.num_tilings):
            gradV = features[tiling,:]
            self.w[tiling,:] = self.w[tiling,:] + alpha*(G-Vs)*gradV
        
        return None

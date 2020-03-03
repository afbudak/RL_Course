#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:12:14 2019

@author: afbudak
"""
import numpy as np
#import matplotlib
import sys

filename = sys.argv[1]

nruns = 300
nsteps = 10000
epsilon = 0.1
std_add = 0.01
alpha = 0.1

total_reward_SA = np.zeros((nruns,nsteps))
total_reward_CS = np.zeros((nruns,nsteps))
total_optimal_SA = np.zeros((nruns,nsteps))
total_optimal_CS = np.zeros((nruns,nsteps))

for run in range(nruns):
    print(run)
    qstarmean = np.zeros(10)
    qstarstd = np.ones(10)
    q_SA = np.zeros(10)
    q_CS = np.zeros(10)
    NSA = np.zeros(10)
    NCS = np.zeros(10)
    #tot_reward_SA = 0
    #tot_reward_CS = 0
    #tot_optimal_SA = 0
    #tot_optimal_CS = 0
    
    for step in range(nsteps):
        greed_options_SA = np.where(q_SA == np.amax(q_SA))[0]
        greed_options_CS = np.where(q_CS == np.amax(q_CS))[0]
        #Find selected arm with epsilon-greedy policy
        if np.random.uniform(0,1)>epsilon:
            sel_arm_SA = np.random.choice(greed_options_SA)
            sel_arm_CS = np.random.choice(greed_options_CS)
        else:
            sel_arm_SA = np.random.choice(10)
            sel_arm_CS = np.random.choice(10)
        NSA[sel_arm_SA] = NSA[sel_arm_SA] + 1
        NCS[sel_arm_CS] = NCS[sel_arm_CS] + 1
        #calculate reward for each bandit
        rewards = np.random.normal(qstarmean,qstarstd)
        reward_SA = rewards[sel_arm_SA]
        reward_CS = rewards[sel_arm_CS]
        q_SA[sel_arm_SA] = q_SA[sel_arm_SA] + (1/NSA[sel_arm_SA])*(reward_SA - q_SA[sel_arm_SA])
        q_CS[sel_arm_CS] = q_CS[sel_arm_CS] + alpha*(reward_CS - q_CS[sel_arm_CS])
        total_reward_SA[run,step] = reward_SA
        total_reward_CS[run,step] = reward_CS
        #assign best arm
        best_arm = np.where(qstarmean == np.amax(qstarmean))[0]
        if len(best_arm)>1:
            best_arm = np.random.choice(best_arm)
        if best_arm == sel_arm_SA:
            #tot_optimal_SA = tot_optimal_SA + 1
            total_optimal_SA[run,step] = 1
        if best_arm == sel_arm_CS:
            #tot_optimal_CS = tot_optimal_CS + 1
            total_optimal_CS[run,step] = 1
        qstarmean = qstarmean + np.random.normal(0,std_add,10)
        
aver_optimal_SA = np.average(total_optimal_SA,axis=0)
aver_optimal_CS = np.average(total_optimal_CS,axis=0)
aver_reward_SA = np.average(total_reward_SA,axis=0)
aver_reward_CS = np.average(total_reward_CS,axis=0)

aver_optimal_SA = np.reshape(aver_optimal_SA, [1,10000])
aver_optimal_CS = np.reshape(aver_optimal_CS, [1,10000])
aver_reward_SA = np.reshape(aver_reward_SA, [1,10000])
aver_reward_CS = np.reshape(aver_reward_CS, [1,10000])

f = open(filename,'w+')
np.savetxt(f,aver_reward_SA)
np.savetxt(f,aver_optimal_SA)
np.savetxt(f,aver_reward_CS)
np.savetxt(f,aver_optimal_CS)
f.close()


            
            
        
    
    

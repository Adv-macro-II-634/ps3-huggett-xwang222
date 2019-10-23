# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 04:50:16 2019

@author: Xu Wang
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import random

#create transition matix
PI=np.array([[0.97, 0.03],
    [0.5, 0.5]])

#Set up parameters
#alpha = 0.35
beta = 0.9932
b=0.5
#delta = 0.025
sigma = 1.5
y_e=1
y_u=b
y_mat = [y_e,y_u]

#Set up discretized state space
a_min = -2
a_max = 5
num_a = 200 #% number of points in the grid for k

a = np.array([np.linspace(a_min, a_max, num_a)])
a_t = a.transpose()
a_mat = np.matlib.repmat(a_t, 1, num_a) #% this will be useful in a bit
a_mat_t = a_mat.transpose()

#set up guess for q
q_min = 0.98
q_max = 1
q_guess = (q_min+q_max)/2

aggsav = 1
while abs(aggsav) >=0.00001:
#Set up consumption and return function
#1st dim(rows): k today, 2nd dim (cols): k' chosen for tomorrow
    cons = a_mat_t - q_guess*a_mat 
    cons_e = cons + y_e
    cons_u = cons + y_u
    
    ret_e = np.power(cons_e, 1-sigma)/(1-sigma)
    ret_u = np.power(cons_u, 1-sigma)/(1-sigma)
    
    ret_e[cons_e<0]=-100000000000000
    ret_u[cons_u<0]=-100000000000000
    
    #Iteration
    dis = 1
    #dis_l = 1
     
    Prob_h =np.array([PI.transpose()[:,0]])
    Prob_l =np.array([PI.transpose()[:,1]])
    
    
    tol = 1e-06 #% tolerance for stopping 
    v_guess = np.zeros((2, num_a))
    
    while dis > tol:
        #compute the utility value for all possible combinations of k and k':
        value_mat_e = ret_e + beta * np.matlib.repmat(np.matmul(Prob_h,v_guess), num_a, 1)
        value_mat_u = ret_u + beta * np.matlib.repmat(np.matmul(Prob_l,v_guess), num_a, 1)
        
        #find the optimal k' for every k:
        vfn = np.array([value_mat_e.max(1),value_mat_u.max(1)])
        pol_indx = np.array([np.argmax(value_mat_e,1),np.argmax(value_mat_u,1)])
        
        #vfn = vfn.transpose()
        
        #what is the distance between current guess and value function
        dis = np.amax(abs(vfn - v_guess))
        
        #if distance is larger than tolerance, update current guess and
        #continue, otherwise exit the loop
        v_guess = vfn
    
    g_h= np.array([a[0,pol_indx[0,:]]])
    g_l= np.array([a[0,pol_indx[1,:]]])
    
    DIS = np.zeros((2, num_a))
    DIS[0,5]=0.5
    DIS[1,4]=0.5
    
    Dis_tol = 1
    while Dis_tol >0.00000001:
        POS = np.where(DIS>0)
        emp_stat = POS[0]
        ast_pos = POS[1]
        
        DIS_it = np.zeros((2, num_a))
        for i in range(1,len(emp_stat)):
            ast_npos = pol_indx[emp_stat[i],ast_pos[i]]
            ADD = PI[emp_stat[i],:]*DIS[emp_stat[i],ast_pos[i]]
            DIS_it[:,ast_npos] = DIS_it[:,ast_npos] + ADD.transpose() 
    
        Dis_tol = np.amax(abs(DIS_it - DIS))
        
        DIS = DIS_it
    
    aggsav = np.sum(g_h*DIS[0,:]+ g_l*DIS[1,:])
    
    if aggsav > 0:
        q_min = q_guess
    if aggsav < 0:
        q_max = q_guess
    q_guess = (q_min+q_max)/2

y_s=np.array(y_mat)
agg_wealth = np.matmul(np.sum(DIS,0),a_t) + np.matmul(np.sum(DIS,1),y_s)
test1 = np.append(DIS[0,:],DIS[1,:])
test2 = np.append(a+y_mat[0],a+y_mat[1])
test2.sort()
wealth_dist=np.zeros((2, 400))
wealth_dist[0,:]=test1
wealth_dist[1,:]=test2
wlth_t=wealth_dist.transpose()
pct_dis = np.cumsum((wlth_t[:,1]/agg_wealth)*wlth_t[:,0])
pct_dis_acc = [0]
pct_dis_acc.extend(pct_dis[0:-1].tolist())
gini = 1 - sum ((pct_dis_acc+ pct_dis)*wlth_t[:,0])


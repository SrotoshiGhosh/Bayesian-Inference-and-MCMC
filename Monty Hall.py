# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:01:49 2024

MONTY HALL SIMULATION

@author: DELL
"""

import random 

'''
We intend to simulate the Monty Hall problem. There are n doors, of which, behind (n-1) doors
are goats and behind 1 door is a car. The contestant chooses any door at random. The game host,
proceeds to open (n-2) doors, apart from the one picked by contestant, to reveal goats.
The contestant is given the choice to stick with his original door or switch to the unopened door. 
The probability of the car being behind his initially chosen door and the probability of the car
being behind the unopened door is calculated.
'''

n_sim=10000
n=int(input(" Enter the total number of doors in the game : "))
switch=0
stick=0

for i in range (0,n_sim):
    #the car is behind a door
    a=random.randint(0,n-1)
    
    #the contestant chooses a door at random
    b=random.randint(0,n-1)
    
    #the indices of the doors that can be opened by host
    c=[]
    for j in range (0,n):
        if (j!=a and j!=b):
            c.append(int(j))
    
    #the doors opened by host
    random.shuffle(c)
    d=c[:(n-2)]
    
    un_op=0
    #index of remaining unopened door
    for k in range (0,n):
        if(k not in d and k!=b):
            un_op=k
    
    if(un_op == a):
        switch+=1
    if(a==b):
        stick+=1

P_switch=switch/n_sim
P_stick=stick/n_sim

print (" By simulating the Monty Hall problem on python : ")
print (' The probability of winning prize if choice of doors is changed : ', P_switch)
print (' The probability of winning prize if choice of door is not changed : ',P_stick)
print (" ")

'''
From analytical calculations, if there are n number of doors, the probability of winning
with unchanged choice is given as 1/n and the probability of winning by changing the 
choice is given as (n-1)/n. We verify this result. 
'''

switch_th= (n-1)/n
stay_th=1/n

print (" The theoretical values of the above probabilities from analytical calulations : ")
print (" Probability of winning prize if choice of doors is changed : ", switch_th)
print (" Pribability of winning prize if choice of doors is not changed : ",stay_th)
    
    
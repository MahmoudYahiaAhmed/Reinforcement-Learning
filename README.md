## Overview
This task aims to learn how to deal with Reinforcement Learning regarding [this reference.](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ "this reference")
In addition to that, it's required to do the following tasks:

1) Turn this code into a module of functions that can use multiple environments.

2) Tune alpha, gamma, and/or epsilon using a decay over episodes.

3) Implement a grid search to discover the best hyperparameters.

#Setup
First, It's needed to install the following libraries while dealing with **Windows OS**:

```python
!pip install cmake "gym[atari]" scipy

```
Furthermore, It's needed to import some important libraries:
```python
import gym
from IPython.display import clear_output
from time import sleep
import random
from IPython.display import clear_output
import numpy as np
import pandas as pd
```
#Training
starting with giving the environment to the train function then return the Q table which contain the knowledge.

#Evaluation
we evaluating the model by providing its env and the Q-table that generated from training to get the penalty score and the timesteps.
#Requirements
```python
# "Taxi-v3" Environment
Q,f=train(env,10000,0.8,0.9,0.8)

##########################################
#	Episode: 10000									  
#	Training finished.										
#																 
#	Results after 100 episodes:					 
#	Average timesteps per episode: 13.24	
#	Average penalties per episode: 0.0		  


```

## Tuning using decay over episodes
It's required to change the hyperparameters while training, so we changed the hyperparameters each every 5000.
alpha=alpha*(1-0.01)
gamma=gamma*(1-0.01)
eps=eps*(1-0.6)

## Implementing Grid Search
It's required to implement Grid Search to find the best combinations of hyper parameters values to get the minimum penalty and minimum steptime.
```python
parameters = {'0.6,0.6,0.7'}
grid(env,alphas,gammas,epsilons)
#########################
after getting "all params" and append all params into list of dictonaries then sorting them. 

#Best parameters are: {'alpha': 0.6, 'gamma': 0.6, 'epsilon': 0.7, 'penalty': 0.0, 'time step': 12.89}

##########################################
#	Episode: 100000
#	Training finished.
#
#	Results after 1000 episodes:
#	Average timesteps per episode: 13.04
#	Average penalties per episode: 0.0 
##########################################

```
# Reinforcement Learning Project
## Table of Contents
- [Overview](#Overview)
- [Reuirements](#Requirements)
- [Training](#Training)
- [Evaluation](#Evaluation)

# Overview :
**Available Environments**<br>
	1-Taxi-v3<br>
	2-FrozenLake-v1<br>
	3-CliffWalking-v0<br>
you can choose from them and the defualt one is Taxi
**Proplem definition :**
There are 4 locations (labeled by different letters), and our job is to pick up
the passenger at one location and drop him off at another. We receive +20
points for a successful drop-off and lose 1 point for every time-step it
takes. There is also a 10 points penalty for illegal pick-up and drop-off
actions.
**Introduction**
In this project we will implement the Q-leaning algorithm and will see how the decay of the hyperparameter such as learning rate and discount factor and eplison will effect the results and we will implement a grid search to select the best parameters.

# Requirements :
It is required to setup this libraries to run the project
```python
!pip install gym
!pip install numpy
```
# Defualt environment info
[![Taxi-env](https://storage.googleapis.com/lds-media/images/Reinforcement_Learning_Taxi_Env.width-1200.png "Taxi-env")](https://storage.googleapis.com/lds-media/images/Reinforcement_Learning_Taxi_Env.width-1200.png "Taxi-env")
The job is to pick up the passenger at one location and drop them off in another. Here are a few things that we'd love our taxi to take care of:
- Drop off the passenger to the right location.
- Save passenger's time by taking minimum time possible to drop off
- Take care of passenger's safety and traffic rules

# Training :
## Random Action :
**Trying random actions to see how the agent movements**<br><br>
![Random Actions](https://drive.google.com/uc?export=view&id=17n6oBPFjce-AjRYUR9NDVb39VYSB3LP4)
## Q-Learning 
**After the agent has been trained**<br><br>
![Q-learning](https://drive.google.com/uc?export=view&id=1UDjqQLfPtllelNdZkTptLZKHuXoG3AOD)<br><br>
We can notice the difference and how the agent has been trained
## Decaying hyper parameters while training

![Decay](https://drive.google.com/uc?export=view&id=1U6W79ftVPSUZ_Wcv762UP_p3dXmNc4fr)
# Evaluation 
![Eval_100](https://drive.google.com/uc?export=view&id=1BAmzOSUVoL148BMOrDsRpKCP5jkn0RCN)
# Grid search 
**We used a brute force algorithm to get the best hyper parameter**<br><br>
![Grid search values](https://drive.google.com/uc?export=view&id=1rS39uEeHeYdOn5SXAYKhAkGykmt9dkSX)

**Best hyperparameters :**
```python
alpha=0.9 ,gamma=0.9, epsilon=0.9
```




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

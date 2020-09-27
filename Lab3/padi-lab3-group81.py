#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 3: The betting domain
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab3-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).

# ### 1. The POMDP model
# 
# Consider once again the POMDP problem from the homework. In this lab you will interact with a larger version of the same problem, corresponding to the following betting game. The game proceeds in rounds. At each round:
# 
# * The player is dealt a random card from the set {A&spades;, K&spades;, Q&spades;}. The card is left facing down.
# * The player must then either bet about which card he/she was dealt, or quit. There is a cost associated to quitting, but which is inferior to that of losing.
# * After betting/quitting, the card is revealed and the round ends. 
# * Before betting, the player may also try to peek which card he/she was dealt (which is cheating). Of course that there is a risk associated with peeking (modeled as a cost). Peeking may or may not succeed.
# 
# This game can be summarized as depicted below, where we have ommitted the transition labels to avoid cluttering the diagram.
# 
# <img src="pomdp.png" width="600px">
# 
# $$\diamond$$

# In this first activity, you will implement an POMDP model in Python. You will start by loading the POMDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, observations, transition probability matrices, observation probability matrices, and cost function.
# 
# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_pomdp` that receives, as input, a string corresponding to the name of the file with the POMDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 6 arrays:
# 
# * An array `S` that contains all the states in the POMDP. There is a total of $14$ states describing the different stages in the game:
# 
#     * $I$ represents the initial state of the game, before the cards are dealt.
#     * $2A$, $2B$ and $2C$ represents the "dealt cards" states. The player only observes that the card has been dealt (corresponding to observation "2"), but does not know which of the three cards ("A", "B" or "C") it has.
#     * $A$, $B$ and $C$ correspond to the states where the player peeked into the hidden card. For example, $A$ represents the state where the player was dealt card "A" and peeked into it. These states are reached from $2A$, $2B$ and $2C$, respectively, upon selecting the action "Peek".
#     * States $3A$, $3B$ and $3C$ correspond to the states where the player bets. For example, $3A$ represents the state where the player was dealt card "A" and must now make a bet, which can be $A$, $B$ or $C$.
#     * States $W$ and $L$ correspond to winning and losing the game. State $Q$ corresponds to the "Quit" state.
#     * State $F$ represents the final state of the game, right before the game resets.
#     
# * An array `A` that contains all the actions in the POMDP. The actions are denoted generically as $a$, $b$ and $c$, but represent different actions depending on the stage of the game. For example, in state $I$, since the agent does nothing but await the shuffle, all actions are equivalent and correspond to "Waiting". However, in the dealt states, the actions $a$, $b$ and $c$ correspond, respectively, to the actions "Peek", "Bet" and "Quit", respectively. In the betting states, actions $a$, $b$ and $c$ correspond to betting $A$, $B$ and $C$, respectively.
# * An array `Z` that contains all the observations in the POMDP. There is a total of 10 observations, corresponding to the observable features of the state: $I$, $2$, $A$, $B$, $C$, $3$, $W$, $Q$, $L$, $F$.
# * An array `P` containing 3 $14\times 14$ sub-arrays, each corresponding to the transition probability matrix for one action.
# * An array `O` containing 3 $14\times 10$ sub-arrays, each corresponding to the observation probability matrix for one action.
# * An array `c` containing the cost function for the POMDP.
# 
# Your function should create the POMDP as a tuple `(S, A, Z, (Pa, a = 0, ..., 2), (Oa, a = 0, ..., 2), c, g)`, where `S` is a tuple containing the states in the POMDP represented as strings (see above), `A` is a tuple containing the actions in the POMDP represented as strings (see above), `Z` is a tuple containing the observations in the POMDP represented as strings (see above), `P` is a tuple with 3 elements, where `P[u]` is an np.array corresponding to the transition probability matrix for action `u`, `O` is a tuple with 3 elements, where `O[u]` is an np.array corresponding to the observation probability matrix for action `u`, `c` is an np.array corresponding to the cost function for the POMDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the POMDP tuple.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[1]:


import numpy as np

def load_pomdp(filename, gamma = 0.9):
    with np.load(filename) as data:
        S = data['S']
        A = data['A']
        Z = data['Z']
        P = data['P']
        O = data['O']
        c = data['c']
    M = S,A,Z,P,O,c,gamma
    return M


# We provide below an example of application of the function with the file `pomdp.npz` that you can use as a first "sanity check" for your code.
# 
# ```python
# import numpy.random as rand
# 
# M = load_pomdp('pomdp.npz', 0.99)
# 
# rand.seed(42)
# 
# # States
# print('Number of states:', len(M[0]))
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('Random state:', M[0][s])
# 
# # Actions
# print('Number of actions:', len(M[1]))
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('Random action:', M[1][a])
# 
# # Observations
# print('Number of observations:', len(M[2]))
# 
# # Random observation
# z = rand.randint(len(M[2]))
# print('Random observation:', M[2][z])
# 
# # Transition probabilities
# print('Transition probabilities for the selected state/action:')
# print(M[3][a][s, :])
# 
# # Observation probabilities
# print('Observation probabilities for the selected state/action:')
# print(M[4][a][s, :])
# 
# # Cost
# print('Cost for the selected state/action:')
# print(M[5][s, a])
# 
# # Discount
# print('Discount:', M[6])
# ```
# 
# Output:
# 
# ```
# Number of states: 14
# Random state: C
# Number of actions: 3
# Random action: a
# Number of observations: 10
# Random observation: Q
# Transition probabilities for the selected state/action:
# [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# Observation probabilities for the selected state/action:
# [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# Cost for the selected state/action:
# 0.0
# Discount: 0.99
# ```

# In[2]:


import numpy.random as rand

M = load_pomdp('pomdp.npz', 0.99)

rand.seed(42)

# States
print('Number of states:', len(M[0]))

# Random state
s = rand.randint(len(M[0]))
print('Random state:', M[0][s])

# Actions
print('Number of actions:', len(M[1]))

# Random action
a = rand.randint(len(M[1]))
print('Random action:', M[1][a])

# Observations
print('Number of observations:', len(M[2]))

# Random observation
z = rand.randint(len(M[2]))
print('Random observation:', M[2][z])

# Transition probabilities
print('Transition probabilities for the selected state/action:')
print(M[3][a][s, :])

# Observation probabilities
print('Observation probabilities for the selected state/action:')
print(M[4][a][s, :])

# Cost
print('Cost for the selected state/action:')
print(M[5][s, a])

# Discount
print('Discount:', M[6])


# ### 2. Sampling
# 
# You are now going to sample random trajectories of your POMDP and observe the impact it has on the corresponding belief.

# ---
# 
# #### Activity 2.
# 
# Write a function called `gen_trajectory` that generates a random POMDP trajectory using a uniformly random policy. Your function should receive, as input, a POMDP described as a tuple like that from **Activity 1** and two integers, `x0` and `n` and return a tuple with 3 elements, where:
# 
# 1. The first element is a `numpy` array corresponding to a sequence of `n+1` state indices, $x_0,x_1,\ldots,x_n$, visited by the agent when following a uniform policy (i.e., a policy where actions are selected uniformly at random) from state with index `x0`. In other words, you should select $x_1$ from $x_0$ using a random action; then $x_2$ from $x_1$, etc.
# 2. The second element is a `numpy` array corresponding to the sequence of `n` action indices, $a_0,\ldots,a_{n-1}$, used in the generation of the trajectory in 1.;
# * The third element is a `numpy` array corresponding to the sequence of `n` observation indices, $z_1,\ldots,z_n$, experienced by the agent during the trajectory in 1.
# 
# The observation and action `numpy` arrays should have a shape `(n,)`; the state `numpy` array should have a shape `(n + 1,)`.
# 
# **Note:** You may find useful to import the numpy module `numpy.random`.
# 
# ---

# In[3]:


import numpy.random as rand

def gen_trajectory(M, x0, n):
    states    = np.empty((n+1,), dtype=int)
    states[0] = x0
    actions   = np.empty((n,), dtype=int)
    obs       = np.empty((n,), dtype=int)
    
    n_actions = len(M[1])
    n_states  = len(M[0])
    n_obs     = len(M[2])
    for i in range(n):
        a           = rand.choice(n_actions) # selected random action
        actions[i]  = a
        
        prob        = M[3][a][states[i],:] # transition for the next states
        next_s      = rand.choice(n_states, p=prob)
        states[i+1] = next_s
        
        obs_prob    = M[4][a][next_s, :]
        z           = rand.choice(n_obs,p=obs_prob)
        obs[i]      = z
    
    return states,actions,obs


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# rand.seed(42)
# 
# # Trajectory of 10 steps from state I - state index 0
# t = gen_trajectory(M, 0,  10)
# 
# print('States:', t[0])
# print('Actions:', t[1])
# print('Observations:', t[2])
# 
# # Check states, actions and observations in the trajectory
# print('Trajectory:\n{', end='')
# 
# for idx in range(10):
#     ste = t[0][idx]
#     act = t[1][idx]
#     obs = t[2][idx]
#     
#     print('(' + M[0][ste], end=', ')
#     print(M[1][act], end=', ')
#     print(M[2][obs] + ')', end=', ')
#     
# print('\b\b}')
# ```
# 
# Output:
# 
# ```
# States: [ 0  3  6  3 11 13  0  2 11 13  0]
# Actions: [2 0 2 2 1 0 0 2 2 2]
# Observations: [1 4 1 7 9 0 1 7 9 0]
# Trajectory:
# {(I, c, 2), (2C, a, C), (C, c, 2), (2C, c, Q), (Q, b, F), (F, a, I), (I, a, 2), (2B, c, Q), (Q, c, F), (F, c, I)}
# ```

# In[4]:


rand.seed(42)

# Trajectory of 10 steps from state I - state index 0
t = gen_trajectory(M, 0,  10)

print('States:', t[0])
print('Actions:', t[1])
print('Observations:', t[2])

# Check states, actions and observations in the trajectory
print('Trajectory:\n{', end='')

for idx in range(10):
    ste = t[0][idx]
    act = t[1][idx]
    obs = t[2][idx]
    
    print('(' + M[0][ste], end=', ')
    print(M[1][act], end=', ')
    print(M[2][obs] + ')', end=', ')
    
print('\b\b}')


# You will now write a function that samples a given number of possible belief points for a POMDP. To do that, you will use the function from **Activity 2**.
# 
# ---
# 
# #### Activity 3.
# 
# Write a function called `sample_beliefs` that receives, as input, a POMDP described as a tuple like that from **Activity 1** and an integer `n`, and return a tuple with `n` elements **or less**, each corresponding to a possible belief state (represented as a $1\times|\mathcal{X}|$ vector). To do so, your function should
# 
# * Generate a trajectory with `n-1` steps from a random initial state, using the function `gen_trajectory` from **Activity 2**.
# * For the generated trajectory, compute the corresponding sequence of beliefs, assuming that the agent does not know its initial state (i.e., the initial belief is the uniform belief). 
# 
# Your function should return a tuple with the resulting beliefs, **ignoring duplicate beliefs or beliefs whose distance is smaller than $10^{-3}$.**
# 
# **Note 1:** You may want to define an auxiliary function `belief_update` that receives a belief, an action and an observation and returns the updated belief.
# 
# **Note 2:** To compute the distance between vectors, you may find useful `numpy`'s function `linalg.norm`.
# 
# 
# ---

# In[5]:


import itertools

def sample_beliefs(M,n):
    
    def belief_update(b,a,z):
        P_a   = M[3][a]
        O     = np.diag(M[4][a][:,z])
        b_new = np.dot(b,P_a@O)
        b_new = b_new / np.sum(b_new)
        return b_new

    n_actions = len(M[1])
    n_states  = len(M[0])
    n_obs     = len(M[2])

    belief = np.ones((1,n_states)) / n_states # starting uniform distribution
    x0     = rand.choice(n_states)
    t      = gen_trajectory(M,x0,n-1)
    belief_state = np.empty((n,1,n_states))
    belief_state[0] = belief

    for i in range(n-1):
        new_b = belief_update(belief_state[i],t[1][i],t[2][i])
        belief_state[i+1,:,None] = new_b[:]
    
    # ignore duplicate beliefs
    _,idx  = np.unique(belief_state,axis=0,return_index=True)
    idx.sort()
    belief_state = belief_state[idx]

    for x in itertools.combinations(belief_state, 2): # pairwise comparison
        if np.linalg.norm(x[0]-x[1]) < 1e-3:
            belief_state = np.delete( belief_state, np.where((belief_state == x[0]).all(axis=2))[0] )
    
    return belief_state


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# rand.seed(42)
# 
# # 10 sample beliefs
# B = sample_beliefs(M, 10)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(B[i])
# 
# # 10 sample beliefs
# B = sample_beliefs(M, 100)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(B[i])
# ```
# 
# Output:
# 
# ```
# 7 beliefs sampled:
# [[0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07]]
# [[0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# 13 beliefs sampled:
# [[0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# [[0.   0.   0.   0.   0.   0.   0.   0.33 0.33 0.33 0.   0.   0.   0.  ]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]
# [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
# ```

# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# rand.seed(42)
# 
# # 10 sample beliefs
# B = sample_beliefs(M, 10)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(B[i])
# 
# # 10 sample beliefs
# B = sample_beliefs(M, 100)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(B[i])
# ```
# 
# Output:
# 
# ```
# 7 beliefs sampled:
# [[0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07]]
# [[0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# 13 beliefs sampled:
# [[0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# [[0.   0.   0.   0.   0.   0.   0.   0.33 0.33 0.33 0.   0.   0.   0.  ]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]
# [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
# ```

# In[6]:


rand.seed(42)

# 10 sample beliefs
B = sample_beliefs(M, 10)
print('%i beliefs sampled:' % len(B))
for i in range(len(B)):
    print(B[i])

# 10 sample beliefs
B = sample_beliefs(M, 100)
print('%i beliefs sampled:' % len(B))
for i in range(len(B)):
    print(B[i])


# ### 3. Solution methods
# 
# In this section you are going to compare different solution methods for POMDPs discussed in class.

# ---
# 
# #### Activity 4
# 
# Write a function `solve_mdp` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **optimal $Q$-function for the underlying MDP**. Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# ** Note:** You may reuse code from previous labs.
# 
# ---

# In[7]:


def solve_mdp(M):
    # init
    k         = 0
    n_actions = len(M[1])
    n_states  = len(M[0])
    J         = np.zeros( n_states )
    epsilon   = 1e-8
    cost      = M[5]
    P         = M[3]
    gamma     = M[6]
    Q         = np.empty( (n_states,n_actions) )

    while True:
        # Q-functions
        for a in range(n_actions):
            Q[:,a] = cost[:,a] + gamma*(P[a]*J).sum(axis=1)
        J_new = np.min(Q,axis=1)
        if np.linalg.norm(J_new-J) < epsilon:
            k += 1
            break
        else:
            J = J_new
            k += 1
    return Q


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# Q = solve_mdp(M)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Q-values at state %s:' % M[0][s], Q[s, :])
# 
# s = rand.randint(len(M[0]))
# print('Q-values at state %s:' % M[0][s], Q[s, :])
# 
# s = rand.randint(len(M[0]))
# print('Q-values at state %s:' % M[0][s], Q[s, :])
# ```
# 
# Output:
# 
# ```
# Q-values at state C: [0. 0. 0.]
# Q-values at state 2C: [0.3  0.   0.74]
# Q-values at state L: [1. 1. 1.]
# ```

# In[8]:


Q = solve_mdp(M)

rand.seed(42)

s = rand.randint(len(M[0]))
print('Q-values at state %s:' % M[0][s], Q[s, :])

s = rand.randint(len(M[0]))
print('Q-values at state %s:' % M[0][s], Q[s, :])

s = rand.randint(len(M[0]))
print('Q-values at state %s:' % M[0][s], Q[s, :])


# ---
# 
# #### Activity 5
# 
# You will now test the different MDP heuristics discussed in class. To that purpose, write down a function that, given a belief vector and the solution for the underlying MDP, computes the action prescribed by each of the three MDP heuristics. In particular, you should write down a function named `get_heuristic_action` that receives, as inputs:
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# 
# Your function should return an integer corresponding to the index of the action prescribed by the heuristic indicated by the corresponding string, i.e., the most likely state heuristic for `"mls"`, the action voting heuristic for `"av"`, and the $Q$-MDP heuristic for `"q-mdp"`.
# 
# ---

# In[9]:


def get_heuristic_action(b,Q,heur):
    
    Qmin      = Q.min(axis=1, keepdims=True)
    epsilon   = 1e-8
    pol_mdp   = np.isclose(Q,Qmin, atol=epsilon,rtol=epsilon).astype(int)     
    pol_mdp   = pol_mdp / pol_mdp.sum(axis=1, keepdims=True) #normalization
    
    if heur == "mls":
        pi = np.argmax(pol_mdp[np.argmax(b)])
    elif heur == "av":
        pi = np.argmax( b.dot(pol_mdp), axis=1 )[0]       
    elif heur == "q-mdp":
        pi = np.argmin( b @ Q )
    else:
        print("Choose a valid heuristic (mls, av, q-mdp)")
        return 0
    return pi


# For example, if you run your function in the examples from **Activity 3** using the $Q$-function from **Activity 4**, you can observe the following interaction.
# 
# ```python
# for b in B:
#     print('Belief:')
#     print(b)
# 
#     print('MLS action:', get_heuristic_action(b, Q, 'mls'), end='; ')
#     print('AV action:', get_heuristic_action(b, Q, 'av'), end='; ')
#     print('Q-MDP action:', get_heuristic_action(b, Q, 'q-mdp'), end='; ')
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Belief:
# [[0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07]]
# MLS action: 0; AV action: 1; Q-MDP action: 1; 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# MLS action: 1; AV action: 1; Q-MDP action: 1; 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[0.   0.   0.   0.   0.   0.   0.   0.33 0.33 0.33 0.   0.   0.   0.  ]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 1; AV action: 1; Q-MDP action: 1; 
# Belief:
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; 
# Belief:
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 1; AV action: 1; Q-MDP action: 1; 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
# MLS action: 2; AV action: 2; Q-MDP action: 2; 
# ```

# In[10]:


for b in B:
    print('Belief:')
    print(b)

    print('MLS action:', get_heuristic_action(b, Q, 'mls'), end='; ')
    print('AV action:', get_heuristic_action(b, Q, 'av'), end='; ')
    print('Q-MDP action:', get_heuristic_action(b, Q, 'q-mdp'), end='; ')

    print()


# ---
# 
# #### Activity 6
# 
# Suppose that the optimal cost-to-go function for the POMDP can be represented using the $\alpha$-vectors:

# In[11]:


G = np.array([[0.86955632, 0.63762306, 1.03088973, 1.03088973, 1.10854268, 1.10854268, 1.10854268, 1.10854268],
              [1.17976963, 0.70570298, 1.37396966, 1.37396966, 0.42351813, 1.23351813, 1.23351813, 1.14341529],
              [0.80068963, 0.70570298, 0.84908966, 0.65488963, 1.23351813, 0.42351813, 1.23351813, 1.14341529],
              [1.32556963, 0.70570298, 1.17976963, 1.37396966, 1.23351813, 1.23351813, 0.42351813, 1.14341529],
              [0.38355632, 0.63762306, 1.35455644, 1.35455644, 1.12066262, 1.12066262, 1.12066262, 1.12066262],
              [1.11255632, 0.63762306, 1.35455644, 0.38355632, 1.03687271, 1.03687271, 1.03687271, 1.03687271],
              [1.11255632, 0.63762306, 0.38355632, 1.35455644, 1.16809271, 1.16809271, 1.16809271, 1.16809271],
              [0.46841529, 0.46841529, 0.46841529, 0.46841529, 1.36841529, 1.36841529, 1.36841529, 1.36841529],
              [1.36841529, 1.36841529, 1.36841529, 1.36841529, 0.46841529, 0.46841529, 0.46841529, 1.36841529],
              [1.36841529, 1.36841529, 1.36841529, 1.36841529, 1.36841529, 1.36841529, 1.36841529, 0.46841529],
              [0.85189761, 0.51886317, 0.85189761, 0.85189761, 0.85189761, 0.85189761, 0.85189761, 0.85189761],
              [1.60189761, 1.26886317, 1.60189761, 1.60189761, 1.60189761, 1.60189761, 1.60189761, 1.60189761],
              [1.85189761, 1.51886317, 1.85189761, 1.85189761, 1.85189761, 1.85189761, 1.85189761, 1.85189761],
              [0.78512296, 0.57517959, 0.78512296, 0.78512296, 0.78512296, 0.78512296, 0.78512296, 0.78512296]])


# where the first 4 vectors correspond to action $a$, the next three vectors correspond to action $b$ and the last vector corresponds to action $c$. Using the $\alpha$-vectors above, write a function `get_optimal_action` that, given a belief vector, computes the corresponding optimal action. Your function should receive, as inputs,
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The set of optimal $\alpha$-vectors, represented as a `numpy` array like `G` above;
# * A list containing the indices of the actions corresponding to each of the $\alpha$-vectors.
# 
# Your function should return an integer corresponding to the index of the optimal action. 
# 
# ---

# In[12]:


def get_optimal_action(b,G,idx):
    pi = b @ G
    return idx[np.argmin(pi)]


# If you compute the optimal actions for the beliefs in the example from **Activity 3**, you can observe the following interaction.
# 
# ```python
# for b in B:
#     print('Belief:')
#     print(b)
# 
#     print('MLS action:', get_heuristic_action(b, Q, 'mls'), end='; ')
#     print('AV action:', get_heuristic_action(b, Q, 'av'), end='; ')
#     print('Q-MDP action:', get_heuristic_action(b, Q, 'q-mdp'), end='; ')
#     print('Optimal action:', get_optimal_action(b, G, [0, 0, 0, 0, 1, 1, 1, 2]))
# 
#     print()
# ```
# 
# Output:
# 
# ```Belief:
# [[0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07]]
# MLS action: 0; AV action: 1; Q-MDP action: 1; Optimal action: 0
# 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[0.   0.33 0.33 0.33 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
# MLS action: 1; AV action: 1; Q-MDP action: 1; Optimal action: 0
# 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[0.   0.   0.   0.   0.   0.   0.   0.33 0.33 0.33 0.   0.   0.   0.  ]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 1; AV action: 1; Q-MDP action: 1; Optimal action: 1
# 
# Belief:
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 0; AV action: 0; Q-MDP action: 0; Optimal action: 0
# 
# Belief:
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# MLS action: 1; AV action: 1; Q-MDP action: 1; Optimal action: 1
# 
# Belief:
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
# MLS action: 2; AV action: 2; Q-MDP action: 2; Optimal action: 2
# ```

# In[13]:


for b in B:
    print('Belief:')
    print(b)

    print('MLS action:', get_heuristic_action(b, Q, 'mls'), end='; ')
    print('AV action:', get_heuristic_action(b, Q, 'av'), end='; ')
    print('Q-MDP action:', get_heuristic_action(b, Q, 'q-mdp'), end='; ')
    print('Optimal action:', get_optimal_action(b, G, [0, 0, 0, 0, 1, 1, 1, 2]))

    print()


# Use the functions `get_heuristic_action` and `get_optimal_action` to compute the optimal action and the action prescribed by the three MDP heuristics at the belief `[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]]` and compare the results.

# In[14]:


b = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]])
print('MLS action:', get_heuristic_action(b, Q, 'mls'), end='; ')
print('AV action:', get_heuristic_action(b, Q, 'av'), end='; ')
print('Q-MDP action:', get_heuristic_action(b, Q, 'q-mdp'), end='; ')
print('Optimal action:', get_optimal_action(b, G, [0, 0, 0, 0, 1, 1, 1, 2]))


# <span style="color:blue">The three heuristics and the optimal action provide the same result, i.e. the first action. This is also in agreement with the underlying MDP since the given belief stands for the state "L" with 100% confidence and, no matter what action is taken, the next state is "F".</span>

# In[ ]:





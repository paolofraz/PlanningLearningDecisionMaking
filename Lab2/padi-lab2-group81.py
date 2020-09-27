#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 2: The Taxi Problem
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab2-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The MDP Model 
# 
# Consider once again the taxi domain described in the Homework which you modeled using a Markov decision process. In this lab you will interact with larger version of the same problem. You will use an MDP based on the aforementioned domain and investigate how to evaluate, solve and simulate a Markov decision problem. The domain is represented in the diagram below.
# 
# <img src="taxi.png" width="200px">
# 
# In the taxi domain above,
# 
# * The taxi can be in any of the 25 cells in the diagram. The passenger can be at any of the 4 marked locations ($Y$, $B$, $G$, $R$) or in the taxi. Additionally, the passenger wishes to go to one of the 4 possible destinations. The total number of states, in this case, is $25\times 5\times 4$.
# * At each step, the agent (taxi driver) may move in any of the four directions -- south, north, east and west. It can also pickup the passenger or drop off the passenger. 
# * The goal of the taxi driver is to pickup the passenger and drop it at the passenger's desired destination.
# 
# **Throughout the lab, use $\gamma=0.99$.**
# 
# $$\diamond$$

# In this first activity, you will implement an MDP model in Python. You will start by loading the MDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, the transition probability matrices and cost function.
# 
# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 4 arrays:
# 
# * An array `S` that contains all the states in the MDP. There is a total of $501$ states describing the possible taxi-passenger configurations. Those states are represented as strings of the form `"(x, y, p, d)"`, where $(x,y)$ represents the position of the taxi in the grid, $p$ represents the position of the passenger ($R$, $G$, $Y$, $B$, or in the taxi), and $d$ the destination of the passenger ($R$, $G$, $Y$, $B$). There is one additional absorbing state called `"Final"` to which the MDP transitions after reaching the goal.
# * An array `A` that contains all the actions in the MDP. Each action is represented as a string `"South"`, `"North"`, and so on.
# * An array `P` containing 5 $501\times 501$ sub-arrays, each corresponding to the transition probability matrix for one action.
# * An array `c` containing the cost function for the MDP.
# 
# Your function should create the MDP as a tuple `(S, A, (Pa, a = 0, ..., 5), c, g)`, where `S` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with 6 elements, where `P[a]` is an np.array corresponding to the transition probability matrix for action `a`, `c` is an np.array corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[1]:


# Add your code here.
import numpy as np

def load_mdp(filename, gamma = 0.99):
    with np.load(filename) as data:
        S = data['S']
        A = data['A']
        P = data['P']
        c = data['c']
    M = S,A,P,c,gamma
    return M


# We provide below an example of application of the function with the file `taxi.npz` that you can use as a first "sanity check" for your code.
# 
# ```python
# import numpy.random as rand
# 
# M = load_mdp('taxi.npz', 0.99)
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
# # Final state
# print('Final state:', M[0][-1])
# 
# # Actions
# print('Number of actions:', len(M[1]))
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('Random action:', M[1][a])
# 
# # Transition probabilities
# print('Transition probabilities for the selected state/action:')
# print(M[2][a][s, :])
# 
# # Cost
# print('Cost for the selected state/action:')
# print(M[3][s, a])
# 
# # Discount
# print('Discount:', M[4])
# ```
# 
# Output:
# 
# ```
# Number of states: 501
# Random state: (1, 0, 0, 2)
# Final state: Final
# Number of actions: 6
# Random action: West
# Transition probabilities for the selected state/action:
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# Cost for the selected state/action:
# 0.7
# Discount: 0.99
# ```

# In[2]:


import numpy.random as rand

M = load_mdp('taxi.npz', 0.99)

rand.seed(42)

# States
print('Number of states:', len(M[0]))

# Random state
s = rand.randint(len(M[0]))
print('Random state:', M[0][s])

# Final state
print('Final state:', M[0][-1])

# Actions
print('Number of actions:', len(M[1]))

# Random action
a = rand.randint(len(M[1]))
print('Random action:', M[1][a])

# Transition probabilities
print('Transition probabilities for the selected state/action:')
print(M[2][a][s, :])

# Cost
print('Cost for the selected state/action:')
print(M[3][s, a])

# Discount
print('Discount:', M[4])


# ### 2. Prediction
# 
# You are now going to evaluate a given policy, computing the corresponding cost-to-go.

# ---
# 
# #### Activity 2.
# 
# You will now describe the policy that, at each state $x$, always moves the taxi down (South). Recall that the action "South" corresponds to the action index $0$. Your policy should be a `numpy` array named `pol` with as many rows as states and as many columns as actions, where `pol[s,a]` should contain the probability of action `a` in state `s` according to the desired policy. 
# 
# ---

# In[3]:


pol = np.zeros( ( len(M[0]),len(M[1]) ) ) # number states \times number actions
pol[:,0] = 1 # Always go down => every state get a 1 in the colomn "down" => index 0
print('Policy of action South:\n ',pol)


# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy.
# 
# ---

# In[4]:


# return the cost-to-go to every state associated to a given policy
def evaluate_pol(M, pol):
    n_actions = len(M[1])
    n_states  = len(M[0])
    I         = np.eye(n_states)
    cost      = M[3]
    P         = M[2]
    
    Ppi = pol[:,0,None]*P[0]
    cpi = (pol * cost).sum(axis=1)
    for a in range(1,n_actions):
        Ppi += pol[:,a,None]*P[a]
    Jpi = np.linalg.inv(I-M[4]*Ppi).dot(cpi)
    
    return Jpi[:,None]


# As an example, you can evaluate the policy from **Activity 2** in the MDP from **Activity 1**.
# 
# ```python
# Jpi = evaluate_pol(M, pol)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# ```
# 
# Output: 
# ```
# Cost to go at state (1, 0, 0, 2): [70.]
# Cost to go at state (4, 1, 3, 3): [70.]
# Cost to go at state (3, 2, 2, 0): [70.]
# ```

# In[5]:


Jpi = evaluate_pol(M, pol)

rand.seed(42)

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jpi[s])


# ### 3. Control
# 
# In this section you are going to compare value and policy iteration, both in terms of time and number of iterations.

# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$. 
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``.
# 
# ---

# In[6]:


import time
# Add your code here.
def value_iteration(M):
    # init
    k         = 0
    n_actions = len(M[1])
    n_states  = len(M[0])
    J         = np.zeros( n_states )
    epsilon   = 1e-8
    cost      = M[3]
    P         = M[2]
    gamma     = M[4]
    Q         = np.empty( (n_states,n_actions) )
    start = time.time()
    while True:
        # Q-functions
        for a in range(n_actions):
            Q[:,a] = cost[:,a] + gamma*(P[a]*J).sum(axis=1)
        J_new = np.min(Q,axis=1)
        if np.linalg.norm(J_new-J) < epsilon:
            elapsed = time.time() - start
            print("Execution time: %.3f seconds" % elapsed)
            k += 1
            print("N. iterations: ", k)
            break
        else:
            J = J_new
            k += 1
    return J_new


# For example, the optimal cost-to-go for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# Jopt = value_iteration(M)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jpi)))
# ```
# 
# Output:
# ```
# Execution time: 0.031 seconds
# N. iterations: 18
# Cost to go at state (1, 0, 0, 2): [4.1]
# Cost to go at state (4, 1, 3, 3): [4.76]
# Cost to go at state (3, 2, 2, 0): [6.69]
# 
# Is the policy from Activity 2 optimal? False
# ```

# In[7]:


Jopt = value_iteration(M)

rand.seed(42)

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jopt[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jopt[s])

s = rand.randint(len(M[0]))
print('Cost to go at state %s:' % M[0][s], Jopt[s])

print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jpi)))


# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Your function should print the time it takes to run before returning, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$).
# 
# ---

# In[8]:


def policy_iteration(M):
    
    ## init ##
    n_actions = len(M[1])
    n_states  = len(M[0])
    cost      = M[3]
    P         = M[2]
    gamma     = M[4]
    k         = 0
    epsilon   = 1e-8
    pol       = np.ones( (n_states,n_actions ) ) / n_actions
    Q         = np.zeros( (n_states,n_actions) )
    start     = time.time()        
    
    while True: #loop over k => step of policy iteration

        ## Policy Evalution ##
        J = evaluate_pol(M, pol) # Contraction
        #print(J.shape) #=> J < 0 ?
        
        ## Q-functions ##
        for a in range(n_actions):
            Q[:,a,None] = cost[:,a,None] + gamma * P[a].dot(J)        
        Qmin = Q.min(axis=1, keepdims=True)
        
        ## Improved policy ##
        pol_new = np.isclose(Q,Qmin, atol=epsilon,rtol=epsilon).astype(int)     
        pol_new = pol_new / pol_new.sum(axis=1, keepdims=True) #normalization
        
        if (pol == pol_new).all():
            elapsed = time.time() - start
            k += 1
            print("Execution time: %.3f seconds" % elapsed)
            print("N. iterations: ", k)
            break
        else:
            pol = pol_new
            k += 1
            
    return pol_new


# For example, the optimal policy for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# popt = policy_iteration(M)
# 
# rand.seed(42)
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# ```
# 
# Output:
# ```
# Execution time: 0.089 seconds
# N. iterations: 3
# Policy at state (1, 0, 0, 2): North
# Policy at state (2, 3, 2, 2): West
# Policy at state (1, 4, 2, 0): West
# ```

# In[9]:


popt = policy_iteration(M)

rand.seed(42)

# Select random state, and action using the policy computed
s = rand.randint(len(M[0]))
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = rand.randint(len(M[0]))
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = rand.randint(len(M[0]))
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))


# ### 4. Simulation
# 
# Finally, in this section you will check whether the theoretical computations of the cost-to-go actually correspond to the cost incurred by an agent following a policy.

# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, corresponding to a state index
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **100** trajectories of 10,000 steps each, starting in the provided state and following the provided policy. 
# * For each trajectory, compute the accumulated (discounted) cost. 
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair ☺️.
# 
# ---

# In[10]:


def simulate(M,pol,state):
    n_actions   = len(M[1])
    n_states    = len(M[0])
    P           = M[2]
    cost        = M[3]
    total_traj  = 0
    n_steps     = 10000
    n_iter      = 100
    
    for i in range(n_iter):
        step = 0
        while( step < n_steps ):
            action = np.random.choice(np.arange(0,n_actions),p=pol[state,:])
            total_traj += pow(M[4],step)*cost[state,action]
            state  = np.random.choice(np.arange(0,n_states),p=P[action][state,:])
            step += 1
        print(i,end = ' ')
    return total_traj 


# For example, we can use this function to estimate the values of some random states and compare them with those from **Activity 4**.
# 
# ```python
# 
# rand.seed(42)
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# ```
# 
# Output:
# ````
# Cost-to-go for state (1, 0, 0, 2):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.39338954193 # 4.096... actually
# Cost-to-go for state (3, 1, 4, 1):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.09638954193
# Cost-to-go for state (3, 2, 2, 2):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.3816865569 # 4.096... actually
# ```

# In[11]:


rand.seed(42)

# Select random state, and evaluate for the optimal policy
s = rand.randint(len(M[0]))
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', Jopt[s])
print('\tEmpirical:', simulate(M, popt, s))

# Select random state, and evaluate for the optimal policy
s = rand.randint(len(M[0]))
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', Jopt[s])
print('\tEmpirical:', simulate(M, popt, s))

# Select random state, and evaluate for the optimal policy
s = rand.randint(len(M[0]))
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', Jopt[s])
print('\tEmpirical:', simulate(M, popt, s))


# In[ ]:





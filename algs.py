import numpy as np
import numpy.random as npr
import random
from simple_grid_envs import *
from matplotlib import pyplot as plt
from scipy.special import softmax
    
class QLearning: 
    def __init__(self, alpha=0.9, gamma=0.8, epsilon=0.5, policy='eps_greedy', env=HairpinMaze(10)):
        #starting aparameters
        self.alpha = alpha
        self.gamma = gamma
        self. epsilon = epsilon
        self.env = env
        #to indicate what maze you may want to run the algorithm on
        self.env = env 
        self.q_table = np.zeros([self.env.num_spaces, self.env.num_actions])
        self.policy = policy
    
    def sample_action(self, p, state):
        if p == 'softmax':
            action = npr.choice(self.env.num_actions, p=softmax(1e6 * self.q_table[self.env.coord_to_discrete(state)]))
        elif p == 'eps_greedy':
            if npr.uniform(0,1)<self.epsilon:
                action = random.randint(0,self.env.num_actions-1)
                
            # rand > eps means take action from state with highest q value
            else:
                #action = np.argmax(self.q_table[self.env.coord_to_discrete(state)])
                action = npr.choice(np.flatnonzero(np.isclose([self.env.coord_to_discrete(state), max(self.q_table[self.env.coord_to_discrete(state)]))))
        else:
            raise Exception("Not a valid policy")
        return action
    def run_alg(self, episodes=300):
        

        #list to hold the number of steps for each episode
        all_steps = []

        for episode in range(1, episodes+1):
            
            
            state = self.env.start_state
            self.env.agent_state = self.env.start_state
            
            steps, reward, score = 0,0,0

            done = False

            while not done:
                #print([state])
                action = self.sample_action(self.policy, state)
                
                next_state, reward, done = self.env.step(action)
                
                #this old q value will be replaced by new_q
                old_q = self.q_table[self.env.coord_to_discrete(state), action]

                #q value at best action to take at the next state
                best_next = np.max(self.q_table[self.env.coord_to_discrete(next_state)])

                new_q = (1-self.alpha)*old_q+self.alpha*(reward+self.gamma*best_next)

                self.q_table[self.env.coord_to_discrete(state), action]=new_q

                score += reward
                state = next_state
                steps += 1
            all_steps.append(steps)
        plt.plot(all_steps)
        print(min(all_steps))
        #print(min(all_steps))
        #return all_steps
"""    
class ValIteration:
    def __init__(self, env=FourRooms(7), gamma):
        self.env = env
        self.gamma = gamma
        self.t = np.zeros((self.env.num_spaces, self.env.num_actions, self.env.num_spaces))

        #initializing transition matrix
        self.to_left = lambda s: s-1
        self.to_right = lambda s: s+1
        self.below = lambda s: s+self.num_cols
        self.above = lambda s: s-self.num_cols
        for s in range(len(self.t)):
            for a in range(len(self.t[0])):
                if s not in self.wall_cells:
                    self.t[self.env.coord_to_discrete(s)][0][self.above(s)] = 1
                    self.t[self.env.coord_to_discrete(s)][1][self.below(s)] = 1
                    self.t[self.env.coord_to_discrete(s)][2][self.to_right(s)] = 1
                    self.t[self.env.coord_to_discrete(s)][3][self.to_left(s)] = 1
                    if self.above(s) in self.wall_cells:
                        self.t[self.env.coord_to_discrete(s)][0][self.above(s)] = 0
                    if self.below(s) in self.wall_cells:
                        self.t[self.env.coord_to_discrete(s)][1][self.below(s)] = 0
                    if self.to_right(s) in self.wall_cells:
                        self.t[self.env.coord_to_discrete(s)][2][self.to_right(s)] = 0
                    if self.to_left(s) in self.wall_cells:
                        self.t[self.env.coord_to_discrete(s)][3][self.to_left(s)] = 0  
    def bellman(self, V):
        for s in range(self.env.num_spaces):
            action_vals = []
            for a in range(self.env.num_actions):
                val_1 = 0
                val_2 = self.env.rewards
"""

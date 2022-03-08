import numpy as np
import numpy.random as npr
import random
from simple_grid_envs import *
from matplotlib import pyplot as plt
from scipy.special import softmax
    
class QLearning: 
    def __init__(self, alpha=0.9, gamma=0.8, epsilon=0.5, policy='softmax', env=FourRooms(7)):
        #starting aparameters
        self.alpha = alpha
        self.gamma = gamma
        self. epsilon = epsilon
        self.env = env
        #to indicate what maze you may want to run the algorithm on
        self.env = env 
        self.q_table = np.zeros([self.env.num_spaces, self.env.num_actions])
    def sample_action(self, p):
        if p == 'softmax':
            action = npr.choice(self.env.num_actions, p=softmax(1e6 * self.q_table[self.env.coord_to_discrete(state)]))
        if p == 'eps_greedy':
            if npr.uniform(0,1)<self.epsilon:
                    action = random.randint(0,self.env.num_actions-1)
                
            # rand > eps means take action from state with highest q value
            else:
                action = np.argmax(self.q_table[self.env.coord_to_discrete(state)])
    def run_alg(self, episodes=30):
        

        #list to hold the number of steps for each episode
        all_steps = []

        for episode in range(1, episodes+1):
            
            #initial state of each episode is the start state
                #as defined by the environment
            state = self.env.start_state
            self.env.agent_state = self.env.start_state
            #state = 0
            #print(state)
            steps, reward, score = 0,0,0

            done = False

            while not done:
                #epsilon greedy--> rand < eps means explore
                
                #sampling action from softmax distribution
                action = npr.choice(self.env.num_actions, p=softmax(1e6 * self.q_table[self.env.coord_to_discrete(state)]))
                action = self.sample_action(policy)
                #if npr.uniform(0,1)<self.epsilon:
                #    action = random.randint(0,self.env.num_actions-1)
                
                # rand > eps means take action from state with highest q value
                #else:
                #    action = np.argmax(self.q_table[self.env.coord_to_discrete(state)])


                next_state, reward, done = self.env.step(action)
                #print([state, next_state, reward,done])
                #this old q value will be replaced by new_q
                old_q = self.q_table[self.env.coord_to_discrete(state), action]

                #print(next_state)
                #print(self.q_table.shape)
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
    
    

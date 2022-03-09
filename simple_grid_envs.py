import numpy as np
from matplotlib import pyplot as plt


    
class LinearTrack:
    def __init__(self, length, reward):
        self.num_spaces = length
        self.start_state = [0,0]
        self.goal_state = [0,self.num_spaces - 1]
        self.num_actions = 2
        self.agent_state = [0,0]
        self.num_rows = 1
        self.num_cols = length
        self.reward = reward
        self.wall_cells = np.array([[0,-1], [0,self.num_spaces]])
    def coord_to_discrete(self, state):
        """takes the state corrdinate [row,col] and turns it into a sate (ex. 1,2,3..)"""
        discrete_vals = []
        for i in range(self.num_rows):
            discrete_vals.append(np.arange(self.num_cols*i, self.num_cols*(i+1)))
        return discrete_vals[state[0]][state[1]]
    def step(self, action):
        
        if action == 0:
            self.next_state = [self.agent_state[0], self.agent_state[1]+1]
        elif action == 1:
            self.next_state = [self.agent_state[0], self.agent_state[1]-1]
        else:
            raise Exception("Not a valid action!")

        if self.next_state in self.wall_cells.tolist():
            #print('hi')
            self.agent_state = self.agent_state
        else:
            self.agent_state = self.next_state
        
        if self.agent_state == self.goal_state:
            done = True
            reward = self.reward
        else:
            done = False
            reward = 0 
        
        return self.agent_state, reward, done
        
        
        

    def visualize(self):
        self.grid = np.zeros((1,self.num_spaces))
        #print(self.grid)
        self.grid[0][self.agent_state] = 1
        self.grid[0][self.goal_state] = 2
        plt.axis('off')
        plt.imshow(self.grid)

class FourRooms:
    def __init__(self, side_length, reward=10):
        self.side_length = side_length
        self.start_state = [1,1]
        #+2 because we will pad all four sides with walls
        self.num_spaces = (self.side_length+2)**2
        self.num_actions = 4
        self.num_rows = self.side_length + 2
        self.num_cols = self.side_length + 2
        self.reward = reward
        self.agent_state = [1,1] #top left corner
        self.goal_state = [self.num_rows-2, self.num_cols - 2] #bottom right corner

        self.mid = int(self.num_cols//2)
        self.earl_mid = int(self.mid//2)
        self.late_mid = self.mid+self.earl_mid 
        self.bottlenecks = [[self.mid,self.earl_mid],[self.mid,self.late_mid],[self.earl_mid,self.mid],[self.late_mid,self.mid]]
        
        ##remove bottlenecks from wall cells
        #x axis blocks
        self.blocks_x = np.array([[self.mid,i] for i in range(self.num_cols) if [self.mid,i] not in self.bottlenecks])
        #y axis blocks
        self.blocks_y = np.array([[i,self.mid] for i in range(self.num_rows) if [i,self.mid] not in self.bottlenecks])

        
        self.wall_cells = np.concatenate((self.blocks_x, self.blocks_y), axis=0)
        
        #add border walls
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i ==0 or j == 0:
                    self.wall_cells = np.append(self.wall_cells, np.array([[i,j]]), axis=0)
                if i == self.num_rows-1 or j == self.num_cols -1:
                    self.wall_cells = np.append(self.wall_cells, np.array([[i,j]]), axis=0)
        

    def coord_to_discrete(self, state):
        """takes the state corrdinate [row,col] and turns it into a sate (ex. 1,2,3..)"""
        discrete_vals = []
        for i in range(self.num_rows):
            discrete_vals.append(np.arange(self.num_cols*i, self.num_cols*(i+1)))
        return discrete_vals[state[0]][state[1]]
        
    def step(self, action):
        if action == 0:
            self.next_state = [self.agent_state[0]-1, self.agent_state[1]]
        elif action == 1:
            self.next_state = [self.agent_state[0]+1, self.agent_state[1]]
        elif action == 2:
            self.next_state = [self.agent_state[0], self.agent_state[1]+1]
        elif action == 3:
            self.next_state = [self.agent_state[0], self.agent_state[1]-1]
        else:
            raise Exception("Not a valid action!")
        
        
                
        if self.next_state in self.wall_cells.tolist():
            #print('hi')
            self.agent_state = self.agent_state
        else:
            self.agent_state = self.next_state
        
        if self.agent_state == self.goal_state:
            done = True
            reward = self.reward
        else:
            done = False
            reward = 0 
        
        return self.agent_state, reward, done
    
    def visualize(self):
        self.grid = np.zeros((self.num_rows, self.num_cols))
        #print(self.grid)
        self.grid[self.agent_state[0]][self.agent_state[1]] = 1
        self.grid[self.goal_state[0]][self.goal_state[1]] = 3
        for wall in self.wall_cells:
        #    print(wall)
            self.grid[wall[0]][wall[1]] = 2
        plt.axis('off')
        plt.imshow(self.grid)

class HairpinMaze:
    def __init__(self, wall_size, reward=10):
        self.wall_size = wall_size
        self.num_rows = self.wall_size + 2 #+2 because of end walls
        self.num_cols = 11
        self.num_spaces = self.num_rows * self.num_cols
        self.num_actions = 4
        self.reward = reward
        self.start_state = [self.num_cols-1, 1] #bottom left side
        self.agent_state = self.start_state
        self.goal_state = [1, self.num_cols -2] # top right side

        self.wall_col_positions = [2,4,6,8]
        self.bottom_wall_cols = [2,6]
        self.top_wall_cols = [4,8]
        self.bottom_wall_rows = [i for i in range(self.num_cols, 1, -1)]
        self.top_wall_rows = [i for i in range(1, self.wall_size)]
        self.wall_cells = []
        
        #adding walls for hallways
        for row in self.bottom_wall_rows:
            for col in self.bottom_wall_cols:
                self.wall_cells.append([row,col])
        for row in self.top_wall_rows:
            for col in self.top_wall_cols:
                self.wall_cells.append([row,col])
        self.wall_cells = np.array(self.wall_cells)
        #add border walls
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i ==0 or j == 0:
                    self.wall_cells = np.append(self.wall_cells, np.array([[i,j]]), axis=0)
                if i == self.num_rows-1 or j == self.num_cols -1:
                    self.wall_cells = np.append(self.wall_cells, np.array([[i,j]]), axis=0)
    def coord_to_discrete(self, state):
        """takes the state corrdinate [row,col] and turns it into a sate (ex. 1,2,3..)"""
        discrete_vals = []
        for i in range(self.num_rows):
            discrete_vals.append(np.arange(self.num_cols*i, self.num_cols*(i+1)))
        return discrete_vals[state[0]][state[1]]
    def step(self, action):
        if action == 0:
            self.next_state = [self.agent_state[0]-1, self.agent_state[1]]
        elif action == 1:
            self.next_state = [self.agent_state[0]+1, self.agent_state[1]]
        elif action == 2:
            self.next_state = [self.agent_state[0], self.agent_state[1]+1]
        elif action == 3:
            self.next_state = [self.agent_state[0], self.agent_state[1]-1]
        else:
            raise Exception("Not a valid action!")
        
        
                
        if self.next_state in self.wall_cells.tolist():
            #print('hi')
            self.agent_state = self.agent_state
        else:
            self.agent_state = self.next_state
        
        if self.agent_state == self.goal_state:
            done = True
            reward = self.reward
        else:
            done = False
            reward = 0 
        
        return self.agent_state, reward, done
    def visualize(self):
        self.grid = np.zeros((self.num_rows, self.num_cols))
        #print(self.grid)
        self.grid[self.agent_state[0]][self.agent_state[1]] = 1
        self.grid[self.goal_state[0]][self.goal_state[1]] = 3
        for wall in self.wall_cells:
        #    print(wall)
            self.grid[wall[0]][wall[1]] = 2
        plt.axis('off')
        plt.imshow(self.grid)

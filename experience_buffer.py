import numpy as np

class BUFFER():
    def __init__(self, buffer_size = 20000, batch_size = 128):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.states = np.empty([buffer_size, 200])
        self.actions = np.empty([buffer_size, 5])
        self.rewards = np.empty([buffer_size, 1])
        self.next_states = np.empty([buffer_size, 200])
        
        self.buffer_pointer = 0
        
    def getRandomExperience(self):
        if self.buffer_pointer < self.buffer_size:
            return None
        
        rand_indexs = np.random.choice(self.buffer_size, self.batch_size)
        return self.states[rand_indexs], self.actions[rand_indexs], self.rewards[rand_indexs], self.next_states[rand_indexs]
    
    def saveExperince(self, state, action, reward, next_state):
        self.states[self.buffer_pointer % self.buffer_size] = state
        self.actions[self.buffer_pointer % self.buffer_size] = action
        self.rewards[self.buffer_pointer % self.buffer_size] = reward
        self.next_states[self.buffer_pointer % self.buffer_size] = next_state
            
        self.buffer_pointer += 1
        
        if self.buffer_pointer >= 3*self.buffer_size:
            self.buffer_pointer -= self.buffer_size
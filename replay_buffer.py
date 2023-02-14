from collections import deque
import random
import numpy as np
import torch

class Buffer(object):
    
    
    def __init__(self,buffer_size):

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        

    def add(self,s,a):

        experience = (s,a)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count+=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            

    def size(self):

        return self.count


    def sample_batch(self,batch_size):

        batch = []
        
        if self.count < batch_size:
            batch = random.sample(self.buffer,self.count)
        else:
            batch = random.sample(self.buffer,batch_size)
          
        s_batch = np.array([_[0].detach().numpy() for _ in batch])
        a_batch = np.array([_[1].detach().numpy() for _ in batch])
    
        return torch.from_numpy(s_batch).double(), torch.from_numpy(a_batch).double()
    

    def clear(self):

        self.buffer.clear()
        self.count = 0
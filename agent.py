import random

class AGENT():
    def __init__(self):
        self.actions = ["KEYUP", "K_UP", "K_DOWN", "K_LEFT", "K_RIGHT"]
    
    def getAction(self):
        return random.choice(self.actions)
    
    def giveData(self):
        pass
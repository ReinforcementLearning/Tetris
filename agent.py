import random
import numpy as np
import tensorflow as tf


def weight_variable(shape, trainable = True):
    initial = tf.truncated_normal(shape, stddev=0.03)
    return tf.Variable(initial, trainable = trainable)

def bias_variable(shape, trainable = True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable = trainable)

class AGENT():
    def __init__(self, gamma = 0.99, LR = 0.01, epsilon = 0.9, epsilon_thres = 1e-4, epsilon_decay_rate = 0.99, 
                 resume = False):
        self.actions = ["KEYUP", "K_UP", "K_DOWN", "K_LEFT", "K_RIGHT"]
        self.gamma = gamma
        self.LR = LR
        self.epsilon = epsilon
        self.epsilon_thres = epsilon_thres
        self.epsilon_decay_rate = epsilon_decay_rate
        self.is_resume = resume
        self.trajectory = []
        
        self.W1 = weight_variable([200, 200])
        self.b1 = bias_variable([200])
        self.W2 = weight_variable([200, 150])
        self.b2 = bias_variable([150])
        self.W3 = weight_variable([150, 5])
        self.b3 = bias_variable([5])
        
        self.t_W1 = tf.Variable(self.W1.initialized_value(), trainable = False)
        self.t_b1 = tf.Variable(self.b1.initialized_value(), trainable = False)
        self.t_W2 = tf.Variable(self.W2.initialized_value(), trainable = False)
        self.t_b2 = tf.Variable(self.b2.initialized_value(), trainable = False)
        self.t_W3 = tf.Variable(self.W3.initialized_value(), trainable = False)
        self.t_b3 = tf.Variable(self.b3.initialized_value(), trainable = False)
        
        self.state, self.Q1 = self.getQFunction()
        self.next_state, self.Q2 = self.getTargetQFunction()
        self.rwd, self.act, self.train = self.trainNetwork()
        
        self.update_targetq = self.assignTargetQ()
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        
        if self.is_resume == True:
            self.restoreNetwork()
        
    def getAction(self, board):
        state = self.preprocessing(board)
        state = np.reshape(state, [1, 200])
        if self.epsilon > random.random():
            action = random.choice(self.actions)
            self.trajectory.append(self.actions.index(action))
            return action
        else:    
            Q_value = self.sess.run(self.Q1, feed_dict = {self.state : state})
            action = self.actions[np.argmax(Q_value[0])]
            self.trajectory.append(self.actions.index(action))
            return action
        
        
    def giveState(self, board):
        state =  self.preprocessing(board)
        state = np.reshape(state, [1,200])
        self.trajectory.append(state)
        
        
    def giveNextState(self, board, score):
        next_state = self.preprocessing(board)
        next_state = np.reshape(next_state, [1,200])
        self.trajectory.append([[score]])
        self.trajectory.append(next_state)
            
            
    def preprocessing(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == '.':
                    board[i][j] = 0
                else:
                    board[i][j] = 1
        return board
    
    
    def training(self):
        s = np.array(self.trajectory[0])
        a = np.zeros((1,5))
        a[0][self.trajectory[1]] = 1
        r = np.array(self.trajectory[2])
        s_ = np.array(self.trajectory[3])
        
        self.sess.run(self.train, feed_dict = {self.state: s,
                                               self.act: a,
                                               self.rwd: r,
                                               self.next_state: s_})
    
    def getQFunction(self):
        state = tf.placeholder(tf.float32, [None, 200])
        
        h1 = tf.nn.relu(tf.matmul(state, self.W1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
        
        Q = tf.matmul(h2, self.W3) + self.b3
        
        return state, Q
        
    def trainNetwork(self):
        rwd = tf.placeholder(tf.float32, [None, 1])
        act = tf.placeholder(tf.float32, [None, 5])
        
        values1 = tf.reduce_sum(tf.mul(self.Q1, act), reduction_indices = 1)
        values2 = rwd + self.gamma * tf.reduce_max(self.Q2, reduction_indices = 1)
        loss = tf.reduce_mean(tf.clip_by_value(tf.square(values1 - values2), 1e-10, 1.0))
        train_step = tf.train.AdamOptimizer(self.LR).minimize(loss)       
        
        self.trajectory = []
        
        return rwd, act, train_step
    
    def getHeuristicScore(self, board_):
        board = self.preprocessing(board_)
        score = 0
        
        for i in range(10):
            for j in range(19):
                if (board[i][j], board[i][j+1]) == (1,0):
                    score -= 1
            for j in range(18):
                if (board[i][j], board[i][j+1], board[i][j+2]) == (1,0,0):
                    score -= 1
            for j in range(17):
                if (board[i][j], board[i][j+1], board[i][j+2], board[i][j+3]) == (1,0,0,0):
                    score -= 1
                    
        return score
    
    def getTargetQFunction(self):
        state = tf.placeholder(tf.float32, [None, 200])
        
        h1 = tf.nn.relu(tf.matmul(state, self.t_W1) + self.t_b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.t_W2) + self.t_b2)
        
        Q = tf.matmul(h2, self.t_W3) + self.t_b3
        
        return state, Q
    
    def assignTargetQ(self):
        assignment = []
        assignment.append(self.t_W1.assign(self.W1))
        assignment.append(self.t_b1.assign(self.b1))
        assignment.append(self.t_W2.assign(self.W2))
        assignment.append(self.t_b2.assign(self.b2))
        assignment.append(self.t_W3.assign(self.W3))
        assignment.append(self.t_b3.assign(self.b3))
        
        return assignment
    
    def decayEpsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        if self.epsilon < self.epsilon_thres:
            self.epsilon = self.epsilon_thres
        
    def updateTargetQ(self):
        self.sess.run(self.update_targetq)
                    
    def saveNetwork(self):
        self.saver.save(self.sess, "Tetris.ckpt")
        
    def restoreNetwork(self):
        self.saver.restor(self.sess, "Tetris.ckpt")
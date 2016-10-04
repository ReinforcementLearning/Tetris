import random
import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class AGENT():
    def __init__(self, gamma = 0.99, LR = 0.01, epsilon = 0.9, resume = False):
        self.actions = ["KEYUP", "K_UP", "K_DOWN", "K_LEFT", "K_RIGHT"]
        self.gamma = gamma
        self.LR = LR
        self.epsilon = epsilon
        self.is_resume = resume
        
        self.state, self.Q1 = self.makeNetwork()
        self.next_state, self.Q2 = self.makeNetwork()
        self.rwd, self.act, self.train = self.trainNetwork()
        
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        
        if self.is_resume == True:
            self.restoreNetwork()
            
        
    def getAction(self, board):
        state = self.preprocessing(board)
        state = np.reshape(state, [1, 200])
        if self.epsilon > random.random():
            return random.choice(self.actions)
        else:    
            Q_value = self.sess.run(self.Q1, feed_dict = {self.state : state})
            action = self.actions[np.argmax(Q_value[0])]
            return action
        
        
    def giveData(self, board, score):
        board_train =  self.preprocessing(board)
        
        
        
    def preprocessing(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == '.':
                    board[i][j] = 0
                else:
                    board[i][j] = 1
        return board
    
    
    def makeNetwork(self):
        state = tf.placeholder(tf.float32, [None, 200])
        
        W1 = weight_variable([200, 200])
        b1 = bias_variable([200])
        
        W2 = weight_variable([200, 150])
        b2 = bias_variable([150])
        
        W3 = weight_variable([150, 5])
        b3 = bias_variable([5])
        
        h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
        
        Q = tf.matmul(h2, W3) + b3
        
        return state, Q
        
    def trainNetwork(self):
        rwd = tf.placeholder(tf.float32, [None, 1])
        act = tf.placeholder(tf.float32, [None, 5])
        
        values1 = tf.reduce_sum(tf.mul(self.Q1, act), reduction_indices = 1)
        values2 = rwd + self.gamma * tf.reduce_max(self.Q2, reduction_indices = 1)
        loss = tf.reduce_mean(tf.square(values1 - values2))
        train_step = tf.train.AdamOptimizer(self.LR).minimize(loss)       
        
        return rwd, act, train_step
             
             
    def saveNetwork(self):
        self.saver.save(self.sess, "Tetris.ckpt")
        
    def restoreNetwork(self):
        self.saver.restor(self.sess, "Tetris.ckpt")
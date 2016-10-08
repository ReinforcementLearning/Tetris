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
        #self.actions = ["KEYUP", "K_UP", "K_DOWN", "K_LEFT", "K_RIGHT"]
        self.gamma = gamma
        self.LR = LR
        self.epsilon = epsilon
        self.is_resume = resume
        self.trajectory = []
        
        self.state, self.x_Q1, self.rot_Q1 = self.makeNetwork()
        self.next_state, self.x_Q2, self.rot_Q2 = self.makeNetwork()
        self.rwd, self.x_act, self.rot_act, self.train = self.trainNetwork()
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        
        if self.is_resume == True:
            self.restoreNetwork()
            
        
    def getAction(self, board):
        state = self.preprocessing(board)
        state = np.reshape(state, [1, 200])
        if self.epsilon > random.random():
            x_action = np.random.choice(10,1)[0]
            rot_action = np.random.choice(4,1)[0]
            self.trajectory.append(x_action)
            self.trajectory.append(rot_action)
            # print("through If -> x_  :",type(x_action),"\n")
            return x_action, rot_action
        else:    
            x_Q_value, rot_Q_value = self.sess.run([self.x_Q1, self.rot_Q1], feed_dict = {self.state : state})
            x_action = np.argmax(x_Q_value[0])
            rot_action = np.argmax(rot_Q_value[0])
            self.trajectory.append(x_action)
            self.trajectory.append(rot_action)
            print("through else -> x_ :",type(x_action),"\n")
            return x_action, rot_action
        
        
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
        x_a = np.zeros((1,10))
        x_a[0][self.trajectory[1]] = 1
        rot_a = np.zeros((1,4))
        rot_a[0][self.trajectory[2]] = 1
        r = np.array(self.trajectory[3])
        s_ = np.array(self.trajectory[4])
        
        self.sess.run(self.train, feed_dict = {self.state: s,
                                               self.x_act: x_a,
                                               self.rot_act: rot_a,
                                               self.rwd: r,
                                               self.next_state: s_})
    
    def makeNetwork(self):
        state = tf.placeholder(tf.float32, [None, 200])
        
        W1 = weight_variable([200, 200])
        b1 = bias_variable([200])
        
        W2 = weight_variable([200, 150])
        b2 = bias_variable([150])
        
        x_W3 = weight_variable([150, 10])
        x_b3 = bias_variable([10])
        
        rot_W3 = weight_variable([150, 4])
        rot_b3 = bias_variable([4])
        
        x_h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
        x_h2 = tf.nn.relu(tf.matmul(x_h1, W2) + b2)
        
        rot_h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
        rot_h2 = tf.nn.relu(tf.matmul(rot_h1, W2) + b2)
        
        Q_x = tf.matmul(x_h2, x_W3) + x_b3
        Q_rot = tf.matmul(rot_h2, rot_W3) + rot_b3 
        
        return state, Q_x, Q_rot
        
    def trainNetwork(self):
        rwd = tf.placeholder(tf.float32, [None, 1])
        x_act = tf.placeholder(tf.float32, [None, 10])
        rot_act = tf.placeholder(tf.float32, [None, 4])
        
        x_values1 = tf.reduce_sum(tf.mul(self.x_Q1, x_act), reduction_indices = 1)
        x_values2 = rwd + self.gamma * tf.reduce_max(self.x_Q2, reduction_indices = 1)
        x_loss = tf.reduce_mean(tf.square(x_values1 - x_values2))
        
        rot_values1 = tf.reduce_sum(tf.mul(self.rot_Q1, rot_act), reduction_indices = 1)
        rot_values2 = rwd + self.gamma * tf.reduce_max(self.rot_Q2, reduction_indices = 1)
        rot_loss = tf.reduce_mean(tf.square(rot_values1 - rot_values2))
        
        loss = tf.clip_by_value((x_loss + rot_loss), 1e-10, 1.0)
        train_step = tf.train.AdamOptimizer(self.LR).minimize(loss)       
        
        self.trajectory = []
        
        return rwd, x_act, rot_act, train_step
    
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
                    
        return score
    
             
    def saveNetwork(self):
        self.saver.save(self.sess, "Tetris.ckpt")
        
    def restoreNetwork(self):
        self.saver.restor(self.sess, "Tetris.ckpt")
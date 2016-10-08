import random
import numpy as np
import tensorflow as tf
from experience_buffer import BUFFER


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class AGENT():
    def __init__(self, gamma = 0.99, LR = 0.001, epsilon = 0.99, epsilon_thres = 1e-4, epsilon_decay_rate = 0.99, 
                 resume = False):
        self.actions = ["KEYUP", "K_UP", "K_DOWN", "K_LEFT", "K_RIGHT"]
        self.gamma = gamma
        self.LR = LR
        self.epsilon = epsilon
        self.epsilon_thres = epsilon_thres
        self.epsilon_decay_rate = epsilon_decay_rate
        self.is_resume = resume
        self.trajectory = []
        self.buffer = BUFFER()
        
        self.conv_W1 = weight_variable([3, 3, 1, 20])
        self.conv_b1 = bias_variable([20])
        self.conv_W2 = weight_variable([3, 3, 20, 40])
        self.conv_b2 = bias_variable([40])
        self.W1 = weight_variable([5*10*40, 512])
        self.b1 = bias_variable([512])
        self.W2 = weight_variable([512, 5])
        self.b2 = bias_variable([5])
        
        
        self.t_conv_W1 = tf.Variable(self.conv_W1.initialized_value(), trainable = False)
        self.t_conv_b1 = tf.Variable(self.conv_b1.initialized_value(), trainable = False)
        self.t_conv_W2 = tf.Variable(self.conv_W2.initialized_value(), trainable = False)
        self.t_conv_b2 = tf.Variable(self.conv_b2.initialized_value(), trainable = False)
        self.t_W1 = tf.Variable(self.W1.initialized_value(), trainable = False)
        self.t_b1 = tf.Variable(self.b1.initialized_value(), trainable = False)
        self.t_W2 = tf.Variable(self.W2.initialized_value(), trainable = False)
        self.t_b2 = tf.Variable(self.b2.initialized_value(), trainable = False)
        
        
        self.state, self.Q1 = self.getQFunction()
        self.next_state, self.Q2 = self.getTargetQFunction()
        self.rwd, self.act, self.loss, self.train = self.trainNetwork()
        
        self.update_targetq = self.assignTargetQ()
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        
        if self.is_resume == True:
            self.restoreNetwork()
        
    def getAction(self, board):
        state = np.reshape(board, [1, 200])
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
        self.trajectory = []
        self.trajectory.append(state)
        
        
    def giveNextState(self, board, score):
        next_state = np.reshape(board, [1,200])
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
        
        self.buffer.saveExperince(s, a, r, s_)
        experience = self.buffer.getRandomExperience()
        if experience != None:
            _, step_loss = self.sess.run([self.train, self.loss], feed_dict = {self.state: experience[0],
                                                                               self.act: experience[1],
                                                                               self.rwd: experience[2],
                                                                               self.next_state: experience[3]})
            
            return step_loss
        
        return 0
    
    def getQFunction(self):
        state = tf.placeholder(tf.float32, [None, 200])
        state_image = tf.reshape(state, [-1, 10, 20, 1])
        
        conv_h1 = tf.nn.relu(tf.nn.conv2d(state_image, self.conv_W1, strides=[1,1,1,1], padding='SAME') + self.conv_b1)
        conv_h2 = tf.nn.relu(tf.nn.conv2d(conv_h1, self.conv_W2, strides=[1,2,2,1], padding='SAME') + self.conv_b2)
        
        conv_h2_flat = tf.reshape(conv_h2, [-1, 50*40])
        
        h1 = tf.nn.relu(tf.matmul(conv_h2_flat, self.W1) + self.b1)
       
        Q = tf.matmul(h1, self.W2) + self.b2
        
        
        return state, Q
        
    def trainNetwork(self):
        rwd = tf.placeholder(tf.float32, [None, 1])
        act = tf.placeholder(tf.float32, [None, 5])
        
        values1 = tf.reduce_sum(tf.mul(self.Q1, act), reduction_indices = 1)
        values2 = rwd + self.gamma * tf.reduce_max(self.Q2, reduction_indices = 1)
        loss = tf.clip_by_value(tf.reduce_mean(tf.square(values1 - values2)), 1e-10, 1.0)
        train_step = tf.train.AdamOptimizer(self.LR).minimize(loss)       
        
        return rwd, act, loss, train_step
    
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
    
    def saveTrajectory(self):
        s = np.array(self.trajectory[0])
        a = np.zeros((1,5))
        a[0][self.trajectory[1]] = 1
        r = np.array(self.trajectory[2])
        s_ = np.array(self.trajectory[3])
        
        self.buffer.saveExperince(s, a, r, s_)
        
        
    def getTargetQFunction(self):
        state = tf.placeholder(tf.float32, [None, 200])
        state_image = tf.reshape(state, [-1, 10, 20, 1])
        
        conv_h1 = tf.nn.relu(tf.nn.conv2d(state_image, self.t_conv_W1, strides=[1,1,1,1], padding='SAME') + self.t_conv_b1)
        conv_h2 = tf.nn.relu(tf.nn.conv2d(conv_h1, self.t_conv_W2, strides=[1,2,2,1], padding='SAME') + self.t_conv_b2)
        
        conv_h2_flat = tf.reshape(conv_h2, [-1, 50*40])
        
        h1 = tf.nn.relu(tf.matmul(conv_h2_flat, self.t_W1) + self.t_b1)
       
        Q = tf.matmul(h1, self.t_W2) + self.t_b2
        
        return state, Q
    
    def assignTargetQ(self):
        assignment = []
        assignment.append(self.t_conv_W1.assign(self.conv_W1))
        assignment.append(self.t_conv_b1.assign(self.conv_b1))
        assignment.append(self.t_conv_W2.assign(self.conv_W2))
        assignment.append(self.t_conv_b2.assign(self.conv_b2))
        assignment.append(self.t_W1.assign(self.W1))
        assignment.append(self.t_b1.assign(self.b1))
        assignment.append(self.t_W2.assign(self.W2))
        assignment.append(self.t_b2.assign(self.b2))
        
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
        self.saver.restore(self.sess, "Tetris.ckpt")
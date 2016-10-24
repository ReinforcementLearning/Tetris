#Tetris
**Reinforcement Leanring for Tetris**  

###목적
유명하고 간단한 그러나 학습하기에는 쉽지 않은 Tetris를 학습시켜봄으로서 강화학습 알고리즘의 적용과 구현에 대한 능력을 키운다
  
###현재까지 진행상황  
기존의 Tetromino라는 Pygame 코드에다가 학습을 위해 Deep Q-Networks(DQN)을 이용해서 Tetris Agent를 만들었다  

CNN, Target Q, Experience Replay 모두 구현.  
네트워크나 Experience Buffer크기나 여러 hyperparamete다r들을 세부조정만 하면 될 것 같습니다.  

####문제점
1. loss가 줄지 않는다.  
2. 일정시간 학습 후에 많은 state들에(거의 모든) 대해 똑같은 action을 보인다.  
  
몇가지 원인이 될 수 있는 것들을 정리해보면  

1. 코드 버그(그냥 코드 자체가 문제가 있을수도....)  
2. Action이 너무 많다. 한번 reward를 받기까지 너무 많은 액션들을 하고 어느 액션이 현재 reward에 영향을 줬는지 모르니 학습이 잘 안되는 것 같습니다.  
3. reward의 문제. reward를 좀 손보면 학습이 될지도...  
4. Network의 크기. Network가 너무 작거나.. 하여 문제가 생기는...   
5. hyperparameter. learning rate같은 몇몇 hyperparameter를 조정해줘야 할 수도 있습니다.  

####State & Action
1. 현재 20x10 크기의 board를 State로 둠.  
Action은 ['right', 'left', 'down', 'rotation', 'no_action'] 다섯가지로 함.  

2. 대안으로 State는 그대로 하고, 모양, 시작 위치를 Action으로 하는 방법도 있음.  

####Network
현재 Convolutional Neural Network이용.  
conv3-16(stride 1x1), conv3-32(stride 2x2), FC-512, FC-5 layers  
activation function은 ReLU이용.  


###Requirements
[Python 2.7 or Python 3.3+](https://www.python.org/downloads/)  
[Pygame 1.9.1+](http://www.pygame.org/download.shtml)  
[Tensorflow r0.9+](https://www.tensorflow.org/)  

OS : Mac OS or Linux  

###References  
[Playing Tetris with Deep Reinforcement Learning](http://cs231n.stanford.edu/reports2016/121_Report.pdf)  
[Reinforcement Learning Tetris Example](http://melax.github.io/tetris/tetris.html)  
[CatchGame DQN](http://solarisailab.com/archives/486)  
[Making Game with Python & Pygame](http://inventwithpython.com/makinggames.pdf)  



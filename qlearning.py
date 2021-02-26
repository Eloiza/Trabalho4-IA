import util
import random
import matplotlib.pyplot as plt
import numpy as np

class Agente():
    def __init__(self, env, epsilon=0.01, alpha=0.1, gamma=0.2):
        self.epsilon = epsilon 
        self.alpha = alpha   
        self.gamma = gamma  
        self.env = env 

        #initiates the Q-Table
        n_states  = env.getStateSpaceSize()      #total states - every different cell in the game   
        n_actions = env.getActionSpaceSize()     #total actions - move up, right, left, down
      
        #optimistic initialization 
        self.qTable = np.full((n_states, n_actions), alpha)

        #stores how many times a action in given state were taken 
        self.action_taken = np.zeros((n_states, n_actions)) 
        self.confidence = self.epsilon
        self.time_stamp = 0 

    '''    
    Return an action based on the  UCB policy 
    '''
    def UCB_policy(self, state):
        actions = self.env.getLegalActions(state)

        #choose the best action based on the UCB policy 
        best_action = (actions[0], 0)
        for action in actions:

            n = self.action_taken[state, action]
            t = self.time_stamp
            
            #calculates the UCB value for each action
            ubc_value = (self.qTable[state, action] / max(self.qTable[state, :]) + 0.001) + np.sqrt(self.confidence*np.log(t+1)/(2*n) + 0.01)
            
            #check if this is the best action
            if(ubc_value > best_action[1]):
                best_action = (action, ubc_value)

        return best_action[0]
    '''    
    Return an action based on the e-greedy policy 
    '''
    def egreedy_policy(self, state):
        actions = self.env.getLegalActions(state)
        
        #perform the best action - exploitation
        if(random.uniform(0, 1) > self.epsilon):
            best_action = (0, self.qTable[state][0]) #tuple to represent (best_action, q_value)
           
            #searches the best action, with the highest Q-value in the table
            for action in actions:
                if(self.qTable[state, action] > best_action[1]):
                    best_action = (action, self.qTable[state, action])

            # action = np.argmax(self.qTable[state])
            action = int(best_action[0]) 
        
        #take a random action - exploration
        else:
            action = random.choice(actions)
        
        return action

    '''    
    Choose an action based in a policy. It can be e-greedy policy or UCB policy 
    '''
    def escolheAcao(self, state, ucb=False):
        self.time_stamp += 1

        if(ucb):
            action = self.UCB_policy(state)
        else:
            action = self.egreedy_policy(state)

        return action

           
    '''
    Updates the Q-value of the chosen action
    '''
    def atualizaAgente(self, state, action, reward, next_state):
        sample = reward + self.gamma * np.max(self.qTable[next_state,:])
        self.qTable[state, action] = (1 - self.alpha)*self.qTable[state,action] + self.alpha*sample
    
        self.action_taken[state,action] += 1

    '''
    Metodo chamado ao fim de cada episodio.
    A variavel is_train ira indicar se eh um episodio de treinamento
    '''
    def novoEpisodio(self, is_train=True):
        #code to low the epsilon value through time
        # self.epsilon = self.epsilon * self.alpha/(self.time_stamp + self.alpha)
        pass
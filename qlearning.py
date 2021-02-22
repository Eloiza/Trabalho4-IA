import util
import random
import matplotlib.pyplot as plt
import numpy as np

class Agente():
    def __init__(self, env, epsilon=0.01, alpha=0.8, gamma=0.2):
        self.epsilon = epsilon # exploracao
        self.alpha = alpha   # taxa de aprendizado
        self.gamma = gamma   # desconto - determina importancia de acoes futuras
        self.env = env #ambiente

        #initiates the Q-Table
        n_states  = 16      #total states - every different cell in the game   
        n_actions = 4       #total actions - move up, right, left, down
        self.qTable = np.zeros((n_states, n_actions))
        
    '''
    Based on a state returns an action - e-greedy policy 
    '''
    def escolheAcao(self, state):
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
    Updates the Q-value of the chosen action
    '''
    def atualizaAgente(self, state, action, reward, next_state):
        sample = reward + self.gamma * np.max(self.qTable[next_state,:])
        self.qTable[state, action] = (1 - self.alpha)*self.qTable[state,action] + self.alpha*sample
    
    '''
    Metodo chamado ao fim de cada episodio.
    A variavel is_train ira indicar se eh um episodio de treinamento
    '''
    def novoEpisodio(self, is_train=True):
        
        pass

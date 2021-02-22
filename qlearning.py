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
        n_actions = 4       #total actions - move up, right, left, down
        n_states  = 16      #total states - every different cell in the game   
        self.qTable = np.zeros((n_states, n_actions))

    '''
    Based on a state returns an action - e-greedy policy 
    '''
    def escolheAcao(self, state):
        actions = self.env.getLegalActions(state)

        #perform the best action
        if(random.uniform(0, 1) > self.epsilon):
            action = np.argmax(self.qTable[state])

        #take a random action 
        else:
           action = random.choice(actions)
        

        return action
           
    '''
    Updates the Q-value of the chosen action
    '''
    def atualizaAgente(self, state, action, reward, next_state):
        self.qTable[state, action] = (1 - self.alpha)*self.qTable[state,action] + self.alpha*(reward + self.gamma * np.max(self.qTable[next_state,:]) - self.qTable[state,action])

    '''
    Metodo chamado ao fim de cada episodio.
    A variavel is_train ira indicar se eh um episodio de treinamento
    '''
    def novoEpisodio(self, is_train=True):
        
        pass

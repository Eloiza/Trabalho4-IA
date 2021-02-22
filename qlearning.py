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
        self.qTable = []
        n_actions = 4       #total actions - move up, right, left, down
        n_states  = 16      #total states - every different cell in the game   
        for i in range(n_actions):
            self.qTable.append([0 for i in range(n_states)])

    '''
    Based on a state returns an action - e-greedy policy 
    '''
    def escolheAcao(self, state):
        actions = self.env.getLegalActions(state)
        action = random.choice(actions)

        # print("Actions:", actions)
        # print("Action:", action)
        # print("State:", state)
        # self.env.env.render()
        # print()

        #perform the best action
        if(random.random() > self.epsilon):
            best_action = (0,0) #tuple to represent (best_action, q_value)
            
            #searches the best action, with the highest Q-value in the table
            for action in range(len(self.qTable)):
                for state in range(len(self.qTable[i])):
                    if(qTable[action][state] > best_action):
                        best_action = (best_action, qTable[action][state])

            action = best_action[0]

        #take a random action 
        else:
           action  = random.choice(actions)
        
        return action
           
    '''
    Updates the Q-value of the choosen action
    '''
    def atualizaAgente(self, state, action, reward, next_state):
        #atualiza os Q-values definidos de acordo com a regra de atualização do alg
        #Equacoes 1 e 2 representam as regras de atualizacao 

        q_value = qTable[action][state]
        sample = reward + self.gamma * max(0, self.qTable[action, next_state])

        self.qTable[action][state] = (1 - self.alpha)*q_value + self.alpha*sample


    def novoEpisodio(self, is_train=True):
        '''
        Metodo chamado ao fim de cada episodio.
        A variavel is_train ira indicar se eh um episodio de treinamento
        '''
        pass

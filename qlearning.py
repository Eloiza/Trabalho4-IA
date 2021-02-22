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
        n_states  = 4       #total states - starting position, frozen cell, hole, goal position   
        for i in range(n_actions):
            self.qTable.append([0 for i in range(n_states)])

    '''
    Based on a state returns an action - e-greedy policy 
    '''
    def escolheAcao(self, state):
        actions = self.env.getLegalActions(state)
        
        #perform the best action
        if(random.random() > self.epsilon):
            action = ...

        #take a random action 
        else:
           action  = random.choice(actions)
        
        return action
           
    '''
    Metodo chamado apos executar cada acao durante o treinamento
    '''
    def atualizaAgente(self, state, action, reward, next_state):
        #atualiza os Q-values definidos de acordo com a regra de atualização do alg
        #Equacoes 1 e 2 representam as regras de atualizacao 

        #(1) amostra = R(s,a,s') + gamma*maxQ(s', a')
        #(2) Temporal DIfference or TD-Update -> Q(s,a) = (1-alpha)*Q(s,a)+alpha*amostra

        pass

    def novoEpisodio(self, is_train=True):
        '''
        Metodo chamado ao fim de cada episodio.
        A variavel is_train ira indicar se eh um episodio de treinamento
        '''
        pass

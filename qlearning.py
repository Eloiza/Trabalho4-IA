import util
import random
import matplotlib.pyplot as plt
import numpy as np

class Agente():
    def __init__(self, env, epsilon=0.01, alpha=0.8, gamma=0.2):
        self.epsilon = epsilon # exploracao
        self.alpha = alpha   # taxa de aprendizado
        self.gamma = gamma   # desconto
        self.env = env #ambiente

        "*** SEU CODIGO AQUI ***"

    def escolheAcao(self, state):
        '''
        Metodo que deve retornar uma acao a partir de um estado
        '''
        actions = self.env.getLegalActions(state)

        # Exemplo: Agente de decisao aleatoria
        return random.choice(actions)
        ##

        "*** SEU CODIGO AQUI ***"

    def atualizaAgente(self, state, action, reward, next_state):
        '''
        Metodo chamado apos executar cada acao durante o treinamento
        '''

        "*** SEU CODIGO AQUI ***"
        pass

    def novoEpisodio(self, is_train=True):
        '''
        Metodo chamado ao fim de cada episodio.
        A variavel is_train ira indicar se eh um episodio de treinamento
        '''

        "*** SEU CODIGO AQUI ***"
        "*** (Eh possivel que a sua implementacao nao utilize este metodo) ***"
        pass

import numpy as np
import matplotlib.pyplot as plt

class ArqvSuportDPI:
    ''' As seguintes funções interagem com o arquivo principal'''

    def __init__(self):
        '''Carrega as constantes que não mudam de valor'''

        # Constantes
    
        m = 1     # Massa do carro [kilograma]
        m1 = 1  # Massa do pêndulo 1 [kilograma]
        m2 = 1   #  Massa do pêndulo 2 [kilograma]
        l1 = 0.05 # Comprimento do centro de massa da haste 1 até a ponta [metros]
        l2 = 0.05 # Comprimento do centro de massa da haste 2 até a ponta [metros]
        g = 9.81  # Aceleração da gravidade [m/s^2]
        f0 = 0.01 # Coeficiente de atrito do carro
        f1 = 0.007 # Coeficiente viscoso de amortecimento rotacional do pêndulo inferior
        f2 = 0.007 # Coeficiente viscoso de amortecimento rotacional do pêndulo superior
        J1 = 0.00083 # Momento de inércia do pêndulo 1
        J2 = 0.00083 # Momento de inércia do pêndulo 2
        Ts = 0.01 # Peíodo de amostragem [segundos]

        saidas = 3 # numero de saidas
        entradas = 1 # numero de entradas
        hz = 10 # horizonte de previsão

        trajetoria = 1 # Escolha 1, 2 ou 3
        version = 2 # Para a traj 3
        intervalo_de_tempo = 10 # [s] - duração de toda a simulação
        
        # Pesos matriciais para a função custo (devem ser diagonais)[Ajuste por tentativa e erro]
        Q = np.matrix('1 0 0 ;0 1 0 ;0 0 1') # Pesos para saídas (todas as amostras, exceto a última)
        S = np.matrix('600 0 0 ;0 600 0;0 0 600') # Pesos para os ultimos resultados do horizonte de previsao
        R = np.matrix('0.01') # pesos para entradas (Apenas 1 entrada no caso)

        self.constantes = {'m':m, 'm1':m1, 'm2':m2, 'g':g, 'l1':l1, 'l2':l2,\
            'J1':J1, 'J2':J2, 'f0':f0, 'f1':f1, 'f2':f2, 'saidas':saidas,'trajetoria':trajetoria,\
            'Ts':Ts, 'hz':hz, 'Q':Q, 'R':R, 'S':S, 'r':r, 'f':f,\
            'intervalo_de_tempo':intervalo_de_tempo}

        return None

    def gerador_de_trajetoria(self, t, r, f):
        '''Este método cria a trajetória para um carro seguir'''

        Ts = self.constantes['Ts']
        trajetoria = self.constantes['trajetoria']

        # Define o comprimento x
        xr = np.linspace(0,t[-1],num=len(t)) #[inicio,fim,qtd de elementos]

        # Define trajetorias
        if trajetoria == 1: # Trajetória que estabiliza (Resposta natural)
            x = np.zeros_like(xr) # 0 
            phi1_r = np.zeros_like(xr) # 0
            phi2_r = np.zeros_like(xr) # 0

        elif trajetoria == 2: # Adiciona degrau
            x = np.zeros_like(xr) # Condicao inicial 0
            x[xr < 2] = 0 # 0 de 0s até 2s 
            x[xr >= 2] = 1 # 1 de 2 até 10
            phi1_r = np.zeros_like(xr)
            phi2_r = np.zeros_like(xr)

        elif trajetoria == 3: # Onda quadrada
            x = np.zeros_like(xr)
            x[xr < 15] = -1.5 # 0 de 0s até 5s 
            # Primeiro degrau
            x[(xr >= 15) & (xr < 30)] = 1.5 # passa de para 3 
            x[(xr >= 30) & (xr < 45)] = -1.5 # passa de para 3 
            x[(xr >= 45) & (xr < 60)] = 1.5 # passa de para 3 
            phi1_r = np.zeros_like(xr)
            phi2_r = np.zeros_like(xr)

        elif trajetoria == 4: # Trajetória oscilatória
            x = 5*np.tanh(t-t[1]/2)
            phi1_r = np.zeros_like(xr)
            phi2_r = np.zeros_like(xr)

        elif trajetoria == 5: # Impulso
            x =  np.zeros_like(xr)
            x[xr < 2] = 0
            x[(xr>=2) & (xr<2.01)] = 10
            x[xr>=2.01] = 0 
            phi1_r = np.zeros_like(xr)
            phi2_r = np.zeros_like(xr)

        return x,phi1_r,phi2_r
    
    def espaco_de_estados(self):
        '''Esta função forma as matrizes do espaço de estados e as transforma na forma discreta'''
        Ts = self.constantes['Ts']
        #x_dot=self.constantes['x_dot']

        # Obtem as matrizes de espaço de estado para o controle
        A0 = self.constantes['A0']

        A1 = self.constantes['A1']
        A2 = self.constantes['A2']

        A3 = self.constantes['A3']
        A4 = self.constantes['A4']
        A5 = self.constantes['A5']
        A6 = self.constantes['A6']
        A7 = self.constantes['A7']
        A8 = self.constantes['A8']

        B1 = self.constantes['B1']
        B2 = self.constantes['B2']
        B3 = self.constantes['B3']

        A=np.array([[0, 0, 0, 1, 0, 0 ],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],
            [0, -(A1*A3)/A0, -(A2*A6)/A0, 0, 0, 0],[0, -(A1*A4)/A0,-(A2*A7)/A0, 0, 0, 0],[0, -(A1*A5)/A0,-(A2*A8)/A0, 0, 0, 0]])
        B=np.array([[0],[0],[0],[B1/A0],[B2/A0],[B3/A0]])
        C=np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0]])
        D=0
        print(A)

        # Discretizar o sistema por avanço de Euler 
        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        Dd=D

        return Ad,Bd,Cd,Dd
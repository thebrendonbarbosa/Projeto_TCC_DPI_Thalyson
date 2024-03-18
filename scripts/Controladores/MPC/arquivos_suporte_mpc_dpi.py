import numpy as np
import matplotlib.pyplot as plt

class DadoSuporteDpi:
    ''' As seguintes funções interagem com o arquivo principal'''

def __init__(self):
    '''Carrega as constantes'''

    # Constantes
    m = 1     # kilograma
    m1 = 1    # kilograma
    m2 = 1    # kilograma
    l1 = 0.05 # metros
    l2 = 0.05 # metros
    g = 9.81  # m/s^2
    f0 = 0.007
    f1 = 0
    f2 = 0
    J1 = 8.2 
    J2 = 0.4
    Ts = 0.01

    # Parâmetros para a mudança de faixa:: [phi1_ref 0;0 phi2_ref]
        
    # Pesos matriciais para a função custo (devem ser diagonais)
    Q=np.matrix('1 0;0 1') # pesos para saídas (todas as amostras, exceto a última)
    S=np.matrix('1 0;0 1') # pesos para os ultimos resultados do horizonte de previsao
    R=np.matrix('1') # pesos para entradas (apenas 1 entrada no caso)

    saidas = 2  # número de saídas
    hz = 20 # horizonte de previsao
    x_dot=20 # velocidade longitudinal pendulo

    # Sinal de referência 
    r=4  # amplitude
    f=0.01 # frequencia
    intervalo_de_tempo = 10 # [s] - duração de toda a simulação

    trajetoria = 1

    self.constantes = {'m':m, 'm1':m1, 'm2':m2, 'g':g, 'l1':l1, 'l2':l2,\
                       'J1':J1, 'J2':J2, 'f0':f0, 'f1':f1, 'f2':f2, 'saidas':saidas,\
                        'Ts':Ts, 'hz':hz, 'Q':Q, 'R':R, 'S':S, 'r':r, 'f':f,\
                            'intervalo_de_tempo':intervalo_de_tempo, 'trajetoria':trajetoria}
    
    return None

def gerador_de_trajetoria(self, t, r, f):
    '''Este método cria a trajetória para um carro seguir'''

    Ts = self.constantes['Ts']
    x_dot = self.constantes['x_dot']
    trajetoria = self.trajetoria['trajetoria']

    # Define o comprimento x, depende da velocidade longitudinal do carro
    x=np.linspace(0,x_dot*t[-1],num=len(t))

    # Define trajetorias

    if trajetoria==1:
        y=-9*np.ones(len(t))
    elif trajetoria==2:
        y=9*np.tanh(t-t[-1]/2)
    elif trajetoria==3:
        aaa=-28/100**2
        aaa=aaa/1.1
        if aaa<0:
            bbb=14
        else:
            bbb=-14
        y_1=aaa*(x+self.constantes['largura_da_faixa']-100)**2+bbb
        y_2=2*r*np.sin(2*np.pi*f*x)
        y=(y_1+y_2)/2
    else:
        print("Para a trajetória, escolhe apenas 1, 2, ou 3 como um valor inteiro")
        exit()


def espaco_de_estados(self):
    '''Esta função forma as matrizes do espaço de estados e as transforma na forma discreta'''

    # Constantes
    m = self.constantes['m']
    m1 = self.constantes['m1']
    m2 = self.constantes['m2']
    l1 = self.constantes['l1']
    l2 = self.constantes['l2']
    g = self.constantes['g']
    f0 = self.constantes['f0']
    f1 = self.constantes['f1']
    f2 = self.constantes['f2']
    J1 = self.constantes['J1']
    J2 = self.constantes['J2']
    Ts = self.constantes['Ts']
    x_dot=self.constantes['x_dot']

    # Obtem as matrizes de espaço de estado para o controle
    A01 =-J1*J2*(m*m1*m2)-2*m2*l2*l2*J1(m+m1+m2)-J2*l1*l1*(m*m1+2*m*m2+m1*m2)
    A02 =  2*l1*l1*m2*(J2*m2-m*l2*l2*m1+3/2 *l2*l2*m1*m2+l2*l2*m2)
    A0 = A01 + A02

    A1 = g*l1*(m1+2*m2)
    A2 = g*l2*m2

    A3 = -l1*l1*m2*(2*J2+2*l2*l2*m+J2*m1/m2)-J1*(J2+2*l2*l2*m2)
    A4 = -J2*(m+m1+m2)-2*l2*l2*m2*(m+m1+ 1/2*m2)
    A5 = l1*l2*m2*(2*m+m1)
    A6 = l2*m2*(l1*l1*m1+2*l1*l1*m2-J1)
    A7 = l1*l2*m2*(2*m+m1)
    A8 = -J1*(m+m1+m2)-l1*l1*m1*(m*m1/m2 +2*m -m1-2*m2)

    B1 = -l1*l1*m2*(2*J2+2*l2*l2*m+J2*m1/m2)-J1*(J2+2*l2*l2*m2)
    B2 = -J2*l1*(m1-2*m2)-2*l1*l2*l2**m2(m1+m2)
    B3 = l2*m2*(l1*l1*m1+2*l1*l1*m2-J1)

    A=np.array([[0, 0, 0, 1, 0, 0 ],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],
                [0, (A1*A3)/A0, (A1*A6)/A0, 0, 0, 0],[0, (A1*A4)/A0,(A1*A7)/A0, 0, 0, 0],[0, (A1*A5)/A0,(A1*A8)/A0, 0, 0, 0]])
    B=np.array([[0],[0],[0],[B1/A0],[B2/A0],[B3/A0]])
    C=np.array([[1, 0, 0, 0, ,0, 0],[0, 1, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]])
    D=0

    # Discretizar o sistema (Av Euler )
    Ad=np.identity(np.size(A,1))+Ts*A
    Bd=Ts*B
    Cd=C
    Dd=D

    return Ad,Bd,Cd,Dd

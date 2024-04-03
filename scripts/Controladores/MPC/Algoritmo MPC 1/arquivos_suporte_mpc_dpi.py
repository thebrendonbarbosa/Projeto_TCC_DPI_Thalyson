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
        f0 = 0.01
        f1 = 0.007
        f2 = 0.007
        J1 = 0.00083 
        J2 = 0.00083
        Ts = 0.01

        # Parâmetros para a mudança de faixa:: [phi1_ref 0;0 phi2_ref]
            
        # Pesos matriciais para a função custo (devem ser diagonais)
        Q=np.matrix('1 0 0;0 5 0; 0 0 5') # pesos para saídas (todas as amostras, exceto a última)
        S=np.matrix('1 0 0;0 1 0; 0 0 1') # pesos para os ultimos resultados do horizonte de previsao
        R=np.matrix('1') # pesos para entradas (apenas 1 entrada no caso)

        saidas = 3  # número de saídas
        hz = 20 # horizonte de previsao
        #x_dot=20 # velocidade longitudinal pendulo

        # Sinal de referência 
        #r=4  # amplitude
        #f=0.01 # frequencia
        intervalo_de_tempo = 10 # [s] - duração de toda a simulação

        #trajetoria = 1


        # simplificação das matrizes:
        A01 =J1*J2*(m+m1+m2)+J1*l2*l2*m2*(m + m1)+J2*l1*l1*(m*m1+4*m*m2+m1*m2)
        A02 = m*l1*l1*l2*l2*m1*m2
        A0 = A01 + A02

        A1 = g*l1*(m1+2*m2)
        A2 = g*l2*m2

        A3 = -J2*l1*(m1+2*m2)-l1*l2*l2*m1*m2
        A4 = J2*(m+m1+m2)+m*l2*l2*m2 + l2*l2*m1*m2
        A5 = -l1*l2*m2*(2*m+m1)

        A6 = -l2*m2*(J1-l1*l1*m1)
        A7 = -l1*l2*m2*(2*m+m1)
        A8 = J1*(m+m1+m2)+l1*l1*m2*(m*m1 +4*m +m1)

        B1 = J1*(J2 +l2*l2*m2) + J2*(l1*l1*m1+4*l1*l1*m2)+l1*l1*l2*l2*m1*m2
        B2 = -J2*l1*m1-2*J2*l1*m2-l1*l2*l2*m1*m2
        B3 = -J1*l2*m2+l1*l1*l2*m1*m2

        self.constantes = {'A01':A01,'A02':A02,'A0':A0,'A1':A1,'A2':A2,'A3':A3,'A4':A4,'A5':A5,'A6':A6,\
            'A7':A7,'A8':A8,'B1':B1,'B2':B2,'B3':B3, 'm':m, 'm1':m1, 'm2':m2, 'g':g, 'l1':l1, 'l2':l2,\
            'J1':J1, 'J2':J2, 'f0':f0, 'f1':f1, 'f2':f2, 'saidas':saidas,\
            'Ts':Ts, 'hz':hz, 'Q':Q, 'R':R, 'S':S,'intervalo_de_tempo':intervalo_de_tempo}
        
        return None

    def gerador_de_trajetoria(self, t):
        '''Este método cria a trajetória para um carro seguir'''

        Ts = self.constantes['Ts']
        #x_dot = self.constantes['x_dot']
        #trajetoria = self.constantes['trajetoria']

        # Define o comprimento x, depende da velocidade longitudinal do carro
        x=np.linspace(0,t[-1],num=len(t)) #[inicio,fim,qtd. de elementos]

        # Define trajetorias (Valores desejados)
        phi1_r = np.zeros_like(x)
        phi2_r = np.zeros_like(x)
        
        return x,phi1_r,phi2_r

    def espaco_de_estados(self):
        '''Esta função forma as matrizes do espaço de estados e as transforma na forma discreta'''

        # Constantes
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
        C=np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0]])
        D=0
        print(B)
        # Discretizar o sistema (Av Euler )
        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        Dd=D

        return Ad,Bd,Cd,Dd

    def mpc_otimizacao(self, Ad, Bd, Cd, Dd, hz):
        '''Esta função cria as matrizes compactas para Controle Preditivo de Modelo'''
        # db - barra dupla
        # dbt - barra dupla transposta
        # dc - duplo circumflexo

        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug=Dd


        Q=self.constantes['Q']
        S=self.constantes['S']
        R=self.constantes['R']

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)


        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc

    def novos_estados_malha_aberta(self, estados, U1):
        '''Esta função calcula o novo vetor de estado para uma amostra posterior'''

        # Constantes
        Ts = self.constantes['Ts']

        A01 = self.constantes['A01']
        A02 = self.constantes['A02']
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
        #x_dot=self.constantes['x_dot']

        estados_atual=estados
        novos_estados=estados_atual

        X = estados_atual[0]
        phi1 = estados_atual[1]
        phi2 =estados_atual[2]
        X_dot = estados_atual[3]
        phi1_dot = estados_atual[4]
        phi2_dot = estados_atual[5]

        sub_loop = 30   # Fatia Ts em 30 pedaços
        for i in range(0,sub_loop):
            # Computa as derivadas dos estados
            X_dot = X_dot
            phi1_dot = phi1_dot
            phi2_dot = phi2_dot
            X_dot_dot = -((A1*A3)/A0)*phi1 -((A1*A6)/A0)*phi2 + (B1/A0)*U1
            phi1_dot_dot = -((A1*A4)/A0)*phi1 -((A1*A7)/A0)*phi2 - (B2/A0)*U1
            phi2_dot_dot = -((A1*A5)/A0)*phi1 -((A1*A8)/A0)*phi2 + (B3/A0)*U1

            # Atualiza os valores de estado com novas derivadas de estado
            X = X + X_dot*Ts/sub_loop
            phi1 = phi1 + phi1_dot*Ts/sub_loop
            phi2 =phi2 + phi2_dot*Ts/sub_loop
            X_dot = X_dot+ X_dot_dot*Ts/sub_loop
            phi1_dot = phi1_dot + phi1_dot_dot*Ts/sub_loop
            phi2_dot = phi2_dot + phi2_dot_dot*Ts/sub_loop

    # Pegue os últimos estados
        novos_estados[0]= X
        novos_estados[1]=phi1
        novos_estados[2]=phi2
        novos_estados[3]= X_dot
        novos_estados[4]= phi1_dot
        novos_estados[5]= phi2_dot

        return novos_estados

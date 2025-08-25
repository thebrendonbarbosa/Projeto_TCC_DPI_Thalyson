# OFICIAL

import numpy as np
import scripts.Controladores.MPC.algoritmo_mpc_1.MPC_Linear_arquivos_suporte as sdpi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
# --- Módulos de Estilo e Formatação ---
import scienceplots  # Para aplicar estilos de formatação científica aos gráficos.
import latex         # Para renderizar texto e equações usando LaTeX nos gráficos.
plt.style.use(['science'])  # Ativa o estilo de gráfico 'science' para todas as plotagens subsequentes.

# Crie um objeto para as funções de suporte.
suporte=sdpi.DadoSuporteDpi()
constantes=suporte.constantes

# Carrega os valores constantes necessários no arquivo principal
Ts=constantes['Ts']
saidas=constantes['saidas'] # número de saídas (psi, Y)
hz = constantes['hz'] # horizonte de previsao
#x_dot=constantes['x_dot'] # velocidade longitudinal constante
intervalo_de_tempo=constantes['intervalo_de_tempo'] # duração da manobra

# Gera os sinais de referência
t=np.arange(0,intervalo_de_tempo+Ts,Ts) # tempo de 0 a 10 segundos, tempo de amostragem (Ts=0,1 segundo)
r=constantes['r']
f=constantes['f']
X_ref, phi1_ref, phi2_ref = suporte.gerador_de_trajetoria(t,r,f)
sim_length=len(t) # Número de iterações de controle no loop
#refSignals=np.zeros(len(X_ref)*saidas)
xr=np.linspace(0,t[-1],num=len(t)) #[inicio,fim,qtd de elementos]
refSignals=np.zeros(len(xr)*saidas)

# Constroi o vetor de sinal de referência:
# refSignal = [X_ref_0, phi1_ref_0, phi2_ref_0, X_ref_1, phi1_ref_1, phi2_ref_1,... etc.]
k=0
for i in range(0,len(refSignals),saidas):
    refSignals[i]= X_ref[k]
    refSignals[i+1]= phi1_ref[k]
    refSignals[i+2]= phi2_ref[k]
    k=k+1

# Carrega os estados iniciais
# Se você quiser colocar números aqui, certifique-se de que eles sejam flutuantes e não
# inteiros. Isso significa que você deve adicionar um ponto aí.
# Exemplo: Por favor escreva 0. em vez de 0 ( adicione o ponto para ser float)
Xp = 0#.3
phi1 = np.deg2rad(5)
phi2 = -np.deg2rad(5)
Xp_dot = 0.
phi1_dot = 0.
phi2_dot = 0.

estados=np.array([Xp, phi1, phi2, Xp_dot, phi1_dot, phi2_dot])
estadosTotal=np.zeros((len(t),len(estados))) # Acompanhará todos os seus estados durante toda a manobra
estadosTotal[0][0:len(estados)]=estados

#estadosTotal1=np.zeros((len(t),len(estados))) # Acompanhará todos os seus estados durante toda a manobra
#estadosTotal1[0][0:len(estados)]=estados

x_opt_total = np.zeros((len(t),hz))
#x_opt_total1 = np.zeros((len(t),hz))

phi1_opt_total=np.zeros((len(t),hz))
#phi1_opt_total1=np.zeros((len(t),hz))

phi2_opt_total=np.zeros((len(t),hz))
#phi2_opt_total1=np.zeros((len(t),hz))

# Carrega a entrada inicial
U1 = 0.0 # Entrada em t = -1 s 
##U2 = 0

UTotal=np.zeros(len(t)) # Para acompanhar todas as entradas ao longo do tempo
#UTotal1=np.zeros(len(t))

UTotal[0]=U1
#UTotal1[0]=U2


# Para extrair x do x_aug_opt previsto
C_x_opt=np.zeros((hz,(len(estados)+np.size(U1))*hz))
#C_x_opt1=np.zeros((hz,(len(estados)+np.size(U2))*hz))
for i in range(0,hz):
    C_x_opt[i][i+6*(i)]=1
    #C_x_opt1[i][i+6*(i)]=1

# Para extrair phi1_opt do x_aug_opt previsto
C_phi1_opt=np.zeros((hz,(len(estados)+np.size(U1))*hz))
#C_phi1_opt1=np.zeros((hz,(len(estados)+np.size(U2))*hz))
for i in range(1,hz+1):
    C_phi1_opt[i-1][i+6*(i-1)]=1
    #C_phi1_opt1[i-1][i+6*(i-1)]=1

# Para extrair phi2_opt do x_aug_opt previsto
C_phi2_opt=np.zeros((hz,(len(estados)+np.size(U1))*hz))
#C_phi2_opt1=np.zeros((hz,(len(estados)+np.size(U2))*hz))
for i in range(2,hz+2):
    C_phi2_opt[i-2][i+6*(i-2)]=1
    #C_phi2_opt1[i-2][i+6*(i-2)]=1


# Gera as matrizes discretas do espaço de estado
Ad,Bd,Cd,Dd=suporte.espaco_de_estados()

Hdb, Fdbt,Cdb,Adc=suporte.mpc_otimizacao(Ad,Bd,Cd,Dd,hz)

# Inicia o controlador - loops de simulação
k=0
for i in range(0,sim_length-1):

    # Gera o estado atual aumentado e o vetor de referência
    x_aug_t=np.transpose([np.concatenate((estados,[U1]),axis=0)])
    #x_aug_t1=np.transpose([np.concatenate((estados,[U2]),axis=0)])
    # Do vetor refSignals, extraia apenas os valores de referência de sua [amostra atual (AGORA) + Ts] para [AGORA+período do horizonte (hz)]
    # Exemplo: t_now tem 3 segundos, hz = 15 amostras, então dos vetores refSignals, você move os elementos para o vetor r:
    # r=[psi_ref_3.1, Y_ref_3.1, psi_ref_3.2, Y_ref_3.2, ..., psi_ref_4.5, Y_ref_4.5]
    # A cada loop, tudo muda em 0,1 segundo porque Ts=0,1 s
    k=k+saidas
    if k+saidas*hz<=len(refSignals):
        r1=refSignals[k:k+saidas*hz]
        #print(r)
    else:
        r1=refSignals[k:len(refSignals)]
        hz=hz-1
        #print(r)

    if hz<constantes['hz']: # Verifica se hz começa a diminuir
        # Essas matrizes (Hdb,Fdbt,Cdb,Adc) foram criadas anteriormente no início do loop.
        # Eles são constantes durante quase toda a simulação. No entanto,
        # ao final da simulação, o período do horizonte (hz) começa a diminuir.
        # Portanto, as matrizes precisam ser constantemente atualizadas ao final da simulação.
        Hdb,Fdbt,Cdb,Adc=suporte.mpc_otimizacao(Ad,Bd,Cd,Dd,hz)
    
    ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r1),axis=0),Fdbt)
    #ft1=np.matmul(np.concatenate((np.transpose(x_aug_t1)[0][0:len(x_aug_t1)],r1),axis=0),Fdbt)
    #du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
    du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
    #du1=-np.matmul(np.linalg.inv(Hdb1),np.transpose([ft1]))
    #x_aug_opt=np.matmul(Cdb,du)+np.matmul(Adc,x_aug_t)
    x_aug_opt=np.matmul(Cdb,du)+np.matmul(Adc,x_aug_t)
    #x_aug_opt1=np.matmul(Cdb,du1)+np.matmul(Adc,x_aug_t1)
    
    x_opt =  np.matmul(C_x_opt[0:hz,0:(len(estados)+np.size(U1))*hz],x_aug_opt)
    #x_opt1 =  np.matmul(C_x_opt1[0:hz,0:(len(estados)+np.size(U2))*hz],x_aug_opt1)

    phi1_opt=np.matmul(C_phi1_opt[0:hz,0:(len(estados)+np.size(U1))*hz],x_aug_opt)
    #phi1_opt1=np.matmul(C_phi1_opt1[0:hz,0:(len(estados)+np.size(U2))*hz],x_aug_opt1)

    phi2_opt=np.matmul(C_phi2_opt[0:hz,0:(len(estados)+np.size(U1))*hz],x_aug_opt)
    #phi2_opt1=np.matmul(C_phi2_opt1[0:hz,0:(len(estados)+np.size(U2))*hz],x_aug_opt1)
    

    # if hz<4:
    #     print(x_aug_opt)
    x_opt = np.transpose((x_opt))[0]
    #x_opt1 = np.transpose((x_opt1))[0]
    x_opt_total[i+1][0:hz]=x_opt
    #x_opt_total1[i+1][0:hz]=x_opt1

    phi1_opt=np.transpose((phi1_opt))[0]
    #phi1_opt1=np.transpose((phi1_opt1))[0]
    phi1_opt_total[i+1][0:hz]=phi1_opt
    #phi1_opt_total1[i+1][0:hz]=phi1_opt1

    phi2_opt=np.transpose((phi2_opt))[0]
    #phi2_opt1=np.transpose((phi2_opt1))[0]
    phi2_opt_total[i+1][0:hz]=phi2_opt
    #phi2_opt_total1[i+1][0:hz]=phi2_opt1

    # Atualize as entradas reais
    U1=U1+du[0][0]
    #U2=U2+du1[0][0]


   # Estabeleça os limites para as entradas reais (máx.: pi/6 radianos)
    limite_entrada = 50000 #4
    if U1 < - limite_entrada:
        U1 = -limite_entrada
        #U2 = -limite_entrada
    elif U1 > limite_entrada:
        U1=limite_entrada
        #U2=limite_entrada
    else:
        U1=U1
        #U2=U2

    # Acompanhe as entradas conforme passa de t = 0 -> t = 7 segundos
    UTotal[i+1]=U1
    #UTotal1[i+1]=U2

    # Calcula novos estados no sistema de malha aberta (intervalo: Ts/30)
    estados=suporte.novos_estados_malha_aberta(estados,U1)
    #estados1=suporte.novos_estados_malha_aberta(estados,U2)

    estadosTotal[i+1][0:len(estados)]=estados
    #estadosTotal1[i+1][0:len(estados)]=estados1
    # print(i)

################################ LOOP DE ANIMAÇÃO ###############################

"""#print(estadosTotal[-1])
qtd_frame=int(intervalo_de_tempo/Ts)
l1 = constantes['l1']
l2 = constantes['l2']

def update_plot(num):

    hz = constantes['hz']

    #pendulo1.set_data([X_ref[num]+l1*np.sin(estadosTotal[num,1]),X_ref+l1*np.sin(estadosTotal[num,1])])#,
    #                      [0-l1*np.cos(estadosTotal[num,1]),0+l1*np.cos(estadosTotal[num,1])] )
    #pendulo1_body.set_data([-10*l1*np.cos(estadosTotal[num,1]),10*l1*np.cos(estadosTotal[num,1])],
    #                     [-10*l1*np.cos(estadosTotal[num,1]),+10*l1*np.cos(estadosTotal[num,1])] )
   # pendulo1_body_extension.set_data([(l1+40)*np.cos(estadosTotal[num,1]),(l1+40)*np.sin(estadosTotal[num,1])])#,
#                                     [(l1+40)*np.sin(estadosTotal[num,1]),0])
    #pendulo1_front_parte.set_data([l1*np.cos(estadosTotal[num,1])-2*np.cos(estadosTotal[num,1]+UTotal[num]),0],
    #    [l1*np.sin(estadosTotal[num,1])-2*np.sin(estadosTotal[num,1]+UTotal[num]),0])
    #pendulo1_front_parte_extension.set_data([l1*np.cos(estadosTotal[num,1]),0],
    #    [l1*np.sin(estadosTotal[num,1]),0])

    forca_aplicada.set_data(t[0:num], UTotal[0:num])
    #forca_aplicada1.set_data(t[0:num], UTotal1[0:num])

    phi1_angulo.set_data(t[0:num], estadosTotal[0:num,1]*180/np.pi)
    phi2_angulo.set_data(t[0:num], estadosTotal[0:num,2]*180/np.pi)
    x_posicao.set_data(t[0:num], estadosTotal[0:num,0])

    #phi1_angulo1.set_data(t[0:num], estadosTotal1[0:num,1]*180/np.pi)
    #phi2_angulo1.set_data(t[0:num], estadosTotal1[0:num,2]*180/np.pi)
    #x_posicao1.set_data(t[0:num], estadosTotal1[0:num,0])

    if num+hz>len(t):
        hz=len(t)-num
    if num!=0:
        phi1_angulo_predito.set_data(t[num:num+hz], phi1_opt_total[num][0:hz]*180/np.pi)
        phi2_angulo_predito.set_data(t[num:num+hz], phi2_opt_total[num][0:hz]*180/np.pi)
        x_posicao_predito.set_data(t[num:num+hz], x_opt_total[num][0:hz])

        #phi1_angulo_predito1.set_data(t[num:num+hz], phi1_opt_total1[num][0:hz]*180/np.pi)
        #phi2_angulo_predito1.set_data(t[num:num+hz], phi2_opt_total1[num][0:hz]*180/np.pi)
        #x_posicao_predito1.set_data(t[num:num+hz], x_opt_total1[num][0:hz])

    return forca_aplicada,phi1_angulo, phi2_angulo, phi1_angulo_predito, phi2_angulo_predito, x_posicao, x_posicao_predito

# Set up your figure properties
#print(plt.style.available)

#plt.style.use(['_mpl-gallery'])
fig_x=13
fig_y=8
fig=plt.figure(figsize=(fig_x,fig_y),dpi=120,facecolor=(0.8,0.8,0.8))
n=2
m=2
gs=gridspec.GridSpec(n,m)

#n1_start=0
#n1_finish=30
#plt.ylim(-(n1_finish-n1_start)/(n*(fig_x/fig_y)*2),(n1_finish-n1_start)/(n*(fig_x/fig_y)*2))
#lt.ylabel('X-distância [m]',fontsize=15)

# Cria um plot para a entrada de controle
ax0=fig.add_subplot(gs[0,0],facecolor=(0.9,0.9,0.9))
forca_aplicada,=ax0.plot([],[],'-r',linewidth=1,label='Força aplicada [N]')
#forca_aplicada1,=ax0.plot([],[],'-g',linewidth=1,label='Força aplicada [N]')
plt.xlim(0,t[-1])
plt.ylim(np.min(UTotal)-0.1,np.max(UTotal)+0.1)
plt.xlabel('tempo [s]',fontsize=13)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

# Cria um plot para a posicao do carro
ax3=fig.add_subplot(gs[0,1],facecolor=(0.9,0.9,0.9))
x_posicao_referencia=ax3.plot(t,X_ref,'-b',linewidth=0.75,label='$x$ ref. (m)')
x_posicao,=ax3.plot([],[],'-r',linewidth=0.75,label='$x$ (m)')
#x_posicao1,=ax3.plot([],[],'-g',linewidth=0.75,label='$x$ (m)')
x_posicao_predito,=ax3.plot([],[],'m',linestyle='dotted',linewidth=1,label='$x$ predito (m)')
#x_posicao_predito1,=ax3.plot([],[],'g',linestyle='dotted',linewidth=1,label='$x$ predito (m)')
plt.xlim(0,t[-1])
plt.ylim(np.min(estadosTotal[:,0])-0.1,np.max(estadosTotal[:,0])+0.1)
plt.xlabel('tempo [s]',fontsize=12)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

# Cria um plot para o angulo phi1
ax1=fig.add_subplot(gs[1,0],facecolor=(0.9,0.9,0.9))
phi1_angulo_referencia=ax1.plot(t,phi1_ref,'-b',linewidth=0.75,label='$\\phi_1$ Ref. (°)')
phi1_angulo,=ax1.plot([],[],'-r',linewidth=0.75,label= '$\\phi_1$ (°)')
#phi1_angulo1,=ax1.plot([],[],'-g',linewidth=0.75,label= '$\\phi_1$ (°)')
phi1_angulo_predito,=ax1.plot([],[],'m',linestyle='dotted',linewidth=1,label='$\phi_1$ predito (°)')
#phi1_angulo_predito1,=ax1.plot([],[],'g',linestyle='dotted',linewidth=1,label='$\phi_1$ predito (°)')
plt.xlim(0,t[-1])
plt.ylim(np.min(estadosTotal[:,1]*180/np.pi)-0.1,np.max(estadosTotal[:,1]*180/np.pi)+0.1)
plt.xlabel('tempo [s]',fontsize=12)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

# Cria um plot para o angulo phi1
ax2=fig.add_subplot(gs[1,1],facecolor=(0.9,0.9,0.9))
phi2_angulo_referencia=ax2.plot(t,phi2_ref,'-b',linewidth=0.75,label='$\phi_2$ Ref. (°)')
phi2_angulo,=ax2.plot([],[],'-r',linewidth=0.75,label='$\phi_2$ (°)')
#phi2_angulo1,=ax2.plot([],[],'-r',linewidth=0.75,label='$\phi_2$ (°)')
phi2_angulo_predito,=ax2.plot([],[],':m',linewidth=1,label='$\phi_2$ predito (°)')
#phi2_angulo_predito1,=ax2.plot([],[],':m',linewidth=1,label='$\phi_2$ predito (°)')
plt.xlim(0,t[-1])
plt.ylim(np.min(estadosTotal[:,2]*180/np.pi)-0.1,np.max(estadosTotal[:,2]*180/np.pi)+0.1)
plt.xlabel('tempo [s]',fontsize=13)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')


car_ani=animation.FuncAnimation(fig, update_plot,
    frames=qtd_frame ,interval=20,repeat=True,blit=True)
plt.show()"""


#gif_writer = animation.PillowWriter(fps=45)
#car_ani.save('animacao_dpi.gif', writer=gif_writer)




"""

# Plot the world
#plt.plot(t,X_ref,'y',linewidth=2,label='Trajetoria')
plt.plot(t,estadosTotal[:,0],'--b',linewidth=2,label='Posição do Carro')
#plt.plot(t,UTotal[:],'g',linewidth=2,label='Sinal de controle')
plt.plot(t,phi1_ref,'b',linewidth=2,label='Ângulo_ref')
plt.plot(t,estadosTotal[:,1],'--k',linewidth=2,label='Ângulo d pêndulo inferior')
#plt.plot(t,phi2_ref,'b',linewidth=2,label='pêndulo superior_ref')
plt.plot(t,estadosTotal[:,2],'--r',linewidth=2,label='Ângulo pêndulo superior')
plt.xlabel('t-tempo [s]',fontsize=15)
plt.ylabel('x-posição [rad]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')
plt.title('MPC')
#plt.ylim(-13,13) # Scale roads (x & y sizes should be the same to get a realistic picture of the situation)
plt.ylim(-2,2) # Scale roads (x & y sizes should be the same to get a realistic picture of the situation)
plt.show()
"""

#plt.style.use(['science'])
plt.figure(figsize=(8,8))
# Plot the the input delta(t) and the outputs: psi(t) and Y(t)
plt.subplot(3,1,1)
plt.plot(t,X_ref,'--b',linewidth=1.3,label='$x$ Referência')
#plt.plot(t,UTotal[:],'r',linewidth=2,label='ângulo do volante')
plt.plot(t,estadosTotal[:,0],'b',linewidth=1.3,label='$x\;(R=0.01)$')
#plt.plot(t,estadosTotal1[:,0],'r',linewidth=1.3,label='$x$')
plt.xlim([-0.1, 10])
plt.ylabel('Posição (m)',fontsize=11)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')

plt.subplot(3,1,2)
plt.plot(t,phi1_ref,'--r',linewidth=1.3,label='$\\phi_1$ Referência')
plt.plot(t,estadosTotal[:,1]*180/np.pi,'r',linewidth=1.3,label='$\\phi_1\;(R=0.01)$')
#plt.plot(t,estadosTotal1[:,1]*180/np.pi,'b',linewidth=1.2,label='$\\phi_1$')
plt.xlim([-0.1, 10])
plt.ylabel('Ângulo[°]',fontsize=11)
plt.grid(True)
#plt.legend(loc='center right',fontsize='small')
plt.legend(loc='upper right',fontsize='small')

plt.subplot(3,1,3)
plt.plot(t,phi2_ref,'--k',linewidth=1.3,label='$\\phi_2$ Referência')
plt.plot(t,estadosTotal[:,2]*180/np.pi,'k',linewidth=1.3,label='$\\phi_2\;(R=0.01)$')
#plt.plot(t,estadosTotal1[:,2]*180/np.pi,'g',linewidth=1.2,label='$\\phi_2$')
plt.xlim([-0.1, 10])
plt.xlabel('Tempo (s)',fontsize=11)
plt.ylabel('Ângulo[°]',fontsize=11)
plt.grid(True)
#plt.legend(loc='center right',fontsize='small')
plt.legend(loc='upper right',fontsize='small')
plt.savefig('Cap4_MPC_Linear_Caso2_Estadoshz70.png', dpi=400,bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,3))
plt.subplot(1,1,1)
plt.plot(t,UTotal[:],'g',linewidth=1.3,label='Sinal u $(R=0.01)$')
plt.xlim([-0.1, 10])
plt.xlabel('Tempo (s)',fontsize=11)
plt.ylabel('Sinal de controle (N) ',fontsize=11)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')
plt.savefig('Cap4_MPC_Linear_Caso2_SinalUhz70.png', dpi=400,bbox_inches='tight')
plt.show()
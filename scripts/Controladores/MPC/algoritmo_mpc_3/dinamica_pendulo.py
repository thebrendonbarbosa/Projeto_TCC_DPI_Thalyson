import numpy as np
import control

import numpy as np
import control

"""
Dynamic equations '
"""

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
Ts = 0.02
l = 0.45

A01 =-J1*J2*(m+m1+m2)-2*m2*l2*l2*J1*(m+m1+(1/2)*m2)-J2*l1*l1*(m*m1+2*m*m2-m1*m2-2*m2*m2)
A02 =  -2*l1*l1*l2*l2*m2*(m*m1-3/2 *m1*m2 -m2*m2)
A0 = A01 + A02

A1 = g*l1*(m1+2*m2)
A2 = g*l2*m2

A3 = -J2*l1*(m1+2*m2)-2*l1*l2*l2*m2*(m1+m2)
A4 = -J2*(m+m1+m2)-2*l2*l2*m2*(m+m1+ 1/2*m2)
A5 = l1*l2*m2*(2*m+m1)
A6 = l2*m2*(l1*l1*m1+2*l1*l1*m2-J1)
A7 = l1*l2*m2*(2*m+m1)
A8 = -J1*(m+m1+m2)-l1*l1*m2*(m*m1/m2 +2*m -m1-2*m2)

B1 = -l1*l1*m2*(2*J2+2*l2*l2*m+J2*m1/m2)-J1*(J2+2*l2*l2*m2)
B2 = -J2*l1*(m1+2*m2)-2*l1*l2*l2*m2*(m1+m2)
B3 = l2*m2*(l1*l1*m1+2*l1*l1*m2-J1)


A=np.array([[0, 0, 0, 1, 0, 0 ],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1],
            [0, (A1*A3)/A0, (A2*A6)/A0, 0, 0, 0],[0, (A1*A4)/A0,(A2*A7)/A0, 0, 0, 0],[0, (A1*A5)/A0,(A2*A8)/A0, 0, 0, 0]])
B=np.array([[0],[0],[0],[B1/A0],[B2/A0],[B3/A0]])
C=np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0]])
D=0


sys = control.StateSpace(A, B, C, D)
sys_discrete = control.c2d(sys, Ts, method='zoh')

A_zoh = np.array(sys_discrete.A)
B_zoh = np.array(sys_discrete.B)
print(sys)
import numpy as np
import matplotlib
font = {'size': 11.0}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.constants as const

######################################################################################################################################################
### Untersuchung der drei Moden

def f(x, a, b, c):
    return a*x**2 + b*x + c

V_3moden = ([70, 85, 92, 122, 140, 150, 205, 220, 230])
A_3moden = ([0, 0.96, 0, 0, 1.24, 0, 0, 1.08, 0])

params_1,cov = curve_fit(f, V_3moden[0:3], A_3moden[0:3])
params_2,cov = curve_fit(f, V_3moden[3:6], A_3moden[3:6])
params_3,cov = curve_fit(f, V_3moden[6:9], A_3moden[6:9])

for name, param in zip(('a_1', 'b_1', 'c_1'), params_1):
    print(r'{0}:  {1:.4f}'.format(name, param))

for name, param in zip(('a_2', 'b_2', 'c_2'), params_2):
    print(r'{0}:  {1:.4f}'.format(name, param))

for name, param in zip(('a_3', 'b_3', 'c_3'), params_3):
    print(r'{0}:  {1:.4f}'.format(name, param))

plt.figure(figsize=(6.4,3.96))#,dpi=300)
plt.plot(V_3moden[0:3]
         ,A_3moden[0:3]
         ,'x'
         ,color='blue'
         ,ms = 7
         ,mew= 1.3
         ,label = '1. Modus'
        )
plt.plot(V_3moden[3:6]
         ,A_3moden[3:6]
         ,'x'
         ,color='red'
         ,ms = 7
         ,mew= 1.3
         ,label = '2. Modus'
        )
plt.plot(V_3moden[6:9]
         ,A_3moden[6:9]
         ,'x'
         ,color='green'
         ,ms = 7
         ,mew= 1.3
         ,label = '3. Modus'
        )
x = np.linspace(V_3moden[0], V_3moden[2])
plt.plot(x
        ,f(x, *params_1)
        ,'--'
         ,color='blue'
        )
x = np.linspace(V_3moden[3], V_3moden[5])
plt.plot(x
        ,f(x, *params_2)
        ,'--'
         ,color='red'
        )
x = np.linspace(V_3moden[6], V_3moden[8])
plt.plot(x
        ,f(x, *params_3)
        ,'--'
         ,color='green'
        )
plt.xlabel(r'$U \;$[V]')
plt.ylabel(r'Leistung')
plt.legend(loc = 'best')
plt.ylim(bottom=-0.05, top=1.6)
plt.tight_layout()
plt.savefig('plots/3moden.pdf')
# plt.show()
plt.close()

### Elektronische Bandbreiten und Abstimm-Empfindlichkeiten berechnen
f_1 = unp.uarray([8985, 8978, 8982], [4, 4, 4]) *10**6
f_2 = unp.uarray([9018, 9030, 9038], [4, 4, 4]) *10**6
B = f_2 - f_1

V_1 = unp.uarray([210, 129, 78], [5, 5, 5])
V_2 = unp.uarray([239, 148, 90], [5, 5, 5])
E = B / (V_2 - V_1)

print('')
for name, b in zip(('1. ', '2. ', '3. '), B):
    print(r'{0}Bandbreiten:  {1:.2e}'.format(name, b))

for name, e in zip(('1. ', '2. ', '3. '), E):
    print(r'{0}Abstimmempfind.:  {1:.2e}'.format(name, e))

### Abstand zwischen Resonator und Reflektor und Moden bestimmen
f_0 = unp.uarray([9000, 9004, 9010], [2, 2, 3]) *10**6
U_c = unp.uarray([220, 140, 85], [5, 5, 5])
U_b = 300
m_e = const.m_e
e = const.e
L_12 = 1 / (np.sqrt(8 * U_b * m_e/e) * ((f_0[1]) / (U_b + U_c[1]) - (f_0[0]) / (U_b + U_c[0])))
L_23 = 1 / (np.sqrt(8 * U_b * m_e/e) * ((f_0[2]) / (U_b + U_c[2]) - (f_0[1]) / (U_b + U_c[1])))
L = (L_12 + L_23) / 2

print('')
print(r'Abstand:  {:.4e}'.format(L))

n = np.sqrt(8 * U_b * m_e/e) * L * f_0 / (U_b + U_c) - 3/4

for name, param in zip(('n_1', 'n_2', 'n_3'), n):
    print(r'{0}:  {1:.3f}'.format(name, param))


######################################################################################################################################################
### Genauere Frequenzbestimmung (Rechnung mit Werten in mm)

b = (23.5 + 25.4)
lam_g = ufloat(b, 0.2)
a = ufloat(22.860, 0.046)
c = 3 * 10**(11)

f = ufloat(8996, 0.5) * 10**(6)
f_lam = c * unp.sqrt((1 / lam_g)**2 + (1 / (2*a))**2)
v_phase = f * (lam_g * 10**(-3))
v_phase_lam = f_lam * (lam_g * 10**(-3))
print('')
print(r'Frequenz:  {:.4e}'.format(f_lam))
print(r'Phasengeschw.:  {:.3e}'.format(v_phase * 3.6))
print(r'Phasengeschw. lam:  {:.3e}'.format(v_phase_lam * 3.6))
print(r'Phasengeschw. in c:  {:.3f}'.format(v_phase / c*10**(3)))
print(r'Phasengeschw. lam in c:  {:.3f}'.format(v_phase_lam / c*10**(3)))

######################################################################################################################################################
### Dämpfungskurve aus an SWR-Meter und in der Eichkurve vom Hersteller abgelesenen Werte in dB

def exp(x, a, b, c):
    return a * np.exp(b*x) + c

x = unp.uarray([0, 1.085, 1.475, 1.780, 2.040, 2.250], [0, 0.01, 0.01, 0.01, 0.01, 0.01])
x_lin = np.linspace(noms(x)[0], noms(x)[-1], 200)
A_SWR = np.array([0, 2, 4, 6, 8, 10])
A_eich = unp.uarray([0, 2.7, 4.2, 6.5, 9.0, 10.5], [0, 1, 1, 1, 1, 1])

params_A_SWR,cov = curve_fit(exp, noms(x), noms(A_SWR))
params_A_eich,cov = curve_fit(exp, noms(x), noms(A_eich))

print('')
for name, e in zip(('a', 'b', 'c'), params_A_SWR):
    print(r'SWR-Messung {0}:  {1:.4f}'.format(name, e))

for name, e in zip(('a', 'b', 'c'), params_A_SWR):
    print(r'Eichkurve {0}:  {1:.4f}'.format(name, e))

plt.figure(figsize=(6.4,3.96))#,dpi=300)
plt.plot(noms(x)
         ,A_SWR
         ,'x'
         ,color='blue'
         ,ms = 6
         ,mew= 1.3
         ,label = 'SWR-Meter Messwerte'
        )
plt.plot(x_lin
         ,exp(x_lin, *params_A_SWR)
         ,'--'
         ,color='blue'
        )
plt.plot(noms(x)
         ,noms(A_eich)
         ,'x'
         ,color='red'
         ,ms = 6
         ,mew= 1.3
         ,label = 'Eichkurve'
        )
plt.plot(x_lin
         ,exp(x_lin, *params_A_eich)
         ,'--'
         ,color='red'
        )
plt.xlabel(r'Schraubeneinstellung $x \;$[mm]')
plt.ylabel(r'Dämpfung $P \;$[dB]')
plt.legend(loc = 'best', markerscale=1.2)
# plt.ylim(bottom=-0.05, top=1.6)
plt.tight_layout()
plt.savefig('plots/daempfung.pdf')
# plt.show()
plt.close()

### 3 dB-Methode
d_1 = ufloat(91.2, 0.05)
d_2 = ufloat(92.9, 0.05)
S_3db = unp.sqrt(1 + 1 / (unp.sin(np.pi * (d_1 - d_2) / lam_g))**2)
# S_3db = lam_g / (np.pi * (d_2 - d_1))

print('')
print(r'S_3dB:  {:.3f}'.format(S_3db))

### Abschwächer-Methode
A_1 = ufloat(20, 1)
A_2 = ufloat(42, 1)
S_schwach = 10**((A_2 - A_1) / 20)

print('')
print(r'S_schwach:  {:.3f}'.format(S_schwach))

import numpy as np
import matplotlib
font = {'size': 11.0}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import scipy.constants as const


if False:
    U, phi, nu = np.genfromtxt('data/test/data1.txt'
                  ,unpack=True
                  ,delimiter = ","
                  ,skip_header = 1
                   )
df1 = pd.read_csv('data/data1.txt'
                 ,header =[0])
df2 = pd.read_csv('data/data2.txt'
                 ,header =[0])
df3 = pd.read_csv('data/data3.txt'
                 ,header =[0])
df_ph1 = pd.read_csv('data/test/data1.txt'
                 ,header =[0])
df_ph2 = pd.read_csv('data/test/data2.txt'
                 ,header =[0])


def compute_V(df):
    """Caculates the Amplification for given Currents.

        Parameters
        ----------
        U_E: float
            The Input Current
        U_A: float
            The Output Current

        Returns
        -------
        V: float
            The Amplification
    """
    if 'V' not in df.columns:
        df['V'] = df['U_A'] / df['U_E']
    return df['V']

def f(x,a,b):
    """Calculates an exponential

        Parameters
        ----------
        x: float
            X-Value
        a: float
            Faktor
        b: float
            Exponent

        Returns
        -------
        f(x): float
            The value of the Funktion
    """
    
    return a*x**b

def horizontale(x,a):
    return a


lim1 = 12

x_val1 = df1['nu'][lim1:].to_numpy()
y_val1 = compute_V(df1[lim1:]).to_numpy()
params1,cov = curve_fit(f, x_val1, y_val1)
errors1 = np.sqrt(np.diag(cov))
params1_err = unp.uarray(params1, errors1)

x_val_h = df1['nu'][:lim1].to_numpy()
y_val_h = compute_V(df1[:lim1]).to_numpy()
params_h,cov = curve_fit(horizontale, x_val_h, y_val_h)
errors_h = np.sqrt(np.diag(cov))
params_h_err = unp.uarray(params_h, errors_h)

plt.figure(figsize=(6.4,3.96))#,dpi=300)
plt.plot(df1['nu'][:lim1]
         ,compute_V(df1[:lim1])
         ,'.'
         ,ms = 8
         ,label = 'Plateaubereich'
        )
x = np.linspace(0, df1['nu'][lim1], num=10)
y = np.full(10, params_h)
# plt.hlines(params_h, 0, df1['nu'][lim1], color='grey', linestyles='--', label='Leerlaufverstärkung')
plt.plot(x
         ,y
         ,'--'
         ,color='grey'
         ,label = 'Leerlaufverstärkung'
        )
plt.plot(df1['nu'][lim1:]
         ,compute_V(df1[lim1:])
         ,'.'
         ,ms = 8
         ,label = 'gefittete Messwerte'
        )
plt.plot(df1['nu']
            ,f(df1['nu'], *params1)
            ,'--'
            ,label = 'Fit'
        )
plt.xlabel(r'$\nu \; [\mathrm{kHz}]$')
plt.ylabel(r'$V`$')
plt.legend(loc = 'best')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig('plots/linearVerstaerker.pdf')
# plt.show()
plt.clf()

# Parameter des 1. Fits:
# mitt = np.sum(compute_V(df1[:lim1]).to_numpy()) / df1['nu'][:lim1].size   #Mittelwert über konstante Messwerte
# print('Leerlaufverstärkung: ',mitt)
print('\n')
llv = (params_h_err[0] * 10000) / (10000 - params_h_err[0] * 1)
print(r'Leerlaufverstärkung: {:.8f}'.format(llv))
for name, param in zip(('a','b'), params1_err):
    print(r'{0}:  {1:.8f}'.format(name, param))
print('\n')

# Grenzfrequenz ausrechnen
a = params1[0]
b = np.abs(params1[1])
a_err = errors1[0]
b_err = errors1[1]
nu_grenz = np.log(np.sqrt(2)/a)
# fehler = np.sqrt((1/(a * np.log(b)))**2 * a_err**2 + (np.log(np.sqrt(2)/a) / b * (np.log(b))**2)**2 * b_err**2)
# print(fehler)

# Bandbreiten-Verstärkungs-Produkt ausrechnen
nu_grenz = 375.0871
produkt = nu_grenz * params_h_err[0]
print(r'Bandbreitenprodukt: {:.8f}'.format(produkt))

##############################################################################################################

plt.figure(figsize=(6.4,3.96))#,dpi=300)
plt.plot(df_ph1['nu']
         ,df_ph1['phi']
         ,'.'
         ,ms = 8
         ,label = 'Messreihe 1'
        )
plt.plot(df_ph2['nu']
         ,df_ph2['phi']
         ,'.'
         ,ms = 8
         ,label = 'Messreihe 2'
        )
plt.xlabel(r'$\nu \; [\mathrm{kHz}]$')
plt.ylabel(r'Phase $\Phi \; [°]$')
plt.legend(loc = 'best')
# plt.yscale('log')
# plt.xscale('log')
plt.tight_layout()
plt.savefig('plots/linearVerstaerkerPhase.pdf')
# plt.show()
plt.clf()


##############################################################################################################
lim2 = 5

x_val2 = df2['nu'][:lim2].to_numpy()
# y_val2 = compute_V(df2[lim2:]).to_numpy()
y_val2 = df2['U_A'][:lim2].to_numpy()
params2, cov = curve_fit(f, x_val2, y_val2)
errors2 = np.sqrt(np.diag(cov))
params2_err = unp.uarray(params2, errors2)

plt.figure(figsize=(6.4,3.96))#,dpi=300)
plt.plot(df2['nu'][lim2:]
         ,df2['U_A'][lim2:]
         ,'.'
         ,ms = 8
         ,label = 'nicht genutzte Messwerte'
        )
plt.plot(df2['nu'][:lim2]
         ,df2['U_A'][:lim2]
         ,'.'
         ,ms = 8
         ,label = 'gefittete Messwerte'
        )
plt.plot(df2['nu']
            ,f(df2['nu'], *params2)
            ,'--'
            ,label = 'Fit'
        )
plt.xlabel(r'$\nu \; [\mathrm{kHz}]$')
plt.ylabel(r'$U_A \; [\mathrm{V}]$')
plt.legend(loc = 'best')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig('plots/integrator.pdf')
# plt.show()

for name, param in zip(('a','b'), params2_err):
    print(r'{0}:  {1:.8f}'.format(name, param))
print('\n')


##############################################################################################################
lim3 = 7

x_val3 = df3['nu'][:lim3].to_numpy()
# y_val3 = compute_V(df3[lim3:]).to_numpy()
y_val3 = df3['U_A'][:lim3].to_numpy()
params3, cov = curve_fit(f, x_val3, y_val3)
errors3 = np.sqrt(np.diag(cov))
params3_err = unp.uarray(params3, errors3)

plt.figure(figsize=(6.4,3.96))#,dpi=300)
plt.plot(df3['nu'][lim3:]
         ,df3['U_A'][lim3:]
         ,'.'
         ,ms = 8
         ,label = 'nicht genutzte Messwerte'
        )
plt.plot(df3['nu'][:lim3]
         ,df3['U_A'][:lim3]
         ,'.'
         ,ms = 8
         ,label = 'gefittete Messwerte'
        )
plt.plot(df3['nu']
            ,f(df3['nu'], *params3)
            ,'--'
            ,label = 'Fit'
        )
plt.xlabel(r'$\nu \; [\mathrm{kHz}]$')
plt.ylabel(r'$U_A \; [\mathrm{V}]$')
plt.legend(loc = 'best')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig('plots/differenzierer.pdf')
# plt.show()

for name, param in zip(('a','b'), params3_err):
    print(r'{0}:  {1:.8f}'.format(name, param))
print('\n')

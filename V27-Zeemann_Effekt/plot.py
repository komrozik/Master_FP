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

B,I = np.genfromtxt('data/data1.txt'
                  ,unpack=True
                  ,delimiter = ","
                  ,skip_header = 2
                   )
df = pd.read_csv('data/data1.txt'
                 ,header =[1])


def f(x,a,b,c,d):
    return a*x+b*x**2+c*x**3+d

def g(x,a,b):
    return a*x+b

params1,cov = curve_fit(f,df['I'],df['B'])
errors = np.sqrt(np.diag(cov))
params1_err = unp.uarray(params1,errors)
err = params1_err[0]*df['I']+params1_err[1]*df['I']**2+params1_err[2]*df['I']**3+params1_err[3]

params2,cov = curve_fit(g,df['I'][0:25],df['B'][0:25])
errors = np.sqrt(np.diag(cov))
params2_err = unp.uarray(params2,errors)
err = params2_err[0]*df['I']+params2_err[1]

x = np.linspace(0,8,1000)

plt.figure(figsize=(6.4,3.96),dpi=300)
plt.plot(df['I']
         ,df['B']
         ,'.'
         ,ms = 3
         ,label = 'Messdaten'
        )
if True:
    plt.vlines(ymin = 0
           ,ymax = 468
           ,x = 5
           ,color = 'k'
           ,alpha = 0.5
          )
    plt.hlines(xmin = 0
           ,xmax = 5
           ,y = 468
           ,color = 'k'
           ,alpha = 0.5
          )
plt.plot(x
         ,f(x,*params1)
         ,alpha = 0.5
         ,label = 'Fit eines Polynom 3. Ordnung'
        )
plt.plot(x
         ,g(x,*params2)
         ,alpha = 0.5
         ,label = 'Fit einer linearen Funktion'
        )
plt.xlabel('B in mT')
plt.ylabel(f'I in Ampere')
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig("plots/B_Feld.pdf")
plt.close()
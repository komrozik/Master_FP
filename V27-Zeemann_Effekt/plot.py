import numpy as np
import matplotlib
font = {'size': 11.0}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from uncertainties import ufloat
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

print('Polynom 3.Ordnung Fit:')
print(f'Parameter 3.Ordnung : {params1_err[2]}')
print(f'Parameter 2.Ordnung : {params1_err[1]}')
print(f'Parameter 1.Ordnung : {params1_err[0]}')
print(f'Parameter 0.Ordnung : {params1_err[3]}')

print('Linearer Fit:')
print(f'Die Steigung ist {params2_err[0]} und der y-Achsenabschnitt ist {params2_err[1]}')

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



df1 = pd.read_csv('data/data_blau_senkrecht.txt'
                 ,header =[1])
df2 = pd.read_csv('data/data_rot_senkrecht.txt'
                 ,header =[1])
df3 = pd.read_csv('data/data_blau_parallel.txt'
                 ,header =[1])

def delta_lambda(lam,n,d):
    """Caculates the Dispersion area from the thickness, the wavelength and the difraction index

        Parameters
        ----------
        lam: float
            The wavelength lambda in meters
        n: float
            The diffraction index of the material
        d: float
            The thickness of the material in meters.

        Returns
        -------
        delta_lambda: float
            The Disperison region for the material with the specified wavelength.
    """
    delta_lambda = lam**2/(2*d)*np.sqrt(1/(n**2-1))
    return delta_lambda

def add_del_lambda(df,lam,n):
    """Adds the Dispersion region to the given data frame.

        Parameters
        ----------
        df: pandas data frame
            The data frame with the information.
        lam: float
            The wavelength lambda in meters
        n: float
            The diffraction index of the material
            

        Returns
        -------
        df: pandas data frame
            Data Frame with the added Disperion region
    """
    df['del_lambda'] = 0.5*(df['del_s']/df['delta_s'])*delta_lambda(lam,n,0.004)
    df['del_lambda_err'] = 0.5*df['del_lambda']*np.sqrt((3/df['delta_s'])**2+(3/df['del_s'])**2)
    return df

def get_landre(del_lam,B,lam):
    """Calculates the landre factor.

        Parameters
        ----------
        del_lam: float
            The wavelength difference in meters
        lam: float
            The wavelength lambda in meters
        B: float
            The magnetic field strength in Tesla
            

        Returns
        -------
        g: float
            The Landre Factor
    """
    g = del_lam*(const.h*const.c)/(const.physical_constants['Bohr magneton'][0]*B*lam**2)
    return g

B_feld = f(7.8,*params1)*10**(-3)

df1 = add_del_lambda(df1,6.438*10**(-7),1.4567)
del_lambda1 = df1['del_lambda'].mean(axis = 0)
del_lambda1_err = 1/np.sqrt(len(df1['del_lambda_err']))*np.sqrt(sum(df1['del_lambda_err']**2))
del_lambda1_unc = ufloat(del_lambda1, del_lambda1_err)

g1 = get_landre(del_lambda1_unc,B_feld,6.438*10**(-7))

df2 = add_del_lambda(df2,4.8*10**(-7),1.4635)
del_lambda2 = df2['del_lambda'].mean(axis = 0)
del_lambda2_err = 1/np.sqrt(len(df2['del_lambda_err']))*np.sqrt(sum(df2['del_lambda_err']**2))
del_lambda2_unc = ufloat(del_lambda2, del_lambda2_err)

g2 = get_landre(del_lambda2_unc,B_feld,4.8*10**(-7))

df3 = add_del_lambda(df3,6.438*10**(-7),1.4567)
del_lambda3 = df3['del_lambda'].mean(axis = 0)
del_lambda3_err = 1/np.sqrt(len(df3['del_lambda_err']))*np.sqrt(sum(df3['del_lambda_err']**2))
del_lambda3_unc = ufloat(del_lambda3, del_lambda3_err)

g3 = get_landre(del_lambda3_unc,1,6.438*10**(-7))

print(f"Für das blaue senkrecht Licht: del = {del_lambda1_unc} g = {g1}")
print(f"Für das blaue parallel Licht: del = {del_lambda3_unc} g = {g3}")
print(f"Für das rote Licht: del = {del_lambda2_unc} g = {g2}")
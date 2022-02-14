import numpy as np
import matplotlib
font = {'size': 11.0}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
# import scipy.constants as const



################################################################################################################################
### Eingenschaften des Röntgenstrahls bestimmen

alpha_detektor, I_detektor  = np.genfromtxt('data/Detector_scan.UXD'
                ,unpack=True
                ,delimiter = ""
                ,skip_header = 0
                )

def f(x,A,x_0,c):
    """Eine Gaußsche Glocke zum fitten an Messwerte.

        Parameters
        ----------
        x: float
            x-value
        A: float
            height of the peak
        x_0: float
            x-position of the center of the peak
        c: float
            standard deviation (controls width of peak)

        Returns
        -------
        f(x): float
            The value of the Funktion
    """
    
    return A * np.exp(-(x - x_0) ** 2 / (2 * c ** 2))


params_detektor, cov = curve_fit(f, alpha_detektor, I_detektor, p0=[max(I_detektor), 0, 0.05])
errors_detektor = np.sqrt(np.diag(cov))
params_detektor_err = unp.uarray(params_detektor, errors_detektor)

FWHM = 2 * params_detektor_err[2] * np.sqrt(np.log(4))

plt.figure(figsize=(7.2,4.4))#,dpi=300)
plt.plot(alpha_detektor
            ,I_detektor
            ,marker='x'
            ,markersize=5
            ,linestyle=''
            ,label = 'Messwerte'
            ,mew=1.5
        )
x = np.linspace(alpha_detektor[0], alpha_detektor[-1], num=300)
plt.plot(x
            ,f(x, *params_detektor)
            ,linestyle='-'
            ,label = 'Fit'
        )
plt.axvline(params_detektor[1] - noms(FWHM)/2, 0, 10**6,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6
            )
plt.axvline(params_detektor[1] + noms(FWHM)/2, 0, 10**6,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6
            )
plt.hlines(params_detektor[0]/2, params_detektor[1] - noms(FWHM)/2, params_detektor[1] + noms(FWHM)/2,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label='Halbwertsbreite'
            )
plt.xlabel(r'$\alpha \; [°]$')
plt.ylabel(r'Intensität')
plt.legend(loc = 'best', markerscale=1.5)
plt.tight_layout()
plt.savefig('plots/DetektorScan.pdf')
# plt.show()
plt.close()


# for i in np.arange(len(I_detektor)):
#     print(alpha_detektor[i], I_detektor[i])

for name, param in zip(('A','x_0','c'), params_detektor_err):
    print(r'{0}:  {1:.4f}'.format(name, param))
# print('\n')
print(r'Halbwertsbreite: {:.4f}'.format(FWHM))


################################################################################################################################
### Geometriefaktor bestimmen

z1, I_z1  = np.genfromtxt('data/z_scan_05.UXD'
                ,unpack=True
                ,delimiter = ""
                ,skip_header = 0
                )
I_z1 = I_z1 / max(I_z1)

plt.figure(figsize=(7.2,4.4))#,dpi=300)
plt.plot(z1
            ,I_z1
            ,linestyle='-'
            ,linewidth=1.5
            ,label = 'Messwerte'
        )
plt.axvline(0.015, 0, 10**6,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label=r'Strahlbreite $d$}'
            )
plt.axvline(0.225, 0, 10**6,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6
            )
plt.xlabel(r'$z \; [\mathrm{mm}]$')
plt.ylabel(r'$I \;/\; I_{\mathrm{max}}$')
plt.legend(loc = 'best', markerscale=1.5)
plt.tight_layout()
plt.savefig('plots/Strahlbreite.pdf')
# plt.show()
plt.close()

######################################

alpha_rock1, I_rock1  = np.genfromtxt('data/rocking_scan_2theta_0.UXD'
                ,unpack=True
                ,delimiter = ""
                ,skip_header = 0
                )
I_rock1 = I_rock1 / max(I_rock1)

plt.figure(figsize=(7.2,4.4))#,dpi=300)
plt.plot(alpha_rock1
            ,I_rock1
            ,linestyle='-'
            ,linewidth=1.5
            ,label = 'Messwerte'
        )
plt.axvline(-0.61, 0, 10**6,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label=r'$\alpha_{\mathrm{g,1}} = 0,61°$'
            )
plt.axvline(0.63, 0, 10**6,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label=r'$\alpha_{\mathrm{g,2}} = 0,63°$'
            )
plt.xlabel(r'$\alpha \; [°]$')
plt.ylabel(r'$I \;/\; I_{\mathrm{max}}$')
plt.legend(loc = 'best', markerscale=1.5)
plt.tight_layout()
plt.savefig('plots/Geometriewinkel.pdf')
# plt.show()
plt.close()


################################################################################################################################
### Reflektivität für eine Schicht
alpha, I  = np.genfromtxt('data/messung.UXD'
                ,unpack=True
                ,delimiter = ""
                ,skip_header = 0
                )
alpha_diff, I_diff  = np.genfromtxt('data/diffuser.UXD'
                ,unpack=True
                ,delimiter = ""
                ,skip_header = 0
                )
I_0 = 884421*5         
R_exp = (I - I_diff) / I_0

def fresnelreflectivity(alpha):
    """Die Reflektivität für einen Röntgenstrahl der Wellenlänge lam bei
    sehr kleinem Einfallswinkeln an einer Silizium-Oberfläche berechnen.

        Parameters
        ----------
        alpha: float
            angle at which the reflectivity is computed

        Returns
        -------
        R(alpha): float
           reflectivity at a certain angle alpha
    """

    alpha_c = 0.223
    beta = 1.73*10**(-7)

    A_plus = np.sqrt(np.sqrt((alpha**2 - alpha_c**2)**2 + 4*beta) + (alpha**2 - alpha_c**2)) / np.sqrt(2)
    A_minus = np.sqrt(np.sqrt((alpha**2 - alpha_c**2)**2 + 4*beta) - (alpha**2 - alpha_c**2)) / np.sqrt(2)

    return ((alpha - A_plus)**2 + A_minus**2) / ((alpha + A_plus)**2 + A_minus**2)
    

plt.figure(figsize=(7.2,4.4))#,dpi=300)
plt.plot(alpha
            ,R_exp
            ,linestyle='-'
            ,linewidth=1.5
            ,label = r'Messwerte $R_{\mathrm{exp}}$'
        )
plt.plot(alpha
            ,fresnelreflectivity(alpha)
            ,linestyle='--'
            ,linewidth=1.5
            ,label = 'Berechnete Fresnelreflektivität\neiner idealen Si-Schicht'
        )
plt.xlabel(r'$\alpha \; [°]$')
plt.ylabel(r'$R$')
plt.legend(loc = 'best', markerscale=1.5)
plt.yscale('log')
plt.xlim(-0.05, 1.4)
plt.tight_layout()
plt.savefig('plots/Reflektivitaetskurve1.pdf')
# plt.show()
plt.close()

################################################################################################################################
### Reflektivität mit Geometriefaktor korrigiert

i=0
while alpha[i] < 0.61:
    i += 1
print('\n'+'Grenzindex ab dem die Winkel \n größer als der Geometriewinkel sind: {}'.format(i))
G = np.concatenate((20 * np.sin(alpha[:i+1]*np.pi/180) / 0.21, np.ones(len(alpha)-i-1)))
with np.errstate(divide='ignore'):#, invalid='ignore'):
    R = R_exp / G

######################################
### Schichtdicke d bestimmen
lam = 1.54*10**(-10)
peaks_untere_grenze = 70
peaks_obere_grenze = 180
indices = find_peaks(R[peaks_untere_grenze:peaks_obere_grenze], distance=5, width=1)[0]

j = indices[0]
alpha_diff = np.array([])
for i in indices[1:]:
    alpha_diff = np.append(alpha_diff, (i - j) * 0.005)
    j = i
alpha_diff_mitt_noms = np.sum(alpha_diff) / len(indices[1:])
alpha_diff_mitt_stds = 1/len(indices[1:]) * np.sqrt(np.sum((alpha_diff - alpha_diff_mitt_noms)**2))
alpha_diff_mitt = ufloat(alpha_diff_mitt_noms, alpha_diff_mitt_stds)
print('Mittl. Abstand zwischen Maxima: {:.4f}'.format(alpha_diff_mitt))
d = lam / (2 * alpha_diff_mitt * np.pi/180)
print('Schichtdicke: {:.4e}'.format(d))

######################################
### Reflektivität mit Parratt-Algorithmus berechnen + Kritische Winkel für Polysterol und Silizium berechnen

z_1 = 0

# d = 8.6*10**(-8)
d = 8.6*10**(-8)
delta_2 = 1 * 10**(-6)
delta_3 = 7 * 10**(-6)
sigma_1 = 8 * 10**(-10)
sigma_2 = 6.8 * 10**(-10)

anfangswerte = ([d, delta_2, delta_3, sigma_1, sigma_2])

def parratt_algorithm(alpha, d, delta_2, delta_3, sigma_1, sigma_2):
    """Berechnet die Gesamtreflektivität an einem 2-Schichtensystem.

        Parameters
        ----------
        alpha: float
            angle at which the reflectivity is computed
        d: float
            thickness of the thin polysterol layer
        delta_2: float
            delta_2 = 1 - n_2, where n_2 is the refractive index of polysterol
        delta_3: float
            delta_3 = 1 - n_3, where n_2 is the refractive index of silizium
        sigma_2: float
            roughness of the polysterol layer
        sigma_3: float
            roughness of the silizium layer

        Returns
        -------
        R: float
            entire reflectivity for the two-layer-system
    """
    lam = 1.54*10**(-10)
    k = 2*np.pi/lam
    mu_2 = 400
    mu_3 = 14100

    n_1 = 1
    n_2 = 1 - delta_2 + 1j*lam/(4*np.pi)*mu_2
    n_3 = 1 - delta_3 + 1j*lam/(4*np.pi)*mu_3
    
    k_z1 = k * np.sqrt(n_1**2 - np.cos(alpha * np.pi/180)**2)
    k_z2 = k * np.sqrt(n_2**2 - np.cos(alpha * np.pi/180)**2)
    k_z3 = k * np.sqrt(n_3**2 - np.cos(alpha * np.pi/180)**2)

    r_1 = np.exp(-2 * k_z1 * k_z2 * sigma_1**2) * (k_z1 - k_z2) / (k_z1 + k_z2)
    r_2 = np.exp(-2 * k_z2 * k_z3 * sigma_2**2) * (k_z2 - k_z3) / (k_z2 + k_z3)

    X_2 = np.exp(-2j*k_z2 * d) * r_2
    # X_1 = np.exp(-2j*k_z1 * z_1) * (r_1 + X_2 * np.exp(2j*k_z2 * z_1)) / (1 + r_1 * X_2 * np.exp(2j*k_z2 * z_1))
    X_1 = (r_1 + X_2) / (1 + r_1 * X_2)

    R = np.abs(X_1)**2

    return R

alpha_c_PS = np.sqrt(2*delta_2)*180/np.pi
alpha_c_Si = np.sqrt(2*delta_3)*180/np.pi

plt.figure(figsize=(7.2,4.4))#,dpi=300)
plt.plot(alpha
            ,R
            ,linestyle='-'
            ,linewidth=1.5
            ,label = r'Messwerte $R_{\mathrm{exp,korr}}$'
        )
plt.plot(alpha[peaks_untere_grenze+indices]
            ,R[peaks_untere_grenze+indices]
            ,'rx'
            ,mew=1.5
            ,label='Benutzte Maxima')
plt.plot(alpha
            ,parratt_algorithm(alpha, *anfangswerte)
            ,linestyle='--'
            ,linewidth=1.5
            ,label = r'Berechnete Gesamtreflektivität $R_{\mathrm{parratt}}$'
        )
plt.axvline(alpha_c_PS, 0, 0.725,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label=r'$\alpha_{\mathrm{c,PS}} = 0,0810°$'
            )
plt.axvline(alpha_c_Si, 0, 0.725,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label=r'$\alpha_{\mathrm{c,Si}} = 0,2144°$'
            )
plt.xlabel(r'$\alpha \; [°]$')
plt.ylabel(r'$R$')
plt.legend(loc = 'best', markerscale=1.5)
plt.yscale('log')
plt.xlim(-0.05, 1.4)
plt.ylim(10**(-6))
plt.tight_layout()
plt.savefig('plots/Reflektivitaetskurve2.pdf')
# plt.show()
plt.close()

# params_parratt, cov_parratt = curve_fit(parratt_algorithm, alpha[1:], R[1:], p0=anfangswerte, maxfev=100000, method='lm')

for name, param in zip(('d', 'delta_2','delta_3', 'sigma_1', 'sigma_2'), anfangswerte):
    print(r'{0}:  {1:.4e}'.format(name, param))


print('Krit. Winkel PS: {:.4f}'.format(alpha_c_PS))
print('Krit. Winkel Si: {:.4f}'.format(alpha_c_Si))

######################################
### Abweichungen von den Theoriewerten berechnen
a_delta_2 = (3.5*10**(-6) - delta_2) / (3.5*10**(-6))
a_delta_3 = (7.6*10**(-6) - delta_3) / (7.6*10**(-6))
print(a_delta_2)
print(a_delta_3)

a_alpha_c_PS = (0.153 - alpha_c_PS) / (0.153)
a_alpha_c_Si = (0.223 - alpha_c_Si) / (0.223)
print(a_alpha_c_PS)
print(a_alpha_c_Si)


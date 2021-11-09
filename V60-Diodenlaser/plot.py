import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pylab import *


#IMPORT Datas
#-----------------------------------
#Stabilitätsbedingung
#konkav/konkav
Lk,Ik=np.genfromtxt("data/stabil_konkav.txt",unpack=True)
#planar/konkav
Lp,Ip=np.genfromtxt("data/stabil_plankonkav.txt",unpack=True)

#TEM-Moden
x2_m,P2_m=np.genfromtxt("data/TEM2_moden.txt",unpack=True)
x_m,P_m=np.genfromtxt("data/TEM_moden.txt",unpack=True)
#Polarisation
a_pol,P_pol=np.genfromtxt("data/polarisaton.txt",unpack=True)

#-----------------------------------

#PLOT Stabilitätsbedingung
#-----------------------------------
vlines(140, 0, 3.5, colors='r', linestyles='--',alpha=0.6)
plt.plot(Lk,Ik,"xb",label="Messwerte (konkav/konkav)")
plt.plot(Lp,Ip,"xk",label="Messwerte (plan/konkav)")
plt.xlabel(r"$L\,/\,$cm")
plt.ylabel(r"$P\,/\,$mW")
plt.legend(loc="best")
plt.grid()
plt.savefig("plots/leistung.pdf")
plt.close()



#Extrapolation
def gerade(x,m,b):
    return m*x+b
def quad(x,a,b,c):
    return a*x+b*x**2+c
params_ger,cov_ger=curve_fit(gerade,Lp/100,1-Lp/(1.4*100))
params_qu,cov_qu=curve_fit(quad,Lk/100,(1-Lk/(100*1.4))**2)
x=np.arange(0, 300,5)  

#PLOT Stabilitätprodukt
plt.axhline(y=0, color='k', linestyle='--')
plt.axhline(y=1, color='k', linestyle='--')
vlines(1.4, -1.5, 1, colors='r', linestyles='--',alpha=0.6)
vlines(2.8, -1.5, 1, colors='r', linestyles='--',alpha=0.6)
plt.plot(x/100,gerade(x/100,*params_ger),"--y",alpha=0.5)
plt.plot(x/100,quad(x/100,*params_qu),"--b",alpha=0.5)
plt.plot(Lk/100,(1-Lk/(100*1.4))**2,"b",label=r"$r_1=r_2=1,40\,$m")
plt.plot(Lp/100,1-Lp/(1.4*100),"y",label=r"$r_1=\infty{,} r_2=1,40\,$m")
plt.grid()
plt.legend()
plt.xlabel(r"$L\,/\,$m")
plt.ylabel(r"$g_1 g_2$")
plt.ylim(-1.5,1.5)
plt.savefig("plots/stabilitätsbedingung.pdf")
plt.close()
#-----------------------------------

#TEM-Moden
#-----------------------------------
#Plot

def tem00(x,I0,x0,w):
    return I0*exp(-2*(x-x0)**2/(w**2))
params_tem00,cov_tem00=curve_fit(tem00,x_m,P_m)
def tem10(x,I0,x0,w):
    return I0*4*((x-x0)**2/(w**2))*exp(-2*(x-x0)**2/(w**2))+0.9
params_tem10,cov_tem10=curve_fit(tem10,x2_m,P2_m)

plt.grid()
plt.plot(x_m,P_m,"x",label="Messwerte")#TEM00
plt.plot(x_m,tem00(x_m,*params_tem00),label="Interpolation")
plt.xlabel(r"$x\,/\,$mm")
plt.ylabel(r"$P\,/\,$mW")
plt.savefig("plots/TEM00.pdf")
plt.close()

plt.grid()
plt.plot(x2_m,P2_m,"x",label="Messwerte")#TEM00
plt.plot(x2_m,tem10(x2_m,*params_tem10),label="Interpolation")
plt.xlabel(r"$x\,/\,$mm")
plt.ylabel(r"$P\,/\,$mW")
plt.savefig("plots/TEM10.pdf")
plt.close()
print("Params TEM-Mode")
print(f"""TEM00 Params {params_tem00}(I0,x0,w)
TEM10 Params {params_tem10}\t(I0,x0,w)""")


#-----------------------------------

#Polarisation
#-----------------------------------

def pol(x,I0,a0):
    return I0*cos(x+a0)**2
params_pol,cov_pol=curve_fit(pol,a_pol*np.pi/180,P_pol)

plt.plot(a_pol,pol(a_pol*np.pi/180,*params_pol),label="Interpolation")
plt.plot(a_pol,P_pol,"x",label="Messwerte")
plt.xlabel(r"Winkel $\theta\,/\,$°")
plt.ylabel(r"$P\,/\,$mW")
plt.legend(loc="best")
plt.savefig("plots/pol.pdf")

print(f"""Polarisation
Params: {params_pol}(I0,a0) a0 in °: {params_pol[1]*180/np.pi}°""")
#-----------------------------------



#Wellenlänge
#-----------------------------------
g1=1/1200
d1=742/2
L1=312.5

g2=1/100
d2=20
L2=312.5

g3=1/600
d3=261/2
d3_2=758/2
L3=312.5


g4=1/80
d4=35/2
d4_2=100/2
d4_3=203/2
L4=312.5

listlam=[637.35,642.25,642.95,638.69,698.90,658.29,643.57]
def lam(g,d,L,n):
    return d*sin(arctan(g/L))/n
print(f"""1200: {lam(d1,g1,L1,1)*10**6}
100: {lam(d2,g2,L2,1)*10**6}
600: {lam(d3,g3,L3,1)*10**6}
600: {lam(d3_2,g3,L3,2)*10**6}
80: {lam(d4,g4,L4,1)*10**6}
80: {lam(d4_2,g4,L4,3)*10**6}
80: {lam(d4_3,g4,L4,6)*10**6}
Mittelwert: {np.mean(listlam)}""")

#-----------------------------------

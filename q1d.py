from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

# constants in cgs
G = 6.67e-8
eV = 1.602e-12 #erg
amu = 1.66e-24 #g
sigma_SB = 5.673e-5 #erg cm-2 K^4 s-1
M_sun = 1.99e33 #g
R_sun = 6.96e10 #cm
L_sun = 3.83e33 #erg s-1
year = 365.25 * 24. * 3600. #s

# constant parameters
n = 3.0     #changed from 1.5 to 3.0 for massive star  
f_acc = 0.75
temp_H = 3500.0 #K

psi_I = 13.6 * eV / amu
psi_M =  2.2 * eV / amu
psi_D =  100 * eV / amu

M_acc = 1e-4 * M_sun / year

# takes the log of R and log of M and returns the slope of plot at that point
# the dependent parameter must come first
def dlnRdlnM(lnR, lnM, n=n, f_acc=f_acc, M_acc=M_acc):
  R = np.exp(lnR)
  M = np.exp(lnM)
  L = max(4.*np.pi*R**2 * sigma_SB * temp_H**4, L_sun * (M/M_sun)**3)
  return (2. - 2.*(5.-n)/3. * 
        (f_acc + (R/(G*M)) *
       (psi_I + psi_M - psi_D + L/M_acc)))

# initial conditions
R0 = 2.5 * R_sun

# our values for lnM
# 10^1.7 = 50.12
lnMs = np.log(np.logspace(-2, 1.698, 500)*M_sun)

# solve for lnR
lnRs = odeint(dlnRdlnM, np.log(2.5*R_sun), lnMs, args=(n, f_acc, M_acc))

# convert back to M and R
Ms = np.exp(lnMs)
Rs = np.exp(lnRs[:,0])

# plot ratios to solar standard
plt.plot(Ms/M_sun, Rs/R_sun, lw=2)
#plt.xscale('log')
plt.xlabel(r'$M/M_{\odot}$')
plt.ylabel(r'$R/R_{\odot}$')
plt.savefig("radiustomass2.eps")
plt.show()

Ls = np.zeros(np.size(Ms))

for i in range(np.size(Ls)):
  Ls[i] = f_acc*G*Ms[i]*M_acc/Rs[i] + \
            max(4*np.pi*Rs[i]**2*sigma_SB*temp_H**4, L_sun * (Ms[i]/M_sun)**3)

plt.clf()
plt.plot(Ms/M_sun, Ls/L_sun)
plt.yscale('log')
plt.xlabel(r'$M/M_{\odot}$')
plt.ylabel(r'$L/L_{\odot}$')
plt.savefig("lumtomass2.eps")
plt.show()

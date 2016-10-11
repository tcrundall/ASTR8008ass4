import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# constants in cgs
G = 6.67e-8
eV = 1.6e-12 #erg
amu = 1.66e-24 #g
sigma_SB = 5.67e-5 #erg cm-2 K^4 s-1
M_sun = 1.99e33 #g
R_sun = 6.96e10 #cm
L_sun = 3.83e33 #erg s-1
#L_sun = 3.9e33 #erg s-1
year = 365.25 * 24. * 3600. #s

# constant parameters
n = 1.5
f_acc = 0.75
temp_H = 3500.0 #K

psi_I = 13.6 * eV / amu
psi_M =  2.2 * eV / amu
psi_D =  100 * eV / amu

M_acc = 1e-5 * M_sun / year

# takes the log of R and log of M and returns the slope of plot at that point
# the dependent parameter must come first
def dlnRdlnM(lnR, lnM, n=n, f_acc=f_acc, M_acc=M_acc):
  R = np.exp(lnR)
  M = np.exp(lnM)
  return (2. - 2.*(5.-n)/3. * 
        (f_acc + (R/(G*M)) *
       (psi_I + psi_M - psi_D + 4.*np.pi*R**2 * sigma_SB * temp_H**4/M_acc)))

# initial conditions
R0 = 2.5 * R_sun

# our values for lnM
lnMs = np.log(np.logspace(-2, 0, 500)*M_sun)
print(lnMs)

lnRs = odeint(dlnRdlnM, np.log(2.5*R_sun), lnMs, args=(n, f_acc, M_acc))
print(np.shape(lnRs))

Ms = np.exp(lnMs)
Rs = np.exp(lnRs[:,0])

plt.plot(Ms/M_sun, Rs/R_sun, lw=2)
plt.xscale('log')
plt.show()

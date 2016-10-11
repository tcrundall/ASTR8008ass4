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

# Massive *** off list of the coefficients' coefficients
eq4coco = [ [1.71, 0.622, -0.926, -1.17, -0.30],
            [6.60, -0.425, -12.1, -10.7, -2.51],
            [10.1, -7.11, -31.7, -24.2, -5.34],
            [1.01, 0.327, -0.00923, -0.0388, -0.00413],
            [0.0749, 0.0241, 0.0723, 0.0304, 0.00198],
            [0.0108, 0, 0, 0, 0],
            [3.08, 0.944, -2.15, -2.49, -0.638],
            [17.8, -7.45, -49.0, -40.1, -9.09],
            [0.000226, -0.00187, 0.00389, 0.00142, -0.0000767] ]

Z = 0.02
Z_sol = 0.0134
Z_log = np.log10(Z/Z_sol)

# list of coefficients solved for the metallicity of 0.02
co = [row[0] + row[1]*Z_log + row[2]*Z_log**2 
             + row[3]*Z_log**3 + row[4]*Z_log**4 
            for row in eq4coco]

# Analytic function for Radius using all these coefficients
def ZAMSR(M):
  R = ( (co[0]*M**2.5 + co[1]*M**6.5 + co[2]*M**11 + co[3]*M**19 + co[4]*M**19.5) /
        (co[5] + co[6]*M**2 + co[7]*M**8.5 + M**18.5 + co[8]*M**19.5) )
  return R

def main():
  Rs2 = ZAMSR(Ms/M_sun)

  # Find the mass where the calculated radius equals the main sequence path
  # as given by Tout et al. (1996)
  diff = abs(Rs/R_sun - Rs2)
  print(Ms[np.argmin(diff)]/M_sun)
  
  # plot ratios to solar standard
  plt.plot(Ms/M_sun, Rs/R_sun, lw=2)
  plt.plot(Ms/M_sun, ZAMSR(Ms/M_sun))
  plt.xlabel(r'$M/M_{\odot}$')
  plt.ylabel(r'$R/R_{\odot}$')
  plt.legend(["Projected path", "Main sequence"])
  plt.savefig("radiustomass3.eps")
  plt.show()
  
main()

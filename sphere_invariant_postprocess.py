import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

np.random.seed(42)

df = pd.read_csv(r'data/sphere_data/Sphere_20_Runs.csv')
df = pd.DataFrame(df, columns= ['Re','Exp C_d'])

data = df.to_numpy()
Re = data[:,0]
Cd = data[:,1]

def sphere_analytic(x):
    return 24/x

def sphere_learned_dimless(x):
    return 23.99/x

def sphere_learned(eta, rho, a, U):
    learned_reln = 0.001/(rho*a*U) + 0.003*eta/(rho**2 * U) + 0.033*eta*a**2/rho**2 + 0.001*eta/(rho * a**2 * U) \
                    + 11.986*eta/(rho * a * U) + 0.002*eta/(a * U) + 0.268*eta * rho**2 * a**2 \
                    + 0.88* eta**2 * a**2/rho + 0.006 * rho**2/(a * U)
    return learned_reln

def sphere_learned2(eta, rho, a, U):
    learned_reln = 0.033*eta*a**2/rho**2 +  + 11.986*eta/(rho * a * U) + 0.268*eta * rho**2 * a**2 \
                    + 0.88* eta**2 * a**2/rho  
    return learned_reln

width = 30

eta1 = random.uniform(1, width)*np.random.rand(20)
rho1 = random.uniform(1, width)*np.random.rand(20)
len_a1 = random.uniform(1, width)*np.random.rand(20)
U1 = Re*eta1/(2*rho1*len_a1)
print(U1.shape)

eta2 = random.uniform(1, width)*np.random.rand(20)
rho2 = random.uniform(1, width)*np.random.rand(20)
len_a2 = random.uniform(1, width)*np.random.rand(20)
U2 = Re*eta2/(2*rho2*len_a2)

eta3 = random.uniform(1, width)*np.random.rand(20)
rho3 = random.uniform(1, width)*np.random.rand(20)
len_a3 = random.uniform(1, width)*np.random.rand(20)
U3 = Re*eta3/(2*rho3*len_a3)

eta4 = random.uniform(1, width)*np.random.rand(20)
rho4 = random.uniform(1, width)*np.random.rand(20)
len_a4 = random.uniform(1, width)*np.random.rand(20)
U4 = Re*eta4/(2*rho4*len_a4)

eta5 = random.uniform(1, width)*np.random.rand(20)
rho5 = random.uniform(1, width)*np.random.rand(20)
len_a5 = random.uniform(1, width)*np.random.rand(20)
U5 = Re*eta5/(2*rho5*len_a5)

Cd_pi = sphere_analytic(Re)
Cd_pred_dimless = sphere_learned_dimless(Re)
Cd_pred1 = sphere_learned2(eta1, rho1, len_a1, U1)
Cd_pred2 = sphere_learned2(eta2, rho2, len_a2, U2)
Cd_pred3 = sphere_learned2(eta3, rho3, len_a3, U3)
Cd_pred4 = sphere_learned2(eta4, rho4, len_a4, U4)
Cd_pred5 = sphere_learned2(eta5, rho5, len_a5, U5)

# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(6, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

plt.loglog(Re, Cd_pi, linewidth=3, linestyle="solid", label=" ")
plt.loglog(Re, Cd_pred_dimless, linewidth=3, linestyle="dashed", label=' ')
plt.loglog(Re, Cd_pred1, '^-.', ms=9, lw=3, label=' ')
plt.loglog(Re, Cd_pred2, 's:', ms=9, lw=3,label=' ')
plt.loglog(Re, Cd_pred3, '.--', ms=15,lw=3, label=' ')

# plt.loglog(Re, Cd_pred3, '.-', label='W/o proposed learning')
# plt.loglog(Re, Cd_pred4, '.-', label='W/o proposed learning')
# plt.loglog(Re, Cd_pred5, '.-', label='W/o proposed learning')
plt.xlabel('$Re$', fontsize = 15)
plt.ylabel('$C_d$', fontsize = 15)
plt.legend(fontsize=12, frameon=False, loc='lower left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(rotation=0)
plt.savefig('Figures/invariant.jpeg', dpi=500, bbox_inches="tight")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from math import pi

def thetafun(theta, phi, re):
    Cd_learned = 12.885 / (re * phi) + 12.388 / re + 0.519 * (phi - 1) * theta**2 / re 
    return Cd_learned

def thetafun2(theta, phi, re):
    Cd_learned = 1/re * (11.15/phi + 12.3 + 0.24 * theta + .4 * theta**2 + 0.46 * phi * theta)
    return Cd_learned


# def thetafun(theta):
#     return 145.355 + 25.95*theta**2

theta = [0, pi/4, pi/2]
theta5 = [0, pi/8, pi/4, 3*pi/8, pi/2] 

# learned = [thetafun(theta5[0]), thetafun(theta5[1]), thetafun(theta5[2]), thetafun(theta5[3]), thetafun(theta5[4])]

empirical = [142.8, 168.6, 194.4]
Anderson = [138.39, 174.22, 203.44]
HnS = [139.58, 166.4, 175.9]
Ouchene = [151.91, 178.79, 205.68] # looks s-shaped in reference. Data inconsistent.
training_pts = [138.77, 151.53, 171.02, 187.96, 195.92]

x = np.linspace(0, 1.57, 1000)
y = 145.355 + 25.95*x**2

plt.plot(x, thetafun(x, 6, 0.1), linewidth=3)
plt.plot(x, thetafun2(x, 6, 0.1), linewidth=3)
plt.plot(theta, empirical, linewidth=3, linestyle="dotted")
plt.plot(theta, Anderson, linewidth=3, linestyle="dashed")
plt.plot(theta, HnS, linewidth=3, linestyle="dashdot")
plt.plot(theta, Ouchene, linewidth=3, linestyle="dotted")
plt.scatter(theta5, training_pts)

plt.xlabel('$θ$', fontsize = 12)
plt.ylabel('$C_d$', fontsize = 12)
plt.legend(['Learned' , 'Learned new', 'Empirical', 'Anderson et al.', 'H and S et al.', 'Ouchene et al.', 'Training pts'], prop={'size': 12}, frameon=False, loc='upper left')
x_ticks_labels = ['0', 'π/4', 'π/2']
plt.xticks(theta, x_ticks_labels, fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(rotation=0)
plt.savefig("comparison.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

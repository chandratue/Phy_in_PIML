import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

print("---------------------------------------")

n005_const_array = np.array([[0, 0, 13.153, 0, 0]])
n005_re_array = np.array([[23.963, 24.015, 23.798, 23.967, 24.041]])
n005_re2_array = np.array([[0, 0, 0, 0, 0]])

n005_const = np.mean(n005_const_array)
n005_re = np.mean(n005_re_array)
n005_re2 = np.mean(n005_re2_array)

print("Mean of n005 constant terms: ", n005_const)
print("Mean of n005 re terms: ", n005_re)
print("Mean of n005 re2 terms: ", n005_re2)

print("---------------------------------------")
print("---------------------------------------")

n01_const_array = np.array([[70.539, 102.861, 0, 111.252, 0]])
n01_re_array = np.array([[23.38, 23.188, 23.62, 22.947, 23.851]])
n01_re2_array = np.array([[0, 0, 0.001, 0, 0.001]])

n01_const = np.mean(n01_const_array)
n01_re = np.mean(n01_re_array)
n01_re2 = np.mean(n01_re2_array)

print("Mean of n01 constant terms: ", n01_const)
print("Mean of n01 re terms: ", n01_re)
print("Mean of n01 re2 terms: ", n01_re2)

print("---------------------------------------")
print("---------------------------------------")

n02_const_array = np.array([[0, 0, 29.494, 0, 40.194]])
n02_re_array = np.array([[24.122, 23.933, 23.626, 23.587, 23.673]])
n02_re2_array = np.array([[0, 0, 0, 0.001, 0]])

n02_const = np.mean(n02_const_array)
n02_re = np.mean(n02_re_array)
n02_re2 = np.mean(n02_re2_array)

print("Mean of n02 constant terms: ", n02_const)
print("Mean of n02 re terms: ", n02_re)
print("Mean of n02 re2 terms: ", n02_re2)

print("---------------------------------------")
print("---------------------------------------")

n05_const_array = np.array([[0, 0, 0, 28.346, 29.177]])
n05_re_array = np.array([[23.948, 24.012, 23.81, 23.779, 23.843]])
n05_re2_array = np.array([[0, 0, 0.001, 0, 0]])

n05_const = np.mean(n05_const_array)
n05_re = np.mean(n05_re_array)
n05_re2 = np.mean(n05_re2_array)

print("Mean of n05 constant terms: ", n05_const)
print("Mean of n05 re terms: ", n05_re)
print("Mean of n05 re2 terms: ", n05_re2)

print("---------------------------------------")
print("---------------------------------------")

n1_const_array = np.array([[168.352, 0, 0, 122.258, 0]])
n1_re_array = np.array([[23.422, 24.358, 24.188, 23.026, 20.61]])
n1_re2_array = np.array([[0, 0, 0, 0, 0.007]])

n1_const = np.mean(n1_const_array)
n1_re = np.mean(n1_re_array)
n1_re2 = np.mean(n1_re2_array)

print("Mean of n1 constant terms: ", n1_const)
print("Mean of n1 re terms: ", n1_re)
print("Mean of n1 re2 terms: ", n1_re2)

print("---------------------------------------")
print("---------------------------------------")

def sphere_analytic(x):
    return 24/x

def sphere_learned_n005(x):
    return n005_const + n005_re/x + n005_re2

def sphere_learned_n01(x):
    return n01_const + n01_re/x + n01_re2

def sphere_learned_n02(x):
    return n02_const + n02_re/x + n02_re2

def sphere_learned_n05(x):
    return n05_const + n05_re/x + n05_re2

def sphere_learned_n1(x):
    return n1_const + n1_re/x + n1_re2

df = pd.read_csv(r'data/sphere_data/Sphere_20_Runs.csv')
df = pd.DataFrame(df, columns= ['Re','Exp C_d'])

data = df.to_numpy()
Re = data[:,0]

Cd_simulation = data[:,1].reshape(-1,1)
Cd_analytic = sphere_analytic(Re).reshape(-1,1)

Cd_learned_n005 = sphere_learned_n005(Re).reshape(-1,1)
Cd_learned_n01 = sphere_learned_n01(Re).reshape(-1,1)
Cd_learned_n02 = sphere_learned_n02(Re).reshape(-1,1)
Cd_learned_n05 = sphere_learned_n05(Re).reshape(-1,1)
Cd_learned_n1 = sphere_learned_n1(Re).reshape(-1,1)

relative_error_analytic_n005 = np.mean((Cd_learned_n005 - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic_n005*100)

relative_error_analytic_n01 = np.mean((Cd_learned_n01 - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic_n01*100)

relative_error_analytic_n02 = np.mean((Cd_learned_n02 - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic_n02*100)

relative_error_analytic_n05 = np.mean((Cd_learned_n05 - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic_n05*100)

relative_error_analytic_n1 = np.mean((Cd_learned_n1 - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic_n1*100)

##################################################
# Given data
x = [0.005, 0.01, 0.02, 0.05, 0.1]
mean_curve = [relative_error_analytic_n005, relative_error_analytic_n01, relative_error_analytic_n02, relative_error_analytic_n05, relative_error_analytic_n1]

# Plot curves
plt.plot(x, mean_curve, label='Analytic error', linestyle='--', linewidth=5)

# Add labels and legend with font size 24
plt.xlabel('Gaussian noise (standard deviation)', fontsize=16)
plt.ylabel('Relative percent error', fontsize=16)
#plt.title('Plot of Curves with Mean and Variance', fontsize=24)
#plt.legend(fontsize=16)

# Remove grid lines
plt.grid(False)

# Set font size for ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Set y-axis range
#plt.ylim(-0.025, 0.15)

# Set tight layout
plt.tight_layout()

# Save the figure as JPEG with DPI 300
plt.savefig('noise_error.jpeg', dpi=300)  # quality parameter is only applicable to JPEG format

# Show plot
plt.show()

# Define colors and linestyles
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
linestyles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 5, 1, 5))]

# Plot lines
plt.loglog(Re, Cd_analytic, label='Analytic', linestyle=linestyles[0], linewidth=5, color=colors[0])
plt.loglog(Re, Cd_learned_n005, label='Noise 0.005', linestyle=linestyles[1], linewidth=5, color=colors[1])
plt.loglog(Re, Cd_learned_n01, label='Noise 0.01', linestyle=linestyles[2], linewidth=5, color=colors[2])
plt.loglog(Re, Cd_learned_n02, label='Noise 0.02', linestyle=linestyles[3], linewidth=5, color=colors[3])
plt.loglog(Re, Cd_learned_n05, label='Noise 0.05', linestyle=linestyles[4], linewidth=5, color=colors[4])
plt.loglog(Re, Cd_learned_n1, label='Noise 0.1', linestyle=linestyles[5], linewidth=5, color=colors[5])

# Add labels and legend with font size 24
plt.xlabel('$Re$', fontsize=16)
plt.ylabel('$C_d$', fontsize=16)
#plt.title('Plot of Curves with Mean and Variance', fontsize=24)
plt.legend(fontsize=14, frameon = False, loc ='lower left')

# Remove grid lines
plt.grid(False)

# Set font size for ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Create inset plot
ax_inset = inset_axes(plt.gca(), width="40%", height="40%", loc='upper right')
ax_inset.loglog(Re, Cd_analytic, linestyle=linestyles[0], linewidth=3, color=colors[0])
ax_inset.loglog(Re, Cd_learned_n005, linestyle=linestyles[1], linewidth=3, color=colors[1])
ax_inset.loglog(Re, Cd_learned_n01, linestyle=linestyles[2], linewidth=3, color=colors[2])
ax_inset.loglog(Re, Cd_learned_n02, linestyle=linestyles[3], linewidth=3, color=colors[3])
ax_inset.loglog(Re, Cd_learned_n05, linestyle=linestyles[4], linewidth=3, color=colors[4])
ax_inset.loglog(Re, Cd_learned_n1, linestyle=linestyles[5], linewidth=3, color=colors[5])

# Set limits for inset plot
ax_inset.set_xlim(0.06, 0.1)
ax_inset.set_ylim(300, 450)

# Reduce the size of ticks in the inset plot
ax_inset.tick_params(axis='both', which='major', labelsize=7)
ax_inset.tick_params(axis='both', which='minor', labelsize=7)


# Save the figure as JPEG with DPI 300
plt.savefig('noise_learned.jpeg', dpi=300)  # quality parameter is only applicable to JPEG format
plt.show()
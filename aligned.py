import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
from sklearn.metrics import mean_absolute_percentage_error

np.random.seed(142)

path = r'data/ellipsoid0_data/' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    frame = pd.read_csv(filename, index_col=None, header=0)
    li.append(frame)

df = pd.concat(li, axis=0, ignore_index=True)

df = pd.DataFrame(df, columns= ['Re', 'phi', 'Exp C_d'])

pd.set_option("display.max_rows", None, "display.max_columns", None)

frac = 1
shuffled_df = df.sample(frac=frac)

data = shuffled_df.to_numpy()
Re = data[:,0]
phi = data[:,1]
Cd = data[:,2]
inp = data[:,0:2]

deg = 2
counter = 0
dict = [None]*((2*deg+1))**2
for i in range(-deg, deg+1):
    for j in range(-deg, deg+1):
        dict[counter] = 'Re^'+str(i) +'phi^'+str(j)
        counter = counter +1

def library(Re, phi, deg):
    counter = 0
    lib = np.zeros((len(Re), (2*deg+1)**2))
    for i in range(-deg, deg+1):
        for j in range(-deg, deg+1):
            lib[:,counter] = Re**i * phi**j
            counter = counter +1
    return lib

library_cv = library(Re, phi, deg)
alphas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

regcv = linear_model.LassoCV(fit_intercept=False, alphas=alphas, positive = True, tol = 1e-4, max_iter=100000)
regcv.fit(library_cv, Cd)

alpha = regcv.alpha_
R2_cv = regcv.score(library_cv, Cd)
rmse_cv = np.sqrt(mean_squared_error(Cd, regcv.predict(library_cv)))

inp_train, inp_test, Cd_train, Cd_test = train_test_split(inp, Cd, test_size=0.2, shuffle=True)
Re_train = inp_train[:,0]
phi_train = inp_train[:,1]
Re_test = inp_test[:,0]
phi_test = inp_test[:,1]

library_train = library(Re_train, phi_train, deg)
library_test = library(Re_test, phi_test, deg)
reg = linear_model.Lasso(fit_intercept=False, alpha=alpha, positive = True, tol = 1e-4, max_iter=1000000)
reg.fit(library_train, Cd_train)
Cd_pred = reg.predict(library_test)
train_score = reg.score(library_train, Cd_train)
test_score = reg.score(library_test, Cd_test)
print("train score: ",train_score)
print("test score: ",test_score)
print("----")
rmse_train = np.sqrt(mean_squared_error(Cd_train, reg.predict(library_train)))
rmse_test = np.sqrt(mean_squared_error(Cd_test, reg.predict(library_test)))
percent_error = mean_absolute_percentage_error(Cd_test, reg.predict(library_test))
print("RMSE train: ",rmse_train)
print("RMSE test: ",rmse_test)
#print("Percentage error:", percent_error*100)
print("----")
print("----")

learned_dict = list(zip(dict, reg.coef_))
np.savetxt('Library/Aligned_Library/aligned_learned_dictionary.csv', [p for p in learned_dict], delimiter=',', fmt='%s')

#############################
#### Truncated Model ########
#############################

df = pd.read_csv(r'Library/Aligned_Library/aligned_learned_dictionary.csv', usecols=[1], header=None)
data = df.to_numpy()

data = np.round(data, 3)

print(data)

learned_dict = list(zip(dict, data))
# np.savetxt('Library/Aligned_Library/Aligned_obtained_formula.csv', [p for p in learned_dict], delimiter=',', fmt='%s')

###########################
#### Error ###############
###########################
path = r'data/ellipsoid0_data/' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    frame = pd.read_csv(filename, index_col=None, header=0)
    li.append(frame)

df = pd.concat(li, axis=0, ignore_index=True)

df = pd.DataFrame(df, columns= ['Re', 'phi', 'Exp C_d'])

pd.set_option("display.max_rows", None, "display.max_columns", None)

data = df.to_numpy()
Re = data[:,0]
phi = data[:,1]

def ellipsoid_aligned_analytic(re, phi):
    X, Y = np.meshgrid(re, phi)
    num = 24 * 4 * (Y ** 2 - 1) * np.sqrt(Y ** 2 - 1)
    denom = X * Y ** (2 / 3) * 3 * ((2 * Y ** 2 - 1) * np.log(Y + np.sqrt(Y ** 2 - 1)) - Y * np.sqrt(Y ** 2 - 1))
    Cd_exact = num / denom
    return Cd_exact

def ellipsoid_aligned_learned(re, phi):
    X, Y = np.meshgrid(re, phi)
    return 12.279/(X*Y) + 12.076/X

def ellipsoid_aligned_learned_vec(re, phi):
    return 12.279 / (re * phi) + 12.076 / re

Cd_simulation = data[:,2].reshape(-1,1)
Cd_analytic = ellipsoid_aligned_analytic(Re, phi).reshape(-1,1)
Cd_learned = ellipsoid_aligned_learned(Re, phi).reshape(-1,1)
Cd_learned_vec = ellipsoid_aligned_learned_vec(Re, phi).reshape(-1,1)

# Compute the relative L2 error norm
relative_error_simulation = np.mean((Cd_learned_vec - Cd_simulation)**2) / np.mean(Cd_simulation**2)
print("Relative Error Simulation : ", relative_error_simulation*100)

relative_error_analytic = np.mean((Cd_learned - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic*100)

###########################
#### Plotting #############
###########################
# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(15, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

Cd_learned = ellipsoid_aligned_learned(Re, phi)
Cd_analytic = ellipsoid_aligned_analytic(Re, phi)
rel_err_analytic = (Cd_learned - Cd_analytic)**2 / Cd_analytic**2
per_err_analytic = rel_err_analytic*100
CS = plt.contourf(Re, phi, per_err_analytic, levels= 2000,cmap='Spectral')
cbar=plt.colorbar(CS,orientation='vertical')
ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=12)
plt.xlabel('Re', fontsize = 12)
plt.ylabel('$\phi$', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

scatter_meshgrid = np.meshgrid(Re_train, phi_train)
x_scat, y_scat = scatter_meshgrid
plt.scatter(x_scat, y_scat, s=20, c='black', marker='*')

plt.savefig('Figures/aligned_analytic_error.jpeg', dpi = 500, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(15, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

rel_err_sim = (Cd_learned_vec - Cd_simulation)**2 / Cd_simulation**2
per_err_sim = rel_err_sim*100
Re = Re.reshape(-1, )
phi = phi.reshape(-1, )
per_err_sim = per_err_sim.reshape(-1, )
CS = plt.tricontourf(Re, phi, per_err_sim, 2000, cmap='Spectral')
cbar = plt.colorbar(CS)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(12)
plt.xlabel('Re', fontsize = 12)
plt.ylabel('$\phi$', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
scatter_meshgrid = np.meshgrid(Re_train, phi_train)
x_scat, y_scat = scatter_meshgrid
plt.scatter(x_scat, y_scat, s=20, c='black', marker='*')

plt.savefig('Figures/aligned_simulation_error.jpeg', dpi = 500, bbox_inches='tight')
plt.show()
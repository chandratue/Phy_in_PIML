import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(142)

df = pd.read_csv(r'data/sphere_data/Sphere_20_Runs.csv')
df = pd.DataFrame(df, columns= ['Re','Exp C_d'])

frac = 1
shuffled_df = df.sample(frac=frac)

data = shuffled_df.to_numpy()
Re = data[:,0]
Cd = data[:,1]

deg = 2 # for library
def library(Re, deg):
    lib = np.zeros((len(Re), 2*deg+1))
    for i in range(-deg, deg+1):
        lib[:,i+deg] = Re**i
    return lib

dict = [None]*(2*deg+1)
for i in range(-deg, deg+1):
    dict[i+deg] = 'Re^'+ str(i)

library_cv = library(Re, deg)
alphas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

regcv = linear_model.LassoCV(fit_intercept=False, alphas=alphas, positive = True, tol = 1e-5)
regcv.fit(library_cv, Cd)

alpha = regcv.alpha_
print("alpha:", alpha)
R2_cv = regcv.score(library_cv, Cd)
rmse_cv = np.sqrt(mean_squared_error(Cd, regcv.predict(library_cv)))

Re_train, Re_test, Cd_train, Cd_test = train_test_split(Re, Cd, test_size=0.2, shuffle=True)

library_train = library(Re_train, deg)
library_test = library(Re_test, deg)
reg = linear_model.Lasso(fit_intercept=False, alpha=alpha, positive = True, tol = 1e-5, max_iter=100000)
reg.fit(library_train, Cd_train)
Cd_pred = reg.predict(library_test)
train_score = reg.score(library_train, Cd_train)
test_score = reg.score(library_test, Cd_test)
print("train score: ",train_score)
print("test score: ",test_score)
print("----")
rmse_train = np.sqrt(mean_squared_error(Cd_train, reg.predict(library_train)))
rmse_test = np.sqrt(mean_squared_error(Cd_test, reg.predict(library_test)))
print("RMSE train: ",rmse_train)
print("RMSE test: ",rmse_test)
print("----")
print("----")

learned_dict = list(zip(dict, reg.coef_))
np.savetxt('Library/Sphere_Library/sphere_learned_dictionary.csv', [p for p in learned_dict], delimiter=',', fmt='%s')

#############################
#### Truncated Model ########
#############################

df = pd.read_csv(r'Library/Sphere_Library/sphere_learned_dictionary.csv', usecols=[1], header=None)
data = df.to_numpy()
data = np.round(data, 3)

print(data)

learned_dict = list(zip(dict, data))
# np.savetxt('Library/Sphere_Library/Sphere_obtained_formula.csv', [p for p in learned_dict], delimiter=',', fmt='%s')

###########################
#### Error #############
###########################

def sphere_analytic(x):
    return 24/x

def sphere_learned(x):
    return 23.999/x

df = pd.read_csv(r'data/sphere_data/Sphere_20_Runs.csv')
df = pd.DataFrame(df, columns= ['Re','Exp C_d'])

data = df.to_numpy()
Re = data[:,0]

Cd_simulation = data[:,1].reshape(-1,1)
Cd_analytic = sphere_analytic(Re).reshape(-1,1)
Cd_learned = sphere_learned(Re).reshape(-1,1)

# Compute the relative L2 error norm
relative_error_simulation = np.mean((Cd_learned - Cd_simulation)**2) / np.mean(Cd_simulation**2)
print("Relative Error Simulation : ", relative_error_simulation*100)

relative_error_analytic = np.mean((Cd_learned - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic*100)

###########################
#### Plotting #############
###########################

abs_err_simulated = np.abs(Cd_simulation - Cd_learned)
abs_err_analytic = np.abs(Cd_analytic - Cd_learned)

rel_err_sim = (Cd_learned - Cd_simulation)**2 / Cd_simulation**2
per_err_sim = rel_err_sim*100

rel_err_analytic = (Cd_learned - Cd_analytic)**2 / Cd_analytic**2
per_err_analytic = rel_err_analytic*100

# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(6, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

# Plot the data with red and blue lines, one with dotted and one with solid style
# ax.plot(Re, abs_err_analytic, color='blue', linestyle='dotted', linewidth=3, label='Analytic')
# ax.plot(Re, abs_err_simulated, color='red', linestyle='dashdot', linewidth=3, label='Simulated')

ax.semilogy(Re, per_err_analytic, 'o', ms=9, color='blue', linewidth=3, label=' ') # analytic
ax.semilogy(Re, per_err_sim, '^', ms=9, color='red', linewidth=3, label=' ') # simulated train
ax.semilogy(list([Re[0]])+list(Re[3:5]), list([per_err_sim[0]])+list(per_err_sim[3:5]), '*', ms=9, color='green', label=' ') # simulated test

# Set the axis labels with bold font weight
# ax.set_xlabel(r"$Re$", fontsize=12, color='black')
ax.set_ylabel(r"$\mathcal{E}(Re)$", fontsize=12, color='black')

# Set tick labels fontweight to bold and increase font size
# ax.tick_params(axis='both', which='major', labelsize=12, width=1, length=10)

# Set the spines linewidth to bold
# ax.spines['top'].set_linewidth(2)
# ax.spines['right'].set_linewidth(2)
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)

# for tx in Re_train:
#     plt.axvline(x=tx, color='cyan', linestyle='dashed', alpha=0.5)

ax.legend(loc='best', fontsize=13, frameon=False)

plt.savefig('Figures/sphere_abs_error.jpeg', dpi=500, bbox_inches="tight")
plt.show()

# ms=3.0
# plt.plot(Re, Cd_learned, 'o', markersize=ms, label='learned')
# plt.plot(Re, Cd_simulation, 'o', markersize=ms, label='data')
# plt.plot(Re, Cd_analytic, 'o', markersize=ms, label='analytical')
# plt.legend()
# plt.savefig('Figures/sphere_learned_scatter.jpeg', dpi=500, bbox_inches="tight")
# plt.show()
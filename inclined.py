import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
from matplotlib.ticker import FixedLocator, FixedFormatter

np.random.seed(142)

path = r'data/ellipsoid_inclined_data' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    frame = pd.read_csv(filename, index_col=None, header=0)
    li.append(frame)

df = pd.concat(li, axis=0, ignore_index=True)

df = pd.DataFrame(df, columns= ['Re', 'phi', 'theta', 'Exp C_d'])

frac = 1 # what fraction of data to work at. 1 for 100 data, .8 for 80, .6 for 60, .4 for 40, .2 for 20, .1 for 10 data
shuffled_df = df.sample(frac=frac)

data = shuffled_df.to_numpy()
Re = data[:,0]
phi = data[:,1]
theta = data[:,2]
Cd = data[:,3]
inp = data[:,0:3]

deg = 2
deg_trigo = 2
terms1 = (2*deg+1)*(2*deg+1)
terms2 = (deg)*(deg+1)*(2*deg+1)
total_terms = terms1 + terms2
dict1 = [None]*terms1
dict2 = [None]*terms2
dict = [None]*total_terms

counter = 0
for i in range(-deg, deg+1):
    for j in range(-deg, deg+1):
        dict1[counter] = 'Re^'+str(i) +'phi^'+str(j)
        counter = counter +1

counter = 0
for i in range(1, deg+1):
    for j in range(0, deg+1):
        for k in range(-deg, deg+1):
            dict2[counter] = 'Re^'+str(k) + '(phi-1)^'+str(i) +'theta^'+str(2*j)
            counter = counter +1

dict = dict1 + dict2

def library(Re, phi, theta, deg, deg_trigo):
    lib1 = np.zeros((len(Re), terms1))
    lib2 = np.zeros((len(Re), terms2))
    lib = np.zeros((len(Re), total_terms))

    counter = 0
    for i in range(-deg, deg+1):
        for j in range(-deg, deg+1):
            lib1[:,counter] = Re**i * phi**j
            counter = counter +1

    counter = 0
    for i in range(1, deg+1):
        for j in range(0, deg+1):
            for k in range(-deg, deg+1):
                lib2[:,counter] = Re**k * (phi-1)**i * theta**(2*j)
                counter = counter +1

    lib = np.concatenate((lib1, lib2), 1)

    return lib

library_cv = library(Re, phi, theta, deg, deg_trigo)
alphas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

regcv = linear_model.LassoCV(fit_intercept=False, alphas=alphas, positive = True, tol = 1e-3, max_iter=10000000)
regcv.fit(library_cv, Cd)

alpha = regcv.alpha_
R2_cv = regcv.score(library_cv, Cd)
rmse_cv = np.sqrt(mean_squared_error(Cd, regcv.predict(library_cv)))

inp_train, inp_test, Cd_train, Cd_test = train_test_split(inp, Cd, test_size=0.2, shuffle=True)
Re_train = inp_train[:,0]
phi_train = inp_train[:,1]
theta_train = inp_train[:,2]
Re_test = inp_test[:,0]
phi_test = inp_test[:,1]
theta_test = inp_test[:,2]

library_train = library(Re_train, phi_train, theta_train, deg, deg_trigo)
library_test = library(Re_test, phi_test, theta_test, deg, deg_trigo)
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
#percent_error = mean_absolute_percentage_error(Cd_test, reg.predict(library_test))
print("RMSE train: ",rmse_train)
print("RMSE test: ",rmse_test)
print("----")
print("----")


learned_dict = list(zip(dict, reg.coef_))
np.savetxt('Library/Inclined_Library/inclined_ellipsoid_learned_dictionary.csv', [p for p in learned_dict], delimiter=',', fmt='%s')

#############################
#### Truncated Model ########
#############################

df = pd.read_csv(r'Library/Inclined_Library/inclined_ellipsoid_learned_dictionary.csv', usecols=[1], header=None)
data = df.to_numpy()

data = np.round(data, 3)

print(data)

learned_dict = list(zip(dict, data))
# np.savetxt('Library/Inclined_Library/Inclined_obtained_formula.csv', [p for p in learned_dict], delimiter=',', fmt='%s')

###########################
#### Error ###############
###########################

path = r'data/ellipsoid_inclined_data' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    frame = pd.read_csv(filename, index_col=None, header=0)
    li.append(frame)

df = pd.concat(li, axis=0, ignore_index=True)

df = pd.DataFrame(df, columns= ['Re', 'phi', 'theta', 'Exp C_d'])

data = df.to_numpy()
Re = data[:,0]
phi = data[:,1]
theta = data[:,2]

def ellipsoid_inclined_analytic(Re, phi, theta):
    X = Re
    Y, Z = np.meshgrid(phi, theta)
    num_0 = 24 * 4 * (Y ** 2 - 1) * np.sqrt(Y ** 2 - 1)
    denom_0 = X * Y ** (2 / 3) * 3 * ((2 * Y ** 2 - 1) * np.log(Y + np.sqrt(Y ** 2 - 1)) - Y * np.sqrt(Y ** 2 - 1))
    Cd_exact_0 = num_0 / denom_0

    num_90 = 24 * 8 * (Y ** 2 - 1) * np.sqrt(Y ** 2 - 1)
    denom_90 = X * Y ** (2 / 3) * 3 * ((2 * Y ** 2 - 3) * np.log(Y + np.sqrt(Y ** 2 - 1)) + Y * np.sqrt(Y ** 2 - 1))
    Cd_exact_90 = num_90 / denom_90

    Cd_exact = Cd_exact_0 + (Cd_exact_90 - Cd_exact_0) * np.sin(Z) * np.sin(Z)
    return Cd_exact

def ellipsoid_inclined_learned(Re, phi, theta):
    X = Re
    Y, Z = np.meshgrid(phi, theta)
    Cd_learned = 12.885 / (X * Y) + 12.388 / X + 0.519 * (Y - 1) * Z * Z / X
    return Cd_learned

def ellipsoid_inclined_learned_vec(Re, phi, theta):
    Cd_learned = 12.885 / (Re * phi) + 12.388 / Re + 0.519 * (phi - 1) * theta * theta / Re
    return Cd_learned

Cd_simulation = data[:,3].reshape(-1,1)
Cd_analytic = ellipsoid_inclined_analytic(Re, phi, theta).reshape(-1,1)
Cd_learned = ellipsoid_inclined_learned(Re, phi, theta).reshape(-1,1)
Cd_learned_vec = ellipsoid_inclined_learned_vec(Re, phi, theta).reshape(-1,1)

#Compute the relative L2 error norm
relative_error_simulation = np.mean((Cd_learned_vec - Cd_simulation)**2) / np.mean(Cd_simulation**2)
print("Relative Error Simulation : ", relative_error_simulation*100)

relative_error_analytic = np.mean((Cd_learned - Cd_analytic)**2) / np.mean(Cd_analytic**2)
print("Relative Error Analytic : ", relative_error_analytic*100)

###########################
#### Plotting #############
###########################
# Create the figure and axis objects with reduced width
fig, ax = plt.subplots(figsize=(15, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

Cd_learned = ellipsoid_inclined_learned(Re, phi, theta)
Cd_analytic = ellipsoid_inclined_analytic(Re, phi, theta)
rel_err_analytic = (Cd_learned - Cd_analytic)**2 / Cd_analytic**2
per_err_analytic = rel_err_analytic*100

CS = plt.imshow(per_err_analytic, cmap='Spectral', extent=(phi.min(), phi.max(), theta.min(), theta.max()), origin='lower', aspect='auto', interpolation='bilinear')
cbar=plt.colorbar(CS)

tick_positions = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
tick_labels = ['0', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']
plt.gca().yaxis.set_major_locator(FixedLocator(tick_positions))
plt.gca().yaxis.set_major_formatter(FixedFormatter(tick_labels))

tick_positions_x = [2.0, 4.0, 6.0]
tick_labels_x = ['2.0', '4.0', '6.0']
plt.gca().xaxis.set_major_locator(FixedLocator(tick_positions_x))
plt.gca().xaxis.set_major_formatter(FixedFormatter(tick_labels_x))

plt.xlabel('$\phi$', fontsize = 12)
plt.ylabel('$θ$', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

scatter_meshgrid = np.meshgrid(phi_train, theta_train)
x_scat, y_scat = scatter_meshgrid
plt.scatter(x_scat, y_scat, s=20, c='black', marker='*')

plt.savefig('Figures/inclined_analytic_error.jpeg', dpi = 500, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(15, 5))  # You can adjust the width (7 inches) and height (5 inches) as needed

rel_err_sim = (Cd_learned_vec - Cd_simulation)**2 / Cd_simulation**2
per_err_sim = rel_err_sim*100
Re = Re.reshape(-1, )
phi = phi.reshape(-1, )
per_err_sim = per_err_sim.reshape(-1, )
CS = plt.tricontourf(phi, theta, per_err_sim, 2000, cmap='Spectral')
cbar = plt.colorbar(CS)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(12)

tick_positions = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
tick_labels = ['0', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']
plt.gca().yaxis.set_major_locator(FixedLocator(tick_positions))
plt.gca().yaxis.set_major_formatter(FixedFormatter(tick_labels))

tick_positions_x = [2.0, 4.0, 6.0]
tick_labels_x = ['2.0', '4.0', '6.0']
plt.gca().xaxis.set_major_locator(FixedLocator(tick_positions_x))
plt.gca().xaxis.set_major_formatter(FixedFormatter(tick_labels_x))

plt.xlabel('$\phi$', fontsize = 12)
plt.ylabel('$θ$', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
scatter_meshgrid = np.meshgrid(phi_train, theta_train)
x_scat, y_scat = scatter_meshgrid
plt.scatter(x_scat, y_scat, s=20, c='black', marker='*')

plt.savefig('Figures/inclined_simulation_error.jpeg', dpi = 500, bbox_inches='tight')
plt.show()
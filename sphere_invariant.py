import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

np.random.seed(142)

df = pd.read_csv(r'data/sphere_data/Sphere_20_Runs.csv')
df = pd.DataFrame(df, columns= ['Re','Exp C_d'])

frac = 1
shuffled_df = df.sample(frac=frac)

data = shuffled_df.to_numpy()
Re = data[:,0]
Cd = data[:,1]

eta = np.random.rand(20)
rho = np.random.rand(20)
len_a = np.random.rand(20)
U = Re*eta/(2*rho*len_a)

inp = np.concatenate((eta.reshape(-1,1), rho.reshape(-1,1), len_a.reshape(-1,1), U.reshape(-1,1)), axis=1)

deg = 2
counter = 0
dict = [None]*((2*deg+1))**4
for i in range(-deg, deg+1):
    for j in range(-deg, deg+1):
        for k in range(-deg, deg+1):
            for l in range(-deg, deg+1):
                dict[counter] = 'eta^'+str(i) +'rho^'+str(j) +'a^'+str(k) +'U^'+str(l)
                counter = counter +1

def library(eta, rho, len_a, U, deg):
    counter = 0
    lib = np.zeros((len(eta), (2*deg+1)**4))
    for i in range(-deg, deg+1):
        for j in range(-deg, deg+1):
            for k in range(-deg, deg + 1):
                for l in range(-deg, deg + 1):
                    lib[:,counter] = eta**i * rho**j * len_a**k * U**l
                    counter = counter +1
    return lib

library_cv = library(eta, rho, len_a, U, deg)
alphas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

regcv = linear_model.LassoCV(fit_intercept=False, alphas=alphas, positive = True, tol = 1e-4, max_iter=100000)
regcv.fit(library_cv, Cd)

alpha = regcv.alpha_
R2_cv = regcv.score(library_cv, Cd)
rmse_cv = np.sqrt(mean_squared_error(Cd, regcv.predict(library_cv)))

inp_train, inp_test, Cd_train, Cd_test = train_test_split(inp, Cd, test_size=0.2, shuffle=True)
eta_train = inp_train[:,0]
rho_train = inp_train[:,1]
len_a_train = inp_train[:,2]
U_train = inp_train[:,3]
eta_test = inp_test[:,0]
rho_test = inp_test[:,1]
len_a_test = inp_test[:,2]
U_test = inp_test[:,3]

library_train = library(eta_train, rho_train, len_a_train, U_train, deg)
library_test = library(eta_test, rho_test, len_a_test, U_test, deg)
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
np.savetxt('Library/Invariant/invariant_learned_dictionary.csv', [p for p in learned_dict], delimiter=',', fmt='%s')

#############################
#### Truncated Model ########
#############################

df = pd.read_csv(r'Library/Invariant/invariant_learned_dictionary.csv', usecols=[1], header=None)
data = df.to_numpy()
data = np.round(data, 3)

print(data)

learned_dict = list(zip(dict, data))
np.savetxt('Library/Invariant/invariant_obtained_formula.csv', [p for p in learned_dict], delimiter=',', fmt='%s')
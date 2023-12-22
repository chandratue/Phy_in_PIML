import numpy as np
import pdb

class Libraries:
    def __init__(self, mode, deg, deg_trigo):
        self.deg = deg
        self.deg_trigo = deg_trigo
        self.mode = mode

    def library(self, Re, phi, theta):
        if self.mode == 'phim1':
            terms1 = (2*self.deg+1) * (2*self.deg+1)
            terms2 = self.deg * (self.deg+1) * (2*self.deg+1)
            lib1 = np.zeros((len(Re), terms1))
            lib2 = np.zeros((len(Re), terms2))
            lib = np.zeros((len(Re), terms1+terms2))
            lib1_names = [None]*terms1
            lib2_names = [None]*terms2

            counter = 0
            for i in range(-self.deg, self.deg+1):
                for j in range(-self.deg, self.deg+1):
                    lib1[:,counter] = Re**i * phi**j
                    lib1_names[counter] = 'Re^'+str(i) +'phi^'+str(j)
                    counter = counter +1

            counter = 0
            for i in range(1, self.deg+1):
                for j in range(0, self.deg+1):
                    for k in range(-self.deg, self.deg+1):
                        lib2[:,counter] = Re**k * (phi-1)**i * theta**(j)
                        lib2_names[counter] = 'Re^'+str(k) + ' (phi-1)^'+str(i) +' theta^'+str(j)
                        counter = counter +1

            lib = np.concatenate((lib1, lib2), 1)
            lib_names = lib1_names + lib2_names


        elif self.mode == 'basic':
            terms3 = (2*self.deg+1) * (2*self.deg+1) * (2*self.deg+1)
            lib = np.zeros((len(Re), terms3))
            lib_names = [None]*terms3

            counter = 0
            for i in range(-self.deg, self.deg+1):
                for j in range(-self.deg, self.deg+1):
                    for k in range(0, self.deg+1):
                        lib[:,counter] = Re**i * phi**j * theta**k
                        lib_names[counter] = 'Re^'+str(i) + ' phi^'+str(j) +' theta^'+str(k)
                        counter = counter +1

        elif self.mode == 'basicphim1':
            terms3 = (2*self.deg+1) * (2*self.deg+1) * (2*self.deg+1)
            lib = np.zeros((len(Re), terms3))
            lib_names = [None]*terms3

            counter = 0
            for i in range(-self.deg, self.deg+1):
                for j in range(-self.deg, self.deg+1):
                    for k in range(0, self.deg+1):
                        lib[:,counter] = Re**i * (phi-1)**j * theta**k
                        lib_names[counter] = 'Re^'+str(i) + ' (phi-1)^'+str(j) +' theta^'+str(k)
                        counter = counter +1

        elif self.mode == 'trig':
            terms3 = (2*self.deg+1) * (2*self.deg+1) * (self.deg_trigo+1)
            lib = np.zeros((len(Re), terms3))
            lib_names = [None]*terms3

            counter = 0
            for i in range(-self.deg, self.deg+1):
                for j in range(-self.deg, self.deg+1):
                    for k in range(0, self.deg_trigo+1):
                            lib[:,counter] = Re**i * phi**j * np.cos(k*theta)
                            lib_names[counter] = 'Re^'+str(i) + ' phi^'+str(j) +'cos('+str(k)+'theta)'
                            counter = counter +1

        return lib, lib_names

import numpy as np
import matplotlib.pyplot as plt

'''
This module defines a class `StructureFactor` that calculates the structure factor for a paracrystal.

Attributes:
    D (float): The lattice constant of the paracrystal.

Methods:
    phi_paracrystal(q, sigma):
        Calculates the paracrystal form factor for a given wave vector `q` and standard deviation `sigma`. sigma < 0.5.
    
    structure_factor_paracrystal(q, phi_q):
        Calculates the structure factor for a paracrystal for a given wave vector `q` and paracrystal form factor `phi_q`.
    
    paracrystal(q, sigma):
        Calculates the absolute value of the structure factor for a paracrystal for a given wave vector `q` and standard deviation `sigma`.
'''

class StructureFactor:
    def __init__(self, D):
        self.D = D

    def phi_paracrystal(self, q, sigma):
        return np.exp(np.pi * (q ** 2) * (sigma ** 2))

    def structure_factor_paracrystal(self, q, phi_q):
        return (1 - phi_q ** 2) / (1 + (phi_q ** 2) - 2 * phi_q * np.cos(q * self.D))

    def paracrystal(self, q, sigma):
        phi_q = self.phi_paracrystal(q, sigma)
        S = np.abs(self.structure_factor_paracrystal(q, phi_q))
        return S
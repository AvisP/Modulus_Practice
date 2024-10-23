import os
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, Eq, Abs, Function, Number, Piecewise, exp, pi, cos
from modulus.sym.eq.pde import PDE

# Define the PDE
class wavePDE (PDE):
    name = "AcousticWaveEquation "
    def __init__(self, u="u", c="c", S="S", dim=3, time=True, mixed_form=False):
        # set params
        self.u = u
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Scalar function (make u function)
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)
            
        # Source Term
        if type(S) is str:
            S = Function(S)(*input_variables)
        elif type(S) in [float, int]:
            S = Number(S)

        # set equations
        self.equations = {}

        if not self.mixed_form: 
            self.equations["wave_PDE"] = (
                u.diff(t, 2)
                - c**2 * u.diff(x, 2)
                - c**2 * u.diff(y, 2)
                - c**2 * u.diff(z, 2)
                - S
            )
        elif self.mixed_form:
            u_x = Function("u_x")(*input_variables)
            u_y = Function("u_y")(*input_variables)
            if self.dim == 3:
                u_z = Function("u_z")(*input_variables)
            else:
                u_z = Number(0)
            if self.time:
                u_t = Function("u_t")(*input_variables)
            else:
                u_t = Number(0)

            self.equations["wave_PDE"] = (
                u_t.diff(t)
                - c**2 * u_x.diff(x)
                - c**2 * u_y.diff(y)
                - c**2 * u_z.diff(z)
            )
            self.equations["compatibility_u_x"] = u.diff(x) - u_x
            self.equations["compatibility_u_y"] = u.diff(y) - u_y
            self.equations["compatibility_u_z"] = u.diff(z) - u_z
            self.equations["compatibility_u_xy"] = u_x.diff(y) - u_y.diff(x)
            self.equations["compatibility_u_xz"] = u_x.diff(z) - u_z.diff(x)
            self.equations["compatibility_u_yz"] = u_y.diff(z) - u_z.diff(y)
            if self.dim == 2:
                self.equations.pop("compatibility_u_z")
                self.equations.pop("compatibility_u_xz")
                self.equations.pop("compatibility_u_yz")
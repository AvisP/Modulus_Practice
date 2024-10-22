class Bioreactor(PDE):
    """

    Parameters
    ==========
    z : float, string
        Wave speed coefficient. If a string then the
        wave speed is input into the equation.
    """

    name = "Bioreactor"

    def __init__(self):
        # coordinates
        z = Symbol("z")
        r = Symbol("r")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"z": z, "r": r, "t": t}

        # make u function
        v_z = Function("v_z")(*input_variables)
        # X_w = Function("X_w")(*input_variables)
        # Y_w = Function("Y_w")(*input_variables)
        # T_g = Function("T_g")(*input_variables)
        # T_s = Function("T_s")(*input_variables)
        # rho_ds = Function("Y_w")(*input_variables)
        # b = Function("b")(*input_variables)

        DF = Number(1.1098) ## rho_f
        # epsilon = function(r)
        # epsilon = Number(0.01)
        AP = Number(2.53885523177210*(3600.**2.))  ## P    
        UF = Number(0.06658) ## mu_eff
        DP = Number(0.01)  ## dp
        alpha = Number(137.53169) 
        beta = Number(2.1)
        RG = Number(9.8*(3600.**2.)) ## g_z
        
        # Given parameters
        # dp_val = 0.01
        h_val = Number(0.005)
        R_val = Number(0.05179 / 2)
        emin_val = Number(0.1518)
        e0_val = Number(0.39)
        c_val = Number(2)
        b_val = Number(0.876)
        N_val = Number(1000)

        # Derived variables
        dpeff = (3 / 2 * DP**2 * h_val)**(1 / 3) 
        ao = 1.8 - 2 * (dpeff / R_val) 
        # Let dt remain symbolic (we don't know it yet)# r_dash expression
        r_dash = ao * ((R_val - r) / dpeff) - 1
        # Conditional logic for 'e'
        epsilon = Piecewise(
            (emin_val + (1 - emin_val) * r_dash**4, r_dash < 0),
            (e0_val + (emin_val - e0_val) * exp(-r_dash / c_val) * cos(pi * r_dash / b_val), r_dash >= 0)
        )


        K = ((epsilon**2)*(DP**2))/(alpha*(1-epsilon)**2)
        K_z = ((epsilon**3)*DP)/(beta*(1-epsilon))
        
        # Bioreactor coefficient
        # if type(c) is str:
        #     c = Function(c)(*input_variables)
        # elif type(c) in [float, int]:
        #     c = Number(c)

        # set equations
        self.equations = {}
        self.equations["material_balance_eq"] = DF * v_z.diff(z)
        self.equations["reynold_eq"] = DF * ((epsilon * v_z).diff(t) + (epsilon * v_z) * v_z.diff(z)) \
        + (epsilon * AP).diff(z) \
        - UF * ((epsilon * v_z).diff(z, 2) + (1/r) * (epsilon * v_z).diff(r) + (epsilon * v_z).diff(r, 2)) \
        + (UF/K) * epsilon * v_z \
        + (DF/K_z) * (epsilon**2) * (v_z**2) \
        + epsilon * DF * RG

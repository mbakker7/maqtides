import numpy as np

class tidal1d:
    """
    Class to compute tidal propagation in a single layer aquifer

    Parameters
    ----------
    Tsea : float
        transmissivity of aquifer below the sea
    Tland : float
        transmissivity of aquifer below the land
    Ssea : float
        storage coefficient of aquifer below the sea
    Sland : float
        storage coefficient of aquifer below the land
    csea : float
        resistance of leaky layer below the sea
    cland : float
        resistance of leaky layer below the sea
    sigsea : float
        storage coefficient of leaky layer below the sea
    sigland : float
        storage coefficient of leaky layer below the land
    beta : float
        loading efficiency of aquifer below the sea
    gamma : float
        loading efficiency of leaky layer below the sea
    tau : float
        tidal period in sea
    hs : float
        tidal amplitude in sea
    cos : boolean
        True if cosine part, False if sine part

    """

    def __init__(self, Tsea, Tland, Ssea, Sland, csea, cland,
                 sigsea, sigland, beta, gamma, tau, hs, cos=True):
        self.Tsea = Tsea
        self.Tland = Tland
        self.Ssea = Ssea
        self.Sland = Sland
        self.csea = csea
        self.cland = cland
        self.sigsea = sigsea
        self.sigland = sigland
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.hs = hs
        self.omega = 2 * np.pi / self.tau
        self.cos = cos
        self.labsea = np.sqrt(1j * self.omega * self.sigsea * self.csea)
        self.labland = np.sqrt(1j * self.omega * self.sigland * self.cland)
        if np.abs(self.labsea) < 1e-12:
            self.fsea = 1 / self.csea
            self.gsea = 1 / self.csea
        else:
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore', r'overflow encountered in sinh')
                if np.isinf(np.sinh(self.labsea)):
                    self.fsea = 0 + 0j
                else:
                    self.fsea = self.labsea / np.sinh(self.labsea) / self.csea
            self.gsea = self.labsea / np.tanh(self.labsea) / self.csea
        if np.abs(self.labland) < 1e-12:
            self.fland = 1 / self.cland
            self.gland = 1 / self.cland
        else:
            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore', r'overflow encountered in sinh')
                if np.isinf(np.sinh(self.labland)):
                    self.fland = 0 + 0j
                else:
                    self.fland = self.labland / np.sinh(self.labland) / self.cland
            self.gland = self.labland / np.tanh(self.labland) / self.cland
        # Needed for testing
        self.phip = (self.fsea + (self.gsea - self.fsea) * self.gamma + 1j * self.omega * self.Ssea * self.beta) / \
                    (self.gsea + 1j * self.omega * self.Ssea) * self.hs
        #
        self.alphasea = (self.gsea + 1j * self.omega * self.Ssea) / self.Tsea
        self.alphaland = (self.gland + 1j * self.omega * self.Sland) / self.Tland
        self.a  = -self.Tland * np.sqrt(self.alphaland) / (self.Tsea * np.sqrt(self.alphasea) + self.Tland * np.sqrt(self.alphaland)) * self.phip
        self.b  =  self.Tsea * np.sqrt(self.alphasea) / (self.Tsea * np.sqrt(self.alphasea) + self.Tland * np.sqrt(self.alphaland)) * self.phip

    def phi(self, x):
        """
        Function to compute phi
    
        Input
        -----
        x : float or array of x values

        Returns
        -------
        phi : array of complex phi values shape (nx) or squeezed shape
    
        """
        
        x = np.atleast_1d(x)
        phi = np.zeros(len(x), 'D')
        phi[x < 0] = (self.phip + self.a * np.exp(x[x < 0] * np.sqrt(self.alphasea)))
        phi[x >= 0] = self.b * np.exp(-x[x >= 0] * np.sqrt(self.alphaland))
        return phi
    
    def headcomplex(self, x, t):
        """
        Function to compute the complex head
    
        Input
        -----
        x : float or array of x values

        Returns
        -------
        head : array of complex head values shape (nt, nx) or squeezed shape
    
        """
        t = np.atleast_1d(t)
        t = t[:, np.newaxis]
        h = self.phi(x) * np.exp(1j * self.omega * t)
        return np.squeeze(h)
    
    def head(self, x, t):
        """
        Function to compute head
    
        Input
        -----
        x : float or array of x values

        Returns
        -------
        head : array of complex head values shape (nt, nx) or squeezed shape
    
        """
        h = self.headcomplex(x, t)
        if self.cos:
            return h.real
        else:
            return h.imag

    def phase_amp(self, x):
        phi = self.phi(x)
        return -np.angle(phi) / self.omega, np.abs(phi)

    def phase(self, x):
        return self.phase_amp(x)[0]

    def amp(self, x):
        return self.phase_amp(x)[1]

    def numcheck(self, x, t, dx, dt):
        '''Numerical check of left and right sides of differential equation using
        complex heads.
        Note that this can be performed with real mathematics if the storage in the leaky
        layers equals zero (otherwise the vertical flux into the leaky layer can
        not be computed easily with real math)'''
        h = self.headcomplex(x, t)
        hpx = self.headcomplex(x + dx, t)
        hmx = self.headcomplex(x - dx, t)
        hpt = self.headcomplex(x, t + dt)
        hmt = self.headcomplex(x, t - dt)
        d2hdx2 = (hpx - 2 * h + hmx) / (dx ** 2)
        dhdt = (hpt - hmt) / (2 * dt)
        h0 = self.hs * np.exp(1j * self.omega * t)
        if x <= 0:
            lhs = self.Tsea * d2hdx2
            qt = self.gsea * h - (self.fsea + (self.gsea - self.fsea) * self.gamma) * h0
            rhs = self.Ssea * dhdt + qt - \
                  (self.beta * self.Ssea) * 1j * self.omega * h0
            rhs = rhs
        elif x >= 0:
            lhs = self.Tland * d2hdx2
            qz = self.gland * h
            rhs = self.Sland * dhdt + qz
            rhs = rhs
        print('lhs', lhs)
        print('rhs', rhs)

    def bccheck(self, t, dx):
        headleft = self.head(-1e-6, t)
        headright = self.head(1e-6, t)
        print('head left', headleft)
        print('head right', headright)
        hleft = self.head(-2 * dx, t)
        hright = self.head(-dx, t)
        dhdx1 = (hright - hleft) / dx
        hleft = self.head(dx, t)
        hright = self.head(2 * dx, t)
        dhdx2 = (hright - hleft) / dx
        print('Qx1', -self.Tsea * dhdx1)
        print('Qx2', -self.Tland * dhdx2)

k = 10.0
H = 40.0
Tsea = k * H
Tland = k * H * 2
Ss = 1e-4
Ssea = Ss * H
Sland = Ss * H * 2
csea = 1000
cland = 1000 * 2
sigsea = 0.5 * Ssea
sigland = 0.5 * Sland * 2
beta = 0.5
gamma = 0.5
tau = 0.5 # days
ml = tidal1d(Tsea, Tland, Ssea, Sland, csea, cland, sigsea, sigland, beta, gamma, tau, hs=1)

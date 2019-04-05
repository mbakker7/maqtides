import numpy as np
from scipy.linalg import expm, sqrtm
from numpy.linalg import inv

class tidalmaq:
    def __init__(self, Tsea, Tland, Ssea, Sland, csea, cland, sigsea, sigland, beta, gamma, tau, hs, cos=True):
        """
        Class to compute tidal propagation in a multi-layer system
    
        Parameters
        ----------
        Tsea : array of length N
            transmissivity of aquifer below the sea
        Tland : array of length N
            transmissivity of aquifer below the land
        Ssea : array of length N
            storage coefficient of aquifer below the sea
        Sland : array of length N
            storage coefficient of aquifer below the land
        csea : array of length N
            resistance of leaky layer below the sea
        cland : float
            resistance of leaky layer below the sea
        sigsea : array of length N
            storage coefficient of leaky layer below the sea
        sigland : array of length N
            storage coefficient of leaky layer below the land
        beta : array of length N
            loading efficiency of aquifer below the sea
        gamma : array of length N
            loading efficiency of leaky layer below the sea
        tau : float
            tidal period in sea
        hs : float
            tidal amplitude in sea
        cos : boolean
            True if cosine part, False if sine part
    
        """

        N = len(Tsea)
        self.N = N
        self.hs = hs * np.ones(N)
        #
        self.tau = tau
        self.omega = 2 * np.pi / tau
        self.gamma = gamma
        self.cos = cos
        # Matrices
        self.Tsea = np.diag(Tsea)
        self.Tland = np.diag(Tland)
        self.Ssea = np.diag(Ssea)
        self.Sland = np.diag(Sland)
        self.B = np.diag(beta)
        # Arrays
        self.csea = csea
        self.cland = cland
        self.sigsea = sigsea
        self.sigland = sigland
        self.labsea = np.sqrt(1j * self.omega * sigsea * csea)
        self.labland = np.sqrt(1j * self.omega * sigland * cland)
        fsea = np.zeros(N, 'D')
        fland = np.zeros(N, 'D')
        gsea = np.zeros(N, 'D')
        gland = np.zeros(N, 'D')
        for i in range(N):
            if np.abs(self.labsea[i]) < 1e-12:
                fsea[i] = 1 / self.csea[i]
                gsea[i] = 1 / self.csea[i]
            else:
                with np.warnings.catch_warnings():
                    np.warnings.filterwarnings('ignore', r'overflow encountered in sinh')
                    if np.isinf(np.sinh(self.labsea[i])):
                        fsea[i] = 0 + 0j
                    else:
                        fsea[i] = self.labsea[i] / np.sinh(self.labsea[i]) / self.csea[i]
                gsea[i] = self.labsea[i] / np.tanh(self.labsea[i]) / self.csea[i]
            if np.abs(self.labland[i]) < 1e-12:
                fland[i] = 1 / self.cland[i]
                gland[i] = 1 / self.cland[i]
            else:
                with np.warnings.catch_warnings():
                    np.warnings.filterwarnings('ignore', r'overflow encountered in sinh')
                    if np.isinf(np.sinh(self.labland[i])):
                        fland[i] = 0 + 0j
                    else:
                        fland[i] = self.labland[i] / np.sinh(self.labland[i]) / self.cland[i]
                gland[i] = self.labland[i] / np.tanh(self.labland[i]) / self.cland[i]
        self.Fsea = np.diag(-gsea[1:], 1) + np.diag(-fsea[1:], -1)
        self.Fsea[np.arange(N - 1), np.arange(N - 1)] = gsea[:-1] + gsea[1:]
        self.Fsea[-1, -1] = gsea[-1]
        self.Fland = np.diag(-gland[1:], 1) + np.diag(-fland[1:], -1)
        self.Fland[np.arange(N - 1), np.arange(N - 1)] = gland[:-1] + gland[1:]
        self.Fland[-1, -1] = gland[-1]
        self.G = np.zeros((N, N), 'D')
        self.G[0, 0] = fsea[0] + (gsea[0] - fsea[0]) * gamma[0] + (gsea[1] - fsea[1]) * gamma[1]
        self.G[np.arange(1, N - 1), np.arange(1, N - 1)] = (gsea[1:N - 1] - fsea[1:N - 1]) * gamma[1:N - 1] + (gsea[2:] - fsea[2:]) * gamma[2:]
        self.G[-1, -1] = (gsea[-1] - fsea[-1]) * gamma[-1]
        #
        Tseainv = inv(self.Tsea)
        Tlandinv = inv(self.Tland)
        Asea = Tseainv @ (self.Fsea + 1j * self.omega * self.Ssea)
        Aland = Tlandinv @ (self.Fland + 1j * self.omega * self.Sland)
        self.sqrtAsea = sqrtm(Asea)
        self.sqrtAland = sqrtm(Aland)
        self.TseasqrtAsea = self.Tsea @ self.sqrtAsea
        self.TlandsqrtAland = self.Tland @ self.sqrtAland 
        I = np.diag(np.ones(N))
        self.phip = inv(self.Fsea + 1j * self.omega * self.Ssea) @ (self.G + 1j * self.omega * self.Ssea @ self.B) @ self.hs
        self.a = -inv(inv(self.TlandsqrtAland) @ self.TseasqrtAsea + I) @ self.phip
        self.b =  inv(inv(self.TseasqrtAsea) @ self.TlandsqrtAland + I) @ self.phip
        
    def phi(self, x):
        x = np.atleast_1d(x)
        phi = np.zeros((self.N, len(x)), 'D')
        for i in range(len(x)):
            if x[i] <= 0:
                phi[:, i] = (self.phip + expm(x[i] * self.sqrtAsea) @ self.a)
            else:
                phi[:, i] = expm(-x[i] * self.sqrtAland) @ self.b
        return phi
    
    def headcomplex(self, x, t):
        x = np.atleast_1d(x)
        t = np.atleast_1d(t)
        nx = len(x)
        nt = len(t)
        h = np.zeros((nt, self.N, nx), 'D')
        phi = self.phi(x)
        for i in range(nt):
            h[i] = phi * np.exp(1j * self.omega * t[i])
        return np.squeeze(h)

    def head(self, x, t):
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
        h = self.headcomplex(x, t)
        hpx = self.headcomplex(x + dx, t)
        hmx = self.headcomplex(x - dx, t)
        hpt = self.headcomplex(x, t + dt)
        hmt = self.headcomplex(x, t - dt)
        d2hdx2 = (hpx - 2 * h + hmx) / (dx ** 2)
        dhdt = (hpt - hmt) / (2 * dt)
        h0 = self.hs * np.exp(1j * self.omega * t)
        if x >= 0:
            lhs = self.Tland @ d2hdx2
            rhs = self.Sland @ dhdt + self.Fland @ h
        elif x <= 0:
            lhs = self.Tsea @ d2hdx2
            rhs = self.Ssea @ dhdt + self.Fsea @ h - self.G @ h0
            rhs -= (self.B @ self.Ssea) @ h0 * 1j * self.omega
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
        Qxleft = -np.diag(self.Tsea) * dhdx1
        Qxright = -np.diag(self.Tland) * dhdx2
        print('Qxleft', Qxleft)
        print('Qxright', Qxright)
        
Naq = 4
k = 10 * np.ones(Naq)
H = 10 * np.ones(Naq)
Tsea = k * H
Tland = k * H
Ssea = 1e-4 * np.ones(Naq)
Sland = 1e-4 * np.ones(Naq)
beta = 0.6 * np.ones(Naq)
csea = 100 * np.ones(Naq)
cland = 100 * np.ones(Naq)
sig1 = 0.5e-4 * np.ones(Naq)
sig2 = 0.5e-4 * np.ones(Naq)
gamma = 0.8 * np.ones(Naq)
tau = 0.5
hs = 1
ml = tidalmaq(Tsea, Tland, Ssea, Sland, csea, cland, sig1, sig2, beta, gamma, tau, hs)
#mlold = tidalmaq_old(Tsea, Ssea, Sland, csea, cland, beta, tau, hs)
    
# N = 4
# k = 10.0 * np.ones(N)
# #k[5] = 1
# H = 1.0 * np.ones(N)
# T = k * H
# Ss = 1e-4 * np.ones(N)
# Ssea = Ss * H
# Sland = Ssea.copy()
# Sland[0] = 0.15
# aniso = 0.1
# csea = H / (k * aniso)
# csea[0] = 0.5 * csea[0]
# cland = csea.copy()
# cland[0] = np.inf
# beta = 1 * np.ones(N)
# beta[0] = 0.5
# beta[2] = 0.25
# tau = 0.5 # days
# #
# ml = tidalmaq(T, Ssea, Sland, csea, cland, beta, tau, hs=1)
# mlnew = tidalmaqnew(T, Ssea, Sland, csea, cland, beta, tau, hs=1)

    
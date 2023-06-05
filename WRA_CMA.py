import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from functools import partial
try:
    import ddcma
    from ddcma import DdCma 
except:
    raise ModuleNotFoundError("`ddcma` not found. Download it from https://gist.github.com/youheiakimoto/1180b67b5a0b1265c204cba991fa8518")        
try:
    from adversarial_cmaes import *
except:
    raise ModuleNotFoundError("`adversarial_cmaes` not found. Download it from https://gist.github.com/youheiakimoto/ab51e88c73baf68effd95b750100aad0")        

## If test functions on the paper is used, "journal_minmaxf.py" is required 
import journal_minmaxf as ff


class MyDdCma(DdCma): 
    def set_dynamic_parameters(self, init_matrix, init_S, init_sigma):
        self.Z = np.array(init_matrix[0,:,:], copy=True)
        self.C = np.array(init_matrix[1,:,:], copy=True)
        self.B = np.array(init_matrix[2,:,:], copy=True)
        self.sqrtC = np.array(init_matrix[3,:,:], copy=True)
        self.invsqrtC = np.array(init_matrix[4,:,:], copy=True)
        self.S = np.array(init_S[:], copy=True)   
        self.sigma = init_sigma.copy()
        pass

def mirror(z, lbound, ubound):
    """ Mirroring constraint handling 

    Parameters
    ----------
    z : np.ndarray (1D)
        solution vector to be mirrorred
    lbound, ubound : np.ndarray (1D)
        lower and upper bound of the box constraint

    Returns
    -------
    np.ndarray (1D) : mirrorred solution
    """

    width = ubound - lbound
    return ubound - np.abs(np.mod(z - lbound, 2 * width) - width)


class WRA:
    """ Worst-case Raking Approximation (WRA) : approximation of the worst-case ranking for the given solution candidates by approximately solving internal maximization problem max_{y} f(x, y) with using CMA-ES """

    def __init__(self, f, init_x, lambdax, yb, boundy, tau_thr, cmax, Vymin, Cymax, Tmin, init_y, init_ymean, init_D_y, init_matrix, init_S, init_sigma, p, pp, pn, p_thr):
        """ Initialization of WRA
        Parameters
        ----------
        f : callable f (a solution candidate, a scenario variable) -> float
            objective function
        init_x : np.ndarray (1D)
            initial mean vector of the outer CMA-ES
        lambdax : int
            number of solution candidates in each iteration on the outer CMA-ES
        yb : np.ndarray (2D)
            lower and upper bound of the box constraint
        boundy : bool
            the domain for scenario variable is bounded (== initialization interval) if it is True
        tau_thr : float, , 0 < tau_thr <= 1
            threshold of Kendall's tau for early stopping strategy
        cmax : int
            parameter for interrupting CMA-ES. If worse scenario is found cmax times, maximaization for y is interrupted.
        Vymin : float
            termination parameter about maximum standard deviation of the inner CMA-ES
        Cymax : float
            termination parameter about condition number of the inner CMA-ES
        Tmin : int
            minimum iteration for the inner CMA-ES
        p :  np.ndarray (1D)
            vector to decide which configuration is refreshed
        pp : float, , 0 < pp <= 1
            parameter for positive update of p
        pn : float, 0 < pn <= 1
            parameter for negative update of p
        p_thr : float, 0 < p_thr <= 1
            threshold parameter for refresh configurations

        Following parameters are configuration parameters for the inner CMA-ES        
        init_y : np.ndarray (2D)
            initial candidates of scenario vector
        init_ymean : np.ndarray (2D)
            initial mean vector of the inner CMA-ES   
        init_D_y : np.ndarray (2D)
            initial coordinate-wise std of the inner CMA-ES  
        Dynamic parameters of the inner CMA-ES
        init_matrix : np.ndarray (4D)
        init_S : np.ndarray (2D)
        init_sigma : np.ndarray (1D)
        Followings are i-th dynamic parameters which are written in DD-CMA
            init_matrix[i, 0, :,:] = Z
            init_matrix[i, 1, :,:] = C
            init_matrix[i, 2, :,:] = B
            init_matrix[i, 3, :,:] = sqrt.C
            init_matrix[i, 4, :,:] = invsqrt.C
            init_S[i, :] = S
            init_sigma[i] = sigma
        """
        self.f = f
        self.ymember = init_y.shape[0]
        self.lambdax = lambdax
        self.id = 0
        self.solution = np.zeros((lambdax, len(init_x)))
        for i in range(lambdax):
            self.solution[i, :] = init_x[:]
        
        self.n = init_y.shape[1]
        self.yb = yb
        self.tau_thr = tau_thr
        self.cmax = cmax
        self.Vymin = Vymin
        self.Cymax = Cymax
        self.Tmin = Tmin
        self.fcalls = 0
        self.boundy = boundy
        self.tau = -1.0
        self.wrac = []
        self.wraneval = []
        self.Fnew = []
        self.kworst = []
        
        self.init_y = init_y
        self.init_ymean = init_ymean
        self.init_D_y = init_D_y
        
        self.yy = init_y
        self.ymean = init_ymean 
        self.D_y = init_D_y
        self.init_matrix = init_matrix
        self.init_S = init_S
        self.init_sigma = init_sigma
                
        ## probability control
        self.p = p
        self.pp = pp
        self.pn = pn
        self.p_thr = p_thr
        
    def __call__(self, arx):
        return self.wra_ddcma(arx)

    def max_f(self, y):
        fv = -1*f(self.solution[self.id, :], y)
        return fv
    
    def wra_ddcma(self, x_t):
        
        """approximating the worst-case ranking of solution candidate 
        Parameters
        ----------
        x^t : np.ndarray (2D)
            solution candidates in iteration t 

        Return
        idr : np.ndarray (1D)
            approximated ranking of the given solution candidates x^t
        """
        
        self.fcalls = 0
        self.worst = []
        ### Initialization part
        fx_arr = np.array([[self.f(x, y) for y in self.yy] for x in x_t])
        self.kworst = np.argmax(fx_arr, axis=1)
        Sworst = np.unique(self.kworst)
        Fold = np.max(fx_arr, axis=1)
        self.Fnew = Fold.copy()
        self.fcalls += self.ymember * self.lambdax

        ymean_tilde = np.zeros((self.lambdax, self.n))
        D_y_tilde = np.zeros((self.lambdax, self.n))
        y_tilde = np.zeros((self.lambdax, self.n))
        init_matrix_tilde = np.zeros((self.lambdax, 5, self.n, self.n))
        init_S_tilde = np.ones((self.lambdax, self.n))
        init_sigma_tilde = np.ones(self.lambdax)
        
        for i in range(self.lambdax):
            self.solution[i, :] = x_t[i, :]
            y_tilde[i, :] = self.yy[self.kworst[i], :]
            ymean_tilde[i, :] = self.ymean[self.kworst[i], :] 
            D_y_tilde[i, :] = self.D_y[self.kworst[i], :]
            init_matrix_tilde[i, :, :, :] = self.init_matrix[self.kworst[i], :, :, :] 
            init_S_tilde[i, :] = self.init_S[self.kworst[i], :]
            init_sigma_tilde[i] = self.init_sigma[self.kworst[i]]
            
        cma_instances = [MyDdCma(ymean_tilde[i, :], D_y_tilde[i, :], flg_variance_update=False, beta_eig=10*self.n**2) for i in range(self.lambdax)]
        for i in range(self.lambdax): 
            cma_instances[i].set_dynamic_parameters(init_matrix_tilde[i, :, :, :], init_S_tilde[i, :], init_sigma_tilde[i])

        ###    Worst-case Ranking Approximation
        h = np.ones(self.lambdax, dtype=bool)
        updaten = np.zeros(self.lambdax)
        tt = np.zeros(self.lambdax)
        self.wrac = np.zeros(self.lambdax)
        self.wraneval = np.zeros(self.lambdax)
        self.tau = -1.0
        while self.tau <= self.tau_thr:
            for i in range(self.lambdax):
                D_y_rd = np.zeros( self.n)
                init_matrix_rd = np.zeros((5, self.n, self.n))
                init_S_rd = np.ones(self.n)
                init_sigma_rd = 1
                if h[i]==True:
                    wracma = cma_instances[i]

                    D_y_rd[:]= np.array(wracma.D, copy=True)
                    init_matrix_rd[0, :,:] = np.array(wracma.Z, copy=True)
                    init_matrix_rd[1, :,:] = np.array(wracma.C, copy=True)
                    init_matrix_rd[2, :,:] = np.array(wracma.B, copy=True)
                    init_matrix_rd[3, :,:] = np.array(wracma.sqrtC, copy=True)
                    init_matrix_rd[4, :,:] = np.array(wracma.invsqrtC, copy=True)
                    init_S_rd[:] = np.array(wracma.S, copy=True)
                    init_sigma_rd = wracma.sigma

                    Fdash = Fold[i]
                    updaten[i] += 1
                    c = 0
                    self.id = i
                    while c<self.cmax:
                        wrax, wray, wraz = wracma.sample() 
                        wracy = np.zeros((wracma.lam, self.n))
                        fy = np.zeros(wracma.lam)
                        for k in range(wracma.lam):
                            wracy[k, :]=wrax[k, :]
                            if self.boundy==True:
                                wracy[k, :]= mirror(wrax[k, :], self.yb[0, :], self.yb[1, :])
                            fy[k] = self.f(x_t[i,:], wracy[k,:])
                            self.fcalls +=1
                            self.wraneval[i] += 1 
                        
                        if max(fy) > Fdash:
                            yid = np.argmax(fy)
                            y_tilde[i, :] = wracy[yid, :]
                            Fdash = max(fy)
                            c += 1
                            self.wrac[i] += 1
                            
                        idy = np.argsort(-fy)
                        wracma.update(idy, wrax, wray, wraz)
                        tt[i] +=1

                        if np.max(wracma.S) / np.min(wracma.S) > self.Cymax:
                            h[i] = False
                            cma_instances[i].set_dynamic_parameters(init_matrix_rd[:, :, :], init_S_rd[:], init_sigma_rd)
                            break

                        wraD_y = np.array(wracma.D[:], copy=True) 
                        for k in range(self.n):
                            wracma.D[k] = min(wraD_y[k], (self.yb[1, k] - self.yb[0, k])/4/wracma.sigma)
                            
                        if np.max(wracma.coordinate_std) < self.Vymin and tt[i] >=self.Tmin:
                            wraD_y = np.array(wracma.D[:], copy=True) 
                            h[i] = False
                            for k in range(self.n):
                                wracma.D[k] = max(wraD_y[k], self.Vymin/wracma.sigma)                          
                            break

                    self.Fnew[i] = Fdash

            self.tau, p_value = stats.kendalltau(self.Fnew, Fold)
            Fold[:] = self.Fnew[:]

        for i in range(self.lambdax):
            wracma = cma_instances[i]
            ymean_tilde[i, :] = wracma.xmean
            D_y_tilde[i, :]= wracma.D
            init_matrix_tilde[i, 0, :,:] = wracma.Z
            init_matrix_tilde[i, 1, :,:] = wracma.C
            init_matrix_tilde[i, 2, :,:] = wracma.B
            init_matrix_tilde[i, 3, :,:] = wracma.sqrtC
            init_matrix_tilde[i, 4, :,:] = wracma.invsqrtC
            init_S_tilde[i, :] = wracma.S 
            init_sigma_tilde[i] = wracma.sigma

        idr = np.argsort(self.Fnew)
        
        ###  postprocess part 
        pnmember = list(range(self.ymember))
        for l in range(len(Sworst)):
            self.p[Sworst[l]] = min(self.p[Sworst[l]]+self.pp,1)
            pnmember = [s for s in pnmember if s != Sworst[l]]
            Fworst = []
            Fworst_id = []
            for i in range(self.lambdax):
                if self.kworst[i] == Sworst[l]:
                    Fworst.append(self.Fnew[i])
                    Fworst_id.append(i)
            if len(Fworst) ==1:
                ridy=Fworst_id[0]
            else:
                dum=np.argmin(Fworst)
                ridy=Fworst_id[dum]
                
            self.yy[Sworst[l],:] = y_tilde[ridy,:]
            self.ymean [Sworst[l],:] = ymean_tilde[ridy, :]
            self.init_matrix[Sworst[l], :, :, : ] = init_matrix_tilde[ridy, :, :, : ] 
            self.init_S[Sworst[l], :] = init_S_tilde[ridy, :]
            self.init_sigma[Sworst[l]] = init_sigma_tilde[ridy]
            self.D_y[Sworst[l], :] = D_y_tilde[ridy, :]


        
        for k in pnmember:
            self.p[k] -= self.pn
            if self.p[k] <= self.p_thr:
                self.reset_y(k)
        return idr

    ## initialization of the configuration parameters for the inner CMA-ES
    def reset_y(self, j):
        self.init_matrix[j, :, :, : ], self.init_S[j, :], self.init_sigma[j] = self.set_init_dp()
        self.ymean[j, :] = np.random.rand(self.n)*(self.yb[1, :] - self.yb[0, :]) + self.yb[0, :]
        self.D_y[j, :] = (self.yb[1, :]- self.yb[0, :])/4
        yz = np.random.randn(self.n)
        for i in range(self.n):
            self.yy[j, i] = min( max(yz[i] * self.D_y[j, i] + self.ymean[j, i], self.yb[0, i]), self.yb[1, i])
        self.p[j] = 1.0

    ## set initial dynamic parameters for the inner CMA-ES
    def set_init_dp (self):
        init_matrix = np.zeros ((5, self.n, self.n))
        init_S = np.zeros (self.n)
        init_matrix[0,:,:] = np.zeros((self.n, self.n))   # Z
        init_matrix[1,:,:] = np.eye(self.n)  #  C
        init_matrix[2,:,:] = np.eye(self.n)  #  B
        init_matrix[3,:,:] = np.eye(self.n)  #  sqrtC
        init_matrix[4,:,:] = np.eye(self.n)  #  invsqrtC

        init_S[:] = np.ones(self.n)   # S 
        init_sigma = 1.   #  sigma
        return init_matrix, init_S, init_sigma


def acma(f, xbest, xsigma, xC, yworst, ysigma, yC, yh, xb, yb, neval, maxeval, acmapath):
        
    """ adversarial CMA-ES for local search after WRA 
    Parameters
    ----------
    f : callable f (a solution candidate, a scenario variable) -> float
        objective function
    xbest : np.ndarray (1D)
        solution candidate considered as the best in iteration t
    xsigma : float
        step size in the outer CMA-ES at iteration t
    xC : np.ndarray (2D)
        covariance matrix in the outer CMA-ES at iteration t
    yworst : np.ndarray (1D)
        scenario vector considered as the worst in iteration t
    ysigma : float
        step size in the configuration which generated yworst
    yC : np.ndarray (2D)
        covariance matrix in the configuration which generated yworst
    yh : np.ndarray (2D)
        set of the scenario vectors at the end of WRA
    xb : np.ndarray (2D)
        lower and upper bound of the box constraint for x
    yb : np.ndarray (2D)
        lower and upper bound of the box constraint for y
    neval : int
        number of f-calls
    maxeval : int
        maximum number of f-calls

    Return
    advcma.neval : float, 
        number of f-calls
    advx : np.ndarray (1D)
        optimum x resulted from adversarial CMA-ES search
    advy : np.ndarray (1D)
        worst y resulted from adversarial CMA-ES search
    """

    def xsampler():
        return xlbound + (xubound - xlbound) * np.random.rand(len(xlbound))
    def ysampler():
        return ylbound + (yubound - ylbound) * np.random.rand(len(ylbound))

    # Initialization
    logchol = False
    
    xlbound = xb[0,:]
    xubound = xb[1,:]

    ylbound = yb[0,:]
    yubound = yb[1,:]

    bounded = True
    etamin = 1e-4
    maxeval = maxeval
    tolgap = 1e-6
    tolxsigma=1e-8
    tolysigma=1e-8
    
    xhist = []
    if bounded:
        xl, xu, yl, yu = xlbound, xubound, ylbound, yubound
        def ff(x, y):
            xmirror = mirror(x, xlbound, xubound)
            ymirror = mirror(y, ylbound, yubound)
            return f(xmirror, ymirror)
    else:
        xl, xu, yl, yu = None, None, None, None
        ff = f

    advcma = AdversarialCmaes(ff, len(xlbound), len(ylbound), tolgap=tolgap, \
                              xsampler=xsampler, ysampler=ysampler, xlbound=xl, \
                              xubound=xu, ylbound=yl, yubound=yu, xsigma_min=tolxsigma, \
                              ysigma_min=tolysigma, logpath=acmapath, logchol=logchol)
    for i in range(len(yh)):
        advcma.register(yh[i, :])
    advcma.neval = neval

    x0 = xbest
    xsigma0 = xsigma
    xchol0 = xC
    y0 = yworst
    ysigma0 = ysigma
    ychol0 = yC
    
    advx, advy = advcma.optimize(etamin, x0, y0, xsigma0, ysigma0, xchol0, ychol0, maxeval)

    return advcma.neval, advx, advy

def optimize(f, Ftolgap, gapitval, Vxmin, Cxmax, boundx, boundy, lambdax, init_x, init_D_x, maxitr, maxeval, MAXBI, xb, yc, ymember, yb, init_y, init_ymean, init_D_y, tau_thr, cmax, Vymin, Cymax, Tmin, p, pp, pn, p_thr, local_adv, acmapath = 'advcma.txt', logpath='test.txt'):

    """Optimize min-max problem min_x max_y f(x,y) by WRA-CMA 
    
    Parameters
    ----------
    f : callable
        min-max objective function f(x, y)
    Ftolgap : float
        termination parameter about minimum objective function value change
    gapitval : float
        iteration interval to observe objective function value change
    Vxmin : float
        termination parameter about maximum standard deviasion for the outer CMA-ES
    Cxmax : float
        termination parameter about condition number of covariance matrix for the outer CMA-ES
    boundx and boundy : bool
        the domain for design variable or scenario variable is bounded (== initialization interval) if it is True
    lambdax : int, (default : 4 + int(3 * log(m)))
        population size for the outer CMA-ES
    init_x : 1D array of size n
        initial mean vector for the outer CMA-ES
    init_D_x : 1D array, positive
        initial coordinate-wise std for the outer CMA-ES
    maxitr : integer
        maximum number of iterations
    maxeval : int
        maximum number of f-calls
    xb and yb : np.ndarray (2D)
        lower and upper bound of the box constraint for x and y, respectively 
    yc : int 
        the factor for the number of configurations
    ymember : int, (= yc × lambdax )
        the number of configuration
    init_y : np.ndarray (2D)
        initial scenario vectors
    init_y_mean : np.ndarray (2D)
        initial mean vectors for the inner CMA-ES 
    init_D_y : np.ndarray (2D)
        initial coordinate-wise std for the inner CMA-ES
    tau_thr : float
        threshold of Kendall's tau for early stopping strategy
    cmax : int
        parameter for interrupting CMA-ES. If worse scenario is found cmax times, CMA-ES is interrupted.
    Vymin : float
        termination parameter about maximum standard deviation of the inner CMA-ES
    Cymax : float
        termination parameter about condition number of the inner CMA-ES
    Tmin : int
        minimum iteration for the inner CMA-ES
    p :  np.ndarray (1D)
        vector to decide which configuration is refreshed
    pp : float, , 0 < pp <= 1
        parameter for positive update of p
    pn : float, 0 < pn <= 1
        parameter for negative update of p
    p_thr : float, 0 < p_thr <= 1
        threshold parameter for refresh configurations
    local_adv : bool
        local search is executed by adversarial CMA-ES if it is True

    Returns
    -------
    fbest : float 
        f(xbest, yworst) value
    xbest : np.ndarray (1D)
        best solution candidate 
    yworst : np.ndarray (1D)
        worst scenario vector
    nominalx : np.ndarray (2D)
        list of candidate x
    nominaly : np.ndarray (2D)
        list of candidate y 
    """

    m = len(init_x)
    n = len(init_y[0, :])
    # Initialization for CMA-ES
    cma = MyDdCma(init_x, init_D_x, lambdax, flg_variance_update=False, beta_eig=10*m**2)

    # Initialization for WRA
    init_matrix = np.zeros((ymember, 5, n, n))
    init_S = np.zeros((ymember, n))
    init_sigma = np.zeros(ymember) 
    for i in range(ymember):
        init_matrix[i, 0,:,:] = np.zeros((n, n)) 
        init_matrix[i, 1,:,:] = np.eye(n) 
        init_matrix[i, 2,:,:] = np.eye(n) 
        init_matrix[i, 3,:,:] = np.eye(n) 
        init_matrix[i, 4,:,:] = np.eye(n)
        init_S[i, :] = np.ones(n) 
        init_sigma[i] = 1.   

    wrar = WRA(f, init_x, lambdax, yb, boundy, tau_thr, cmax, Vymin, Cymax, Tmin, init_y, init_ymean, init_D_y, init_matrix, init_S, init_sigma, p, pp, pn, p_thr)
    neval = 0
    conv = 0
    
    ## History
    res = np.zeros(2+3*m+lambdax)
    Fz_hist = []
    
    ## Preparation for termination of minimum objective function value change
    tg = 0
    
    ## Preparation of result list
    nominalx = []
    nominaly = []
    
    # Main Loop
    for t in range(maxitr):
        tg += 1
        # CMA Sampling
        arx, ary, arz = cma.sample()
        arc = arx.copy()
        if boundx==True:
            for i in range(cma.lam):
                arc [i, :] = mirror (arx[i, :], xb[0, :], xb[1, :]) 

        #   wradd_cma
        idr = wrar(arc)
        neval += wrar.fcalls
        
        # X Update
        cma.update(idr, arx, ary, arz)
        cp_D_x = cma.D.copy()
        for j in range(m):
            cma.D[j] = min(cp_D_x[j], (xb[1, j] - xb[0, j])/4/cma.sigma)

        cmean = cma.xmean
        if boundx==True:
            cmean = mirror (cma.xmean, xb[0, :], xb[1, :])

        # Output
        idx = 0
        res[idx] = neval; idx += 1
        res[idx] = cma.sigma; idx += 1
        res[idx:idx+m] = cmean; idx += m
        res[idx:idx+m] = cma.coordinate_std; idx += m
        res[idx:idx+m] = cma.S; idx += m
        res[idx:idx+lambdax] = wrar.Fnew; idx+=lambdax
        
        with open(logpath, 'ba') as flog:
            np.savetxt(flog, res, newline=' ')
            flog.write(b"\n")

        # Termination
        Fz_hist.append(min(wrar.Fnew))
        if tg-1 >= gapitval:
            Fzgap = max(Fz_hist[t-gapitval:])-min(Fz_hist[t-gapitval:])


            if Fzgap <= Ftolgap:
                conv = 2
                
        if np.max(cma.S) / np.min(cma.S) > Cxmax:
            conv = 3
        if np.max(cma.coordinate_std) < Vxmin:
            conv = 4

        ###       restart 
        if conv > 1:            
            ##  For adversarial part
            if local_adv==True:
                bid = np.argmin(wrar.Fnew)
                xbest = np.array(arc[bid], copy=True) 
                xsigma = cma.sigma
                xC = np.array(cma.transform(np.eye(n)) / cma.sigma, copy=True)

                wid = wrar.kworst[bid]
                yworst = np.array(wrar.yy[wid, :], copy=True)
                ysigma = wrar.init_sigma[wid]
                ty = np.dot(np.eye(n), wrar.init_matrix[wid, 3, :,:])
                yC = np.array(ty * (wrar.D_y[wid,:]), copy=True)

                neval, advx, advy = acma(f, xbest, xsigma, xC, yworst, ysigma, yC, wrar.yy, xb, yb, neval, maxeval, acmapath)

                # Output
                idx = 0
                res[idx] = neval; idx += 1
                res[idx] = cma.sigma; idx += 1
                res[idx:idx+m] = advx; idx += m
                res[idx:idx+m] = cma.coordinate_std; idx += m
                res[idx:idx+m] = cma.S; idx += m
                res[idx:idx+lambdax] = wrar.Fnew; idx+=lambdax

                with open(logpath, 'ba') as flog:
                    np.savetxt(flog, res, newline=' ')
                    flog.write(b"\n")

                for i in range(wrar.lambdax):
                    nominalx.append(arc[i])
                nominalx.append(advx)
                nominaly.append(advy)

            ##############################
            if MAXBI == 0:
                neval = maxeval
                break
            print ('restart  '+str(neval)+'  '+str(conv))
 
            init_x = np.random.rand(m)*(xb[1, :]-xb[0, :])+xb[0, :]
            init_D_x = (xb[1, :]-xb[0, :])/4
            cma = MyDdCma(init_x, init_D_x, lambdax, flg_variance_update=False, beta_eig=10*m**2)
            tg = 0
            ymember = int(lambdax*yc)
            init_ymean = np.zeros((ymember, wrar.n))
            init_D_y = np.zeros((ymember, wrar.n))
            init_y = np.zeros((ymember, wrar.n))
            for i in range(ymember):
                init_ymean[i,:] = np.random.rand(n)*(yb[1, :] - yb[0, :]) + yb[0, :]
                init_D_y[i,:] = (yb[1, :]-yb[0, :])/4
                yz = np.random.randn(n)
                for j in range(n):
                    init_y[i, j] = min(max(yz[j] * init_D_y[i,j] + init_ymean[i,j], yb[0, j]), yb[1, j])
    
            init_matrix = np.zeros((ymember, 5, wrar.n, wrar.n))
            init_S = np.zeros((ymember, wrar.n))
            init_sigma = np.zeros(ymember) 
            for i in range(ymember):
                init_matrix[i, 0,:,:] = np.zeros((wrar.n, wrar.n)) 
                init_matrix[i, 1,:,:] = np.eye(wrar.n) 
                init_matrix[i, 2,:,:] = np.eye(wrar.n) 
                init_matrix[i, 3,:,:] = np.eye(wrar.n) 
                init_matrix[i, 4,:,:] = np.eye(wrar.n)
                init_S[i, :] = np.ones(wrar.n) 
                init_sigma[i] = 1.   
            wrar = WRA(f, init_x, lambdax, yb, boundy, tau_thr, cmax, Vymin, Cymax, Tmin, init_y, init_ymean, init_D_y, init_matrix, init_S, init_sigma, p, pp, pn, p_thr)
            conv = 0
        
        if neval >= maxeval:
            break
    # check best solution candidate
    for i in range(cma.lam):
        nominalx.append(arc[i])
        
    for i in range(wrar.ymember):
        nominaly.append(wrar.yy[i])

    fxsize=len(nominalx)
    fysize = len(nominaly)
    flist=np.zeros(fxsize)
    wlist=np.zeros(fxsize, dtype='int')
    for i in range(fxsize):
        f_arr =  np.array([f(nominalx[i], nominaly[j]) for j in range(fysize)])
        flist[i] = np.max(f_arr)
        wlist[i] = np.argmax(f_arr)
    fbest = np.min(flist)
    bid = np.argmin(flist)
    xbest = nominalx[bid]
    yworst = nominaly[wlist[bid]]
    return fbest, xbest, yworst, nominalx, nominaly

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    logpre = 'CMA-WRA_ftest'
    logpath = logpre + '.txt'
    logpathx = logpre + '_x.csv'
    logpathy = logpre + '_y.csv'
    with open(logpath, 'w'): 
        pass
    with open(logpathx, 'w'): 
        pass
    with open(logpathy, 'w'): 
        pass

    acmapath = 'advcma.txt'
    with open(acmapath, 'w'): 
        pass

    ## Experiment
    n = 5
    m = 5

    xb = np.zeros((2, m))
    xb[0, :] = np.ones(m)*-3
    xb[1, :] = np.ones(m)*3
    
    yb = np.zeros((2, n))
    yb[0, :] = np.ones(n)*-3
    yb[1, :] = np.ones(n)*3
    
    boundx = True
    boundy = True

    ## termination criteria for outer CMA
    Vxmin = 1e-8
    Cxmax = 1e+7
    Ftolgap = 0
    gapitval = 10
    maxeval = int(1e+06) 
    maxitr = int(1e+5)

    ## termination criteria for inner CMA
    Vymin = 1e-4
    Cymax = 1e+7
    Tmin = 10

    ## initial parameters for outer CMA-ES
    init_x = np.random.rand(m)*(xb[1, :] - xb[0, :]) + xb[0, :]
    init_D_x = (xb[1, :]-xb[0, :])/4
    lambdax = (4 + int(3 * math.log(m)))

    ## initial parameters for inner CMA-ES
    yc = 3
    ymember = int(yc*lambdax)
    init_ymean = np.zeros((ymember, n))
    init_D_y = np.zeros((ymember, n))
    init_y = np.zeros((ymember, n))
    for i in range(ymember):
        init_ymean[i,:] = np.random.rand(n)*(yb[1, :] - yb[0, :]) + yb[0, :]
        init_D_y[i,:] = (yb[1, :]-yb[0, :])/4
        yz = np.random.randn(n)
        for j in range(n):
            init_y[i, j] = min(max(yz[j] * init_D_y[i,j] + init_ymean[i,j], yb[0, j]), yb[1, j])


    ## local search by adversarial CMA-ES
    local_adv = False

    ## parameter for restart
    MAXBI = 1

    ## initial parameters for WRA
    tau_thr = 0.7
    gamma = 1/n
    cmax = int(n*gamma)
    p = np.ones(ymember) * 1
    pn = 0.05
    pp = 0.4
    p_thr = 0.1

    ## If Tfunc is True, test functions can be used
    Tfunc=True
    if Tfunc==True:
        fn = 3
        b=1
        setB = ff.setB(m, n, b)
        B= setB.setB()
        fobj =ff.minmaxf(B, yb[:, 0], xb[:, 0])
        if fn==1:
            f = fobj.flxly
        elif fn==2:
            f = fobj.fcnc
        elif fn==3:
            f = fobj.fncc
        elif fn==4:
            f = fobj.fmsp
        elif fn==5:
            f = fobj.fqcc
        elif fn==6:
            f = fobj.fnsscc
        elif fn==7:
            f = fobj.fnscc
        elif fn==8:
            f = fobj.fnscnsc
        elif fn==9:
            f = fobj.fcvncc    
        elif fn==10:
            f = fobj.fc4
        elif fn==11:
            f = fobj.fellqcc
        
    print ('Start')
    fxy, xbest, yworst, x_arr, y_arr = optimize(\
    f, Ftolgap, gapitval, Vxmin, Cxmax, boundx, boundy, lambdax, init_x, init_D_x, maxitr, maxeval, MAXBI, xb, yc, ymember, yb, init_y, init_ymean, init_D_y, tau_thr, cmax, Vymin, Cymax, Tmin, p, pp, pn, p_thr, local_adv, acmapath = acmapath, logpath = logpath
    )

    print ('f(x,y)=', fxy)
    print ('xbest=', xbest)

    # Evaluation
    print("Worst-case search has started.")
    fy, y, f_arr, y_arr = worstsearch(f, xbest, yb[0, :], yb[1, :], boundy, maxeval=100*n, tolsigma=1e-8, n_restart=100)
    print("Worst-case search has finished.")
    print("worst y:", y)
    print("f(x,y):", fy)

    np.savetxt(logpathx, x_arr)
    np.savetxt(logpathy, y_arr)

    dat = np.loadtxt(logpath)
    hatF = np.array([np.min(dat[t, 2+3*m:2+3*m+lambdax]) for t in range(len(dat[:,0]))])

    ## If test functions are used, worst y can be obtained.
    if Tfunc==True:
        callworst = ff.call_worst(fn, yb[:, 0], n, B)
        cmean=np.array(dat[:,2:2+m])
        for i in range(len(dat[:,0])):
            worstS = np.array(callworst.yworst(cmean[i,:]))
            hatF[i] = f(cmean[i,:], worstS)


    plt.rcParams['font.family'] ='sans-serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 5.0
    plt.rcParams['ytick.major.width'] = 5.0
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.linewidth'] = 1.0
    plt.figure(figsize=(10,7))

    ## Plot
    idp=0
    ax1 = plt.subplot(221)
    ax1.plot(dat[:, 0], hatF[:], label='$min_{i=1,\dots,\lambda_x} F^{rd}(x^t_i)$')
    ax1.plot(dat[:, 0], dat[:, 1], label=r'$\sigma_x$')
    plt.xlabel("#f-calls")
    plt.yscale('log')
    plt.legend()
    plt.grid()

    ax2 = plt.subplot(222)
    ax2.plot(dat[:, 0], abs(dat[:, 2:2+m]))
    plt.ylabel("$m^t$")
    plt.xlabel("#f-calls")

    plt.yscale('log')
    plt.grid()
    idp = 2+m

    ax3 = plt.subplot(223)
    ax3.plot(dat[:, 0], abs(dat[:, idp:idp+m]))
    plt.xlabel("#f-calls")
    plt.ylabel("Standard deviation")
    plt.yscale('log')
    plt.grid()
    idp +=m

    ax4 = plt.subplot(224)
    ax4.plot(dat[:, 0], dat[:, idp:idp+m])
    plt.xlabel("#f-calls")
    plt.ylabel("Eigen value")
    plt.yscale('log')
    plt.grid()

    plt.tight_layout()
    plt.savefig('WRA_CMA_ftest.pdf')

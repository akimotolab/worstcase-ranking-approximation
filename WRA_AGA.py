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
    """ Worst-case Raking Approximation (WRA) : approximation of the worst-case ranking 
    for the given solution candidates by approximately solving internal maximization problem max_{y} f(x, y) with using CMA-ES
    """
    def __init__(self, f, init_x, lambdax, yb, boundy, tau_thr, cmax, Umin, beta, init_y, p, pp, pn, p_thr):
        """ Initialization of WRA
        Parameters
        ----------
        f : callable f (a solution candidate, a scenario variable) -> float
            objective function
        init_x : np.ndarray (1D)
            initial candidate vector of design variable
        lambdax : int
            number of solution candidate 
        yb : np.ndarray (2D)
            lower and upper bound of the box constraint
        boundy : bool
            the domain for scenario variable is bounded (== initialization interval) if it is True
        tau_thr : float
            threshold of Kendall's tau for early stopping strategy
        cmax : int
            parameter for interrupting CMA-ES
            If worse scenario is found cmax times, CMA-ES is interrupted.
        Umin : float
            termination parameter for minimum update of y
        beta : float, 0<beta<=1
            learnign rate for scenario update
        init_y : np.ndarray (2D)
            initial candidate vector of scenario variables
        p :  np.ndarray (1D)
            vector to decide which configuration is refreshed
        pp : float, , 0 < pp <= 1
            parameter for positive update of p
        pn : float, 0 < pn <= 1
            parameter for negative update of p
        p_thr : float, 0 < p_thr <= 1
            threshold parameter for refresh configurations
        """

        self.f = f
        self.ymember = init_y.shape[0]
        self.lambdax = lambdax
        self.id = 0
        self.f_instances = []
        self.solution = np.zeros((lambdax, len(init_x)))
        for i in range(lambdax):
            self.f_instances.append(partial(f, init_x[:]))
            self.solution[i, :] = init_x[:]
        self.yb = yb
        self.yy = init_y
        self.init_y = init_y
        self.alpha = np.ones(self.ymember)*(self.yb[1,0] - self.yb[0,0])/2
        self.beta = beta
        self.n = init_y.shape[1]
        self.tau_thr = tau_thr
        self.cmax = cmax
        self.Umin = Umin
        self.fcalls = 0
        self.boundy = boundy
        self.tau = 0
        self.wrac = []
        self.wraneval = []
        self.Fnew = []
        self.kworst = []

        ## probability control
        self.p = p
        self.pp = pp
        self.pn = pn
        self.p_thr = p_thr
        
    def __call__(self, arx):
        return self.wra_aga(arx)

    def max_f(self, y):
        fv = -1*self.f(self.solution[self.id, :], y)
        return fv
    
    def wra_aga(self, x_t):
        
        """approximating the worst-case ranking of solution candidate 
        Parameters
        ----------
        x^t : np.ndarray (2D)
            solution candidates in iteration t 

        Return
        idr : np.ndarray (1D)
            approximated ranking of solution candidates x^t
        """
        
        self.fcalls = 0
        ### Initialization part
        self.kworst = np.zeros(self.lambdax, dtype='int')
        Fold = np.zeros(self.lambdax)
        y_tilde = np.zeros((self.lambdax, self.n))
        alpha_tilde = np.ones(self.lambdax)
        for i in range(self.lambdax):
            fx_arr = np.array([self.f(x_t[i,:], y) for y in self.yy])
            self.kworst[i] = np.argmax(fx_arr)
            Sworst = np.unique(self.kworst)
            Fold[i] = np.max(fx_arr)

            self.solution[i, :] = x_t[i, :]
            y_tilde[i, :] = self.yy[self.kworst[i], :]
            alpha_tilde[i] = self.alpha[self.kworst[i]]
            
        self.Fnew = Fold.copy()
        self.fcalls += self.ymember * self.lambdax

        ###    Worst-case Ranking Approximation
        h = np.ones(self.lambdax, dtype=bool)
        updaten = np.zeros(self.lambdax)
        tt = np.zeros(self.lambdax)
        self.wrac = np.zeros(self.lambdax)
        self.wraneval = np.zeros(self.lambdax)
        self.tau = -1.0
        bounds = [(self.yb[0, i], self.yb[1, i]) for i in range(self.n)]
        while self.tau <= self.tau_thr:
            for i in range(self.lambdax):
                self.id = i
                if h[i]==True:
                    Fdash = Fold[i]
                    updaten[i] += 1
                    c = 0
                    nominaly = y_tilde[i, :]
                    while c<self.cmax:
                        alpha_tilde[i], nominaly, nominalf, Lfcalls, switch = self.Lsearch(nominaly, bounds, alpha_tilde[i], -Fdash)
                        self.fcalls +=Lfcalls
                        self.wraneval[i] += Lfcalls
                        if nominalf <= Fdash:
                            h[i] = False
                            break
                        else: 
                            y_tilde[i, :] = nominaly
                            Fdash = nominalf
                            c += 1
                            self.wrac[i] += 1

                        tt[i] +=1
                    self.Fnew[i] = Fdash
            self.tau, p_value = stats.kendalltau(self.Fnew, Fold)
            Fold[:] = self.Fnew[:]

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
            self.alpha[Sworst[l]] = alpha_tilde[ridy]
        
        for k in pnmember:
            self.p[k] -= self.pn
            if self.p[k] <= self.p_thr:
                self.reset_y(k)
        
        return idr

    def Lsearch(self, nominaly, bounds, alpha, fold):
        switch = 0
        Lfcalls = 0
        res = minimize(self.max_f, nominaly, args=(), method='SLSQP', jac=False, bounds=bounds, options={'maxiter': 1})
        fdirect = res.jac
        Lfcalls += res.nfev
        nominalyy = nominaly - alpha * fdirect
        bcount = 0
        if self.boundy==True:
            for k in range(len(nominaly)):
                if nominalyy[k] >= self.yb[1, k] or nominalyy[k] <= self.yb[0, k]:
                    bcount += 1
        newy = nominalyy.copy()
        if self.boundy==True:
            newy[:] = [min(self.yb[1, i], max(nominalyy[i], self.yb[0, i])) for i in range(self.n)]
        fnew = self.max_f(newy)
        Lfcalls +=1
        if fnew < fold and bcount != len(nominaly):
            alpha = 1/self.beta*alpha
        elif fnew > fold:
            while fnew > fold:
                alpha = self.beta*alpha
                nominalyy = nominaly - alpha * fdirect
                if self.boundy==True:
                    newy[:] = [min(self.yb[1, i], max(nominalyy[i], self.yb[0, i])) for i in range(self.n)]
                fnew = self.max_f(newy)
                Lfcalls +=1
                if max(abs(alpha * fdirect)) <= self.Umin:
                    switch = 1
                    break
        return alpha, newy, -fnew, Lfcalls, switch
        
    def reset_y(self, j):
        self.yy[j, :] = np.random.rand(n)*(yb[1, :] - yb[0, :]) + yb[0, :]
        self.alpha[j] = (self.yb[1,0] - self.yb[0,0])/2
        self.p[j] = 1.0

def acma(f, xbest, xsigma, xC, yworst, yh, xb, yb, neval, maxeval, acmapath):

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

    x_init, sig, chol = generate_cmaes_parameter(xlbound, xubound)
    y_init, sig, chol = generate_cmaes_parameter(ylbound, yubound)
    x_int = xbest
    
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

    y0, ysigma0, ychol0 = generate_cmaes_parameter(ylbound, yubound)
    y0 = yworst
    ysigma0 = ysigma0*0.1
    ychol0 = ychol0
    
    advx, advy = advcma.optimize(etamin, x0, y0, xsigma0, ysigma0, xchol0, ychol0, maxeval)

    return advcma.neval, advx, advy


def optimize(f, Ftolgap, gapitval, Vxmin, Cxmax, boundx, boundy, lambdax, init_x, init_D_x, maxitr, maxeval, MAXBI, xb, yc, ymember, yb, init_y, tau_thr, cmax, Umin, beta, p, pp, pn, p_thr, local_adv, acmapath = 'advcma.txt', logpath='test.txt'):

    """Optimize min-max problem min_x max_y f(x,y) by CMA-ES with WRA 
    
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
    tau_thr : float
        threshold of Kendall's tau for early stopping strategy
    cmax : int
        parameter for interrupting CMA-ES. If worse scenario is found cmax times, CMA-ES is interrupted.
    Umin : float
        termination parameter for maximum standard deviation of CMA-ES
    beta : float, 0<beta<=1
        learnign rate for scenario update
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
    n = init_y.shape[1]
    # Initialization for CMA-ES
    cma = DdCma(init_x, init_D_x, lambdax, flg_variance_update=False, beta_eig=10*m**2)

    wrar = WRA(f, init_x, lambdax, yb, boundy, tau_thr, cmax, Umin, beta, init_y, p, pp, pn, p_thr)
    neval = 0
    conv = 0
    
    ## History
    res = np.zeros(2+3*m+lambdax)
    Fz_hist = []
    bx_hist = []

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
        bxid = np.argmin(wrar.Fnew)
        bx = arc[bxid]
        bx_hist.append(bx)
        if tg-1 >= gapitval:
            Fzgap = max(Fz_hist[t-gapitval:])-min(Fz_hist[t-gapitval:])
            if Fzgap <= Ftolgap:
                conv = 2
                
        if np.max(cma.S) / np.min(cma.S) > Cxmax:
            conv = 3
        if np.max(cma.coordinate_std) < Vxmin:
            conv = 4
        if neval >= maxeval:
            break

        ###       restart 
        if conv > 1:

            ##  For adversarial part
            if local_adv == True:
                bid = np.argmin(wrar.Fnew)
                xbest = np.array(arc[bid], copy=True) 
                xsigma = cma.sigma
                xC = np.array(cma.transform(np.eye(m)) / cma.sigma, copy=True)

                wid = wrar.kworst[bid]
                yworst = np.array(wrar.yy[wid, :], copy=True)
                
                neval, advx, advy = acma(f, xbest, xsigma, xC, yworst, wrar.yy, xb, yb, neval, maxeval, acmapath)

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

                for i in range(wrar.lambdax):
                    nominalx.append(arc[i])
                nominalx.append(advx)
                nominaly.append(advy)

            ##############################
            if MAXBI == 0:
                neval = maxeval
                break
            print ('restart  '+str(neval))
                
            init_x = np.random.rand(m)*(xb[1, :]-xb[0, :])+xb[0, :]
            init_D_x = (xb[1, :]-xb[0, :])/4
            cma = DdCma(init_x, init_D_x, lambdax, flg_variance_update=False, beta_eig=10*m**2)
            tg = 0
            
            ## initial parameters for WRA
            ymember = int(lambdax*yc)
            init_y = np.zeros((ymember, wrar.n))
            p = np.ones(ymember) * 1 
            for i in range(ymember):
                init_y[i, :] = np.random.rand(n)*(wrar.yb[1, :] - wrar.yb[0, :]) + wrar.yb[0, :]

            wrar = WRA(f, init_x, lambdax, yb, boundy, tau_thr, cmax, Umin, beta, init_y, p, pp, pn, p_thr)
            conv = 0

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

## test function
a = 1.
b = 100.
c = 1.
def fcon_cav(x, y, a=1, b=10, c=1):
    fxy = a/2*np.dot(x, x) + b * np.dot(x, y) - c/2*np.dot(y, y)
    return fxy

def fbinary(x, y, a=1, b=1, c=1):
    fxy = np.dot(x, y)
    return fxy

def f(x, y):
    return fbinary(x, y, a, b, c)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    logpre = 'wra_aga'
    logpath = logpre + '.txt'
    logpathx = logpre + '_x.txt'
    logpathy = logpre + '_y.txt'
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
    m = 10
    n = 10

    xb = np.zeros((2, m))
    xb[0, :] = np.ones(m)*-3
    xb[1, :] = np.ones(m)*3
    
    yb = np.zeros((2, n))
    yb[0, :] = np.ones(n)*-3
    yb[1, :] = np.ones(n)*3
    
    boundx = True
    boundy = True

    tau_thr = 0.7
    gamma = 1/n
    cmax = int(n*gamma)

    ## termination criteria
    Vxmin = 1e-10
    Cxmax = 1e+7
    Ftolgap = 1e-5
    gapitval = 100
    
    ## initial parameters for CMA-ES
    init_x = np.random.rand(m)*(xb[1, :] - xb[0, :]) + xb[0, :]
    init_D_x = (xb[1, :]-xb[0, :])/4
    lambdax = (4 + int(3 * math.log(m)))
    maxeval = int(1e+06) 
    maxitr = int(1e+5)

    ## initial parameters for WRA
    local_adv = False
    yc = 3
    beta = 0.5
    ymember = int(lambdax*yc)
    init_y = np.zeros((ymember, n))
    MAXBI = 1
    p = np.ones(ymember) * 1 
    pn = 0.05
    pp = 0.4
    p_thr = 0.1
    Umin = 1e-5

    for i in range(ymember):
        init_y[i, :] = np.random.rand(n)*(yb[1, :] - yb[0, :]) + yb[0, :]

    print ('Start')

    fxy, xbest, yworst, x_arr, y_arr = optimize(\
f, Ftolgap, gapitval, Vxmin, Cxmax, boundx, boundy, lambdax, init_x, init_D_x, maxitr, maxeval, MAXBI, xb, yc, ymember, yb, init_y, tau_thr, cmax, Umin, beta, p, pp, pn, p_thr, local_adv, acmapath = acmapath, logpath = logpath
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

    plt.rcParams['font.family'] ='sans-serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 5.0
    plt.rcParams['ytick.major.width'] = 5.0
    plt.rcParams['font.size'] = 12
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
    plt.savefig('WRA_AGA_ftest.pdf')

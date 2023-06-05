import numpy as np
import math

class minmaxf:

    def __init__(self, B, yb, xb):
        self.B=B
        self.yb = yb
        self.xb = xb
## 1
    def flxly(self, x, y):
        fxy = np.dot(np.dot(x, self.B), y)
        return fxy

## 2
    def fcnc(self, x, y):
        fxy = 1/2*np.dot(x, x) + np.dot(np.dot(x, self.B), y)
        return fxy

## 3
    def fncc(self, x, y):
        n1=np.ones(len(y))
        b=np.linalg.pinv(self.B)
        bdagger=np.zeros(len(x))
        for i in range(len(x)):
            bdagger[i]=np.linalg.norm(b[:,i],ord=1)
        alpha=-7/30*min(abs(self.xb[0]), abs(self.xb[1]))/max(bdagger)
        gamma=1
        t=np.dot(x, self.B)  - (alpha - gamma * self.yb[1])*n1
        fxy = 1/2 * np.dot(t, t) + gamma* np.dot(np.dot(x, self.B), y)
        return fxy

## 4
    def fmsp(self, x, y):
        fxy = 1/2*np.dot(x, x) + np.dot(np.dot(x, self.B), y) + 1/2*np.dot(y, y)
        return fxy

## 5
    def fqcc(self, x, y):
        fxy = 1/2*np.dot(x, x) + np.dot(np.dot(x, self.B), y) - 1/2*np.dot(y, y)
        return fxy
    
## 6
    def fnsscc(self, x, y):
        xl1norm = sum(abs(x))
        yl1norm = sum(abs(y))
        fxy = 1/2*np.dot(x, x)+xl1norm + np.dot(np.dot(x, self.B), y) -yl1norm - 1/2*np.dot(y, y) 
        return fxy
    
## 7
    def fnscc(self, x, y):
        x2norm = np.sqrt(np.dot(x,x))
        y2norm = np.sqrt(np.dot(y,y))
        fxy = 1/4*x2norm**4 + np.dot(np.dot(x, self.B), y) - 1/4*y2norm**4
        return fxy

## 8
    def fnscnsc(self, x, y):
        xl1norm = sum(abs(x))
        yl1norm = sum(abs(y))
        fxy = xl1norm + np.dot(np.dot(x, self.B), y) - yl1norm
        return fxy

##  9
    def fcvncc(self,x,y):
        fxy = 0
        xi=np.dot(x, self.B)
        nstar = min(len(xi), 3)
        for i in range(len(xi)):
            siny = math.sin(np.pi*y[i]/self.yb[1])
            if i<nstar:
                fxy = fxy + (xi[i] + np.exp(np.sign(y[i]))*siny)**2 
            else:
                fxy = fxy + xi[i]**2 - y[i]**2
        return fxy

## 10
    def fc4(self,x,y):
        xi = np.dot(self.B, x)
        fxy = np.dot(xi,xi)-2*np.dot(y-xi,y-xi)
        return fxy

## 11
    def fellqcc(self, x, y):
        fxy = 0
        xi=np.dot(x, self.B)
        n=len(xi)
        for i in range(n):
            fxy = fxy + 1/2*x[i]**2 + (xi[i]*y[i])*10**(-3*(i+1)/n) - (1/2*y[i]**2)*10**(-6*(i+1)/n)
        return fxy

class setB:
    def __init__(self, n, m, b):
        self.n =n
        self.m =m
        self.b=b
    def setB(self):
        B = np.zeros((self.n, self.m))
        B_diag = np.diag(B)
        B_diag.flags.writeable = True
        B_diag[:] = np.ones(min(self.m, self.n))*self.b
        if self.m != self.n:
            for i in range(abs(self.m-self.n)):
                if self.n>=self.m:
                    kk=-(i+1)
                else:
                    kk=i+1
                B_diag = np.diag(B, k=kk)
                B_diag.flags.writeable = True
                B_diag[:] = np.ones(min(self.m, self.n))*self.b
        return B
 
class call_worst:

    def __init__(self, fn, yb, n, B):
        self.n =n
        self.yb = yb
        self.fn = fn
        self.B=B
        
    def yworst(self, x):
        worstS=np.zeros(self.n)
        xi=np.dot(x, self.B)
        m = len(xi)
        ## if self.fn = 1, 2, 3, 4
        for i in range(m):
            worstS[i] = self.yb[1]*np.sign(xi[i])
            if self.fn==4 and xi[i]==0.0:
                worstS[i] = self.yb[1]

        if self.fn==5:
            for i in range(m):
                worstS[i] = self.yb[1]*np.sign(xi[i])
                if self.yb[0]<=xi[i] and xi[i]<=self.yb[1]:
                    worstS[i] = xi[i]
                
        if self.fn==6:
            for i in range(m):
                if -1<=x[i] and xi[i]<=1:
                    worstS[i] = 0
                if 1<abs(xi[i])<=(self.yb[1]+1):
                    worstS[i] = (xi[i]-np.sign(xi[i]))
                if (self.yb[1]+1)<abs(xi[i]):
                    worstS[i] = self.yb[1]*np.sign(xi[i])
            
        if self.fn==7:
            x2norm = np.sqrt(np.dot(xi, xi))
            for i in range(m):
                if abs(xi[i])/(x2norm**(2/3))<=self.yb[1]:
                    worstS[i] = xi[i]/(x2norm**(2/3))
                if abs(xi[i])/(x2norm**(2/3))>self.yb[1]:
                    worstS[i] = self.yb[1]*np.sign(xi[i])

        if self.fn==8:
            for i in range(m):
                if abs(xi[i]) <= 1:
                    worstS[i] = 0
                if abs(xi[i]) > 1:
                    worstS[i] = self.yb[1]*np.sign(xi[i])

        if self.fn==9:
            nstar = min(len(xi),3)
            for i in range(m):
                if i < nstar and xi[i] >= -np.sinh(1):
                    worstS[i] = self.yb[1] /2
                elif i < nstar and xi[i] <= -np.sinh(1):
                    worstS[i] = -self.yb[1] /2
                else :
                    worstS[i] = 0.0
            
        if self.fn==10:
            for i in range(m):
                worstS[i] = self.yb[1]*np.sign(xi[i])
                if self.yb[0]<=xi[i] and xi[i]<=self.yb[1]:
                    worstS[i] = xi[i]
                    
        if self.fn==11:
            for i in range(m):
                worstS[i] = self.yb[1]*np.sign(xi[i])
                xxi=xi[i]*10**(3*(i+1)/m)
                if self.yb[0]<=xxi and xxi<=self.yb[1]:
                    worstS[i] = xxi

        return worstS



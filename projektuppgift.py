import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from random import *

class projektuppgift:

    def __init__(self, N, npl):     #Definierar samtliga indata
        self._n = N
        self._nupplag = N + 2
        self._npl = npl
        self._L = np.array([6.3, 6.3, 6.3])
        self._E = np.array([1, 1, 1])
        self._I = np.array([1, 1, 1])
        self._q = np.array([7.16, 10.11, 10.11])
        
    def mycheck(self):
        lista_placeholdernamn = [self._L, self._E, self._I, self._q]    #Kontrollerar så att alla indata har korrekt antal element i sig
        for element in lista_placeholdernamn:
            if len(list(element)) != self._n:
                raise ValueError('Inte alla givna fack innehar längd, elasticitetsmodul, yttröghetsmoment och last. Vänligen se till att samtliga arrays innehåller lämplig data')
        for x in range(0, 2):      #Kontrollerar så att alla längder, elasticitetsmoduler och yttröghetsmoment är positiva.
            for indata in lista_placeholdernamn[x]:   #Loopar genom första till tredje elementet i lista_placeholdernamn för att undersöka längd, elasicitetsmodul och yttröghetsmoment
                if indata <= 0:
                    raise ValueError('Givna längder, elasticitetsmoduler och yttröghetsmoment är ogiltiga. Vänligen ange korrekta sådana')
        print('Samtliga data är rimliga')
        
                
    
    def create_tridiagonal_matrix(self):
        list_f = []
        for n in range(self._n):        #Skapar ett f för vardera fack enligt formeln från del 1.
            f = (self._L[n] / (3 * self._E[n] * self._I[n]))
            list_f.append(f)
        array_f = np.array(list_f)
        
        list_g = []
        for n in range(self._n):        #Skapar ett g för vardera fack enligt formeln från del 1.
            g = ((self._q[n] * (self._L[n] ** 3)) / (24 * self._E[n] * self._I[n]))
            list_g.append(g)
        array_g = np.array(list_g)
        
        a = np.zeros(self._n - 1)
        b = np.zeros(self._n - 1)
        c = np.zeros(self._n - 1)
        d = np.zeros(self._n - 1)
        
        for fack in range(1, self._n):      #Nyttjar arrayerna för f respektive g för att skapa en lösbar matrisekvation på formen AX = d
            a[fack-1] = 0.5 * array_f[fack-1]                                   # \\
            b[fack-1] = array_f[fack-1] + array_f[fack]                         # a, b och c utgör systemmatrisen
            c[fack-1] = 0.5 * array_f[fack]                                     # \\
            
            d[fack-1] = array_g[fack-1] + array_g[fack]                         #d utgör högerledet
        
        A = diags([a, b, c], offsets=[-1, 0, 1], shape=(self._n - 1, self._n - 1)).toarray()    #Formatterar om a, b och c till en systemmatris
        print(f"Konditionstalet = {np.linalg.cond(A)}")
        return(A, d)
    
        
    def solve_tridiagonal_matrix(self, A, d):
        z = np.linalg.solve(A, d)       #Tar fram samtliga stödmoment över upplagen genom att lösa ekvationen AX = d 
        return z
        
    def create_stödmoment(self, z):     #Nyttjar stödmomenten för att ta fram Ma respektive Mb i vardera fack.
        Ma = np.zeros(self._n)
        Mb = np.zeros(self._n)
        
        Ma[1:] = z
        Mb[:-1] = z
        
        return(Ma, Mb)
    
    def calculate_stödreaktion(self, Ma, Mb):   #Nyttjar Ma och Mb för att ta fram Ra respektive Rb enligt formeln R = qL^2 +- (Ma - Mb) / L
        Ra = -(self._q * self._L / 2) - (Ma - Mb) / self._L
        Rb = -(self._q * self._L / 2) + (Ma - Mb) / self._L
        return(Ra, Rb)
    
    def böjmoment_längst_balken(self, Ma, Ra):
        
        xbeam = np.array([])
        mbeam = np.array([])
        Mextreme = np.zeros(self._n)

        for fack in range(self._n):
            xloc = np.linspace(0, self._L[fack], self._npl)     #xloc tar fram npl antal jämnt fördelade punkter mellan start till slut i facket.
            
            Mloc = (                            #Mloc tar fram momentet i en given punkt genom momentekvationen
                Ma[fack] + 
                Ra[fack] * xloc +
                self._q[fack] * xloc ** 2 / 2)
            
            Mextreme[fack] = self.myextreme(Mloc, self._npl)
            
            if fack > 0:                    #Ifall vi är efter första facket tar vi bort första punkten för att undvika dubletter från tidigare facket
                xloc = xloc[1:]
                Mloc = Mloc[1:]
                   
            xloc += sum(self._L[:fack])     #Lägger till alla längder fram tills nuvarande fack till xloc så att rätt längder appendas till xbeam
            
            xbeam = np.concatenate((xbeam, xloc))
            mbeam = np.concatenate((mbeam, Mloc))
            
        return(xbeam, mbeam, Mextreme)
    
    
    def myextreme(self, Mloc, npl):
        max_moment = 0
        for point in range(1, npl - 1):       #Punkt 0 behöver inte tas hänsyn till då momentet i balkens början är 0.
            if abs(Mloc[point]) > abs(Mloc[point - 1]) and abs(Mloc[point]) > abs(Mloc[point + 1]):
                max_moment = Mloc[point]
        return max_moment
        
    def tvärkraft_längst_balken(self, Ra):
        
        vbeam = np.array([])

        for fack in range(self._n):
            xloc = np.linspace(0, self._L[fack], self._npl)
            
            Vloc = (
                - Ra[fack]
                - self._q[fack] * xloc)
            
            if fack > 0:
                Vloc = Vloc[1:]
                   
            xloc += sum(self._L[:fack])
            
            vbeam = np.concatenate((vbeam, Vloc))
            
        return vbeam

    def myplot(self, xbeam, mbeam, vbeam):
        #Tvärkraftsplot
        plt.plot(xbeam, vbeam)
        plt.xlabel("Position [m]")
        plt.ylabel("Sheer force [kN]")
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.show()
        
        #Moment plot
        plt.plot(xbeam, mbeam)
        plt.xlabel("Position [m]")
        plt.ylabel("Moment force [kNm]")
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.show()
        
    def mytable(self, Ma, Mb, Ra, Rb, Mextreme, xbeam, mbeam):
        
        #Tabellera upplagskrafterna
        upplagskrafter = np.concatenate(([Ra[0]], np.array(Rb[:-1]) + np.array(Ra[1:]), [Rb[-1]]))
        print("Upplagskrafter")
        print("Upplag:      R:")
        for upplag in range(self._n):
            print(f"{upplag}           {upplagskrafter[upplag]:.3f}")
            
        #Stödreaktioner vid upplagen
        print("Stödreaktioner")
        print("Fack:        Ra:         Rb:")
        for fack in range(self._n):
            print(f"{fack}          {Ra[fack]:.3f}          {Rb[fack]:.3f}")
            
        #Stödmomenten vid upplagen
        print("Stödmomenten")
        print("Fack:        Ma:         Mb:")
        for fack in range(self._n):
            print(f"{fack}          {Ma[fack]:.3f}          {Mb[fack]:.3f}")
            
        #Maximalt fältmoment i fack
        print("Maximalt fältmoment")
        print("Fack:        Max Fältmoment:")
        for fack in range(self._n):
            print(f"{fack}               {Mextreme[fack]:.3f}")
        
        #Max och Min böjmoment
        print("Max respektive Min böjmoment")
        print("Max böjmoment:           Position:")
        Max_moment = max(mbeam)
        Min_moment = min(mbeam)
        mlist = list(mbeam)
        index = (mlist.index(Max_moment), mlist.index(Min_moment))
        print(f"{Max_moment}            {xbeam[index[0]]:.3f}")
        print("Min böjmoment:           Position:")
        print(f"{Min_moment}            {xbeam[index[1]]:.3f}")
        
             
ja = projektuppgift(3, 10000)
ja.mycheck()
A, d = ja.create_tridiagonal_matrix()
z = ja.solve_tridiagonal_matrix(A, d)
Ma, Mb = ja.create_stödmoment(z)
Ra, Rb = ja.calculate_stödreaktion(Ma, Mb)
xbeam, mbeam, Mextreme = ja.böjmoment_längst_balken(Ma, Ra)
vbeam = ja.tvärkraft_längst_balken(Ra)
ja.myplot(xbeam, mbeam, vbeam)
ja.mytable(Ma, Mb, Ra, Rb, Mextreme, xbeam, mbeam)
        
        
        
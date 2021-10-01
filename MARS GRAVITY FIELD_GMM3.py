import datetime as dt
n1=dt.datetime.now()
import numpy as np
from math import atan2,degrees,sqrt,atan
from numpy import sin,cos,tan
from math import radians as r

def geocentric2xyz(phi,lon,hplusr):
    phi=r(phi)
    lon=r(lon)
    xi=hplusr*cos(phi)*cos(lon)
    yi=hplusr*cos(phi)*sin(lon)
    zi=hplusr*sin(phi)    
    return xi,yi,zi

def ell2xyz(lat,lon,h):
    lat=r(lat)
    lon=r(lon)
    N=R/((1-e**2*sin(lat)**2)**(0.5))
    X=(N+h)*cos(lat)*cos(lon)
    Y=(N+h)*cos(lat)*sin(lon)
    Z=((1-e**2)*N+h)*sin(lat)
    return X,Y,Z

def legendre(nmax,t):
    pnm=np.zeros((nmax+1,nmax+1),dtype=np.longdouble)
    pnm[0,0]=1
    pnm[1,1]=sqrt((1-t**2)*3)
    
    # köşegenleri doldurur
    for n in range(2,nmax+1):
        pnm[n,n] = sqrt((2*n+1)/(2*n))*sqrt(1-t**2)*pnm[n-1,n-1]

    # KÖŞEGENİN ALTINI DOLDURUR
    for n in range(1,nmax+1):
        for m in range(M[-1]+1):
            if n>=2 and m==0:
                anm= -(sqrt(2*n+1)/n)*((n-1)/sqrt(2*n-3))
                bnm= (sqrt(2*n+1)/n)*sqrt(2*n-1)
                pnm[n,m]= anm*(pnm[n-2,0])+bnm*t*pnm[n-1,0]

            elif n>=3 and 1<=m<=(n-2):
                anm1= -sqrt(((2*n+1)*(n+m-1)*(n-m-1))/((n-m)*(n+m)*(2*n-3)))
                bnm1= sqrt(((2*n-1)*(2*n+1))/((n-m)*(n+m)))
                pnm[n,m]= anm1*pnm[n-2,m]+bnm1*t*pnm[n-1,m]

            elif n>=1 and m==(n-1):
                pnm[n,m]= sqrt(2*n+1)*t*pnm[n-1,n-1]
    return pnm

def gama(phi):
    gama=(a*gamaE*(cos(phi)**2)+b*gamaP*sin(phi)**2)/(sqrt((a*cos(phi))**2+(b*sin(phi))**2))
    return gama

with open("gmm3_120_sha.txt", "r") as file:
    veri=file.readlines()
    
N,M,Cnm,Snm,Cnm_var,Snm_var=[],[],[],[],[],[]
for satır in veri:
    line=satır.split(",")
    N.append(int(line[0]))
    M.append(int(line[1]))
    Cnm.append(float(line[2]))
    Snm.append(float(line[3]))
#    Cnm_var.append(float(line[4]))
#    Snm_var.append(float(line[5]))

N=np.array(N)     
M=np.array(M)       
Cnm=np.array(Cnm)
Snm=np.array(Snm)
R=3.3960000000000000e+03
GM=4.2828372854187757e+04

#if N[1]==1 and N[2]==1:
#    N=np.delete(N,(1,2),axis=0)
#    M=np.delete(N,(1,2),axis=0)
#    Cnm=np.delete(N,(1,2),axis=0)
#    Snm=np.delete(N,(1,2),axis=0)

    
f=1/191.18  # BASIKLIK
e=sqrt(2*f-f**2)
eı=e/sqrt(1-e**2)
b=R-f*R
a=R
w=7.08822000000E-05  #marsın ekvatordaki açısal dönme hızı

# Normal Potansiyel Katsayıları
m=((w**2*a**2)*b)/GM
qo=((1+3/(eı**2))*atan(eı)-3/eı)/2
qoı=3*(1+1/(eı**2))*(1-atan(eı)/eı)-1
gamaE=(GM/(a*b))*(1-m-(m*eı*qoı)/(6*qo))
gamaP=(GM/(a*a))*(1+(m*eı*qoı)/(3*qo))
J2=((e**2)/3)*(1-(2*m*eı)/(15*qo))

CnmU=np.zeros(30)
for n in range(11):
    # Un-normalize harmonik katsayı
    J2n=((-1)**(n))*((3*e**(2*n))/((2*n+1)*(2*n+3)))*(1-n+(5*n*J2)/(e**2))    
    # Normalize harmonik katsayı
    C2n=(1/sqrt(4*n+1))*J2n   
    CnmU[n]=C2n
  

# Bozucu Potansitel Katsayıları
CnmT=Cnm.copy()
CnmT[1]=Cnm[1]-CnmU[1]      # C2
CnmT[8]=Cnm[8]-CnmU[2]      # C4
CnmT[19]=Cnm[19]-CnmU[3]    # C6
CnmT[34]=Cnm[34]-CnmU[4]    # C8
CnmT[53]=Cnm[53]-CnmU[5]    # C10
CnmT[76]=Cnm[78]-CnmU[6]    # C12
CnmT[103]=Cnm[103]-CnmU[7]  # C14
CnmT[134]=Cnm[134]-CnmU[8]  # C16
CnmT[169]=Cnm[169]-CnmU[9]  # C18
CnmT[208]=Cnm[208]-CnmU[10] # C20

gridV=np.zeros((181,361),dtype=np.longdouble)
gridT=np.zeros((181,361),dtype=np.longdouble)
gridN=np.zeros((181,361),dtype=np.longdouble)

x,y,z=[],[],[]

count=-1
for lat in range(-90,91):
    
    X,Y,Z=ell2xyz(lat,0,0)

    # Geocentric Coordinates
    phi=degrees(atan2(Z,sqrt(X**2+Y**2)))
    radius=sqrt(X**2+Y**2+Z**2)
    
    t=sin(r(phi))
    # legendre enleme bağlı olduğundan bir enlem boyunca legendre polinomu aynı
    pnm=legendre(N[-1],t)
    
    # Pnm matrisini sütun matris haline getirir.
    pnm=pnm[np.tril_indices(N[-1]+1)]
    if N[1]==2:    
        pnm=np.delete(pnm,(1,2),axis=0)  

    for lon in range(-180,181):

        lonr=r(lon)
        Vterm=sum((((R/radius)**N)*((Cnm*cos(M*lonr))+(Snm*sin(M*lonr))))*pnm)
        Tterm=sum((((R/radius)**N[1:])*((CnmT[1:]*cos(M[1:]*lonr))+(Snm[1:]*sin(M[1:]*lonr))))*pnm[1:]) # C00 hesaba katılmaz.
        
        V=(GM/radius)*Vterm
        T=(GM/radius)*Tterm
        
        Ni=T/gama(r(phi))
        
        gridV[lat,lon]=V
        gridT[lat,lon]=T
        gridN[lat,lon]=Ni
        
        count+=1
        rı=(radius+500*gridN[lat,lon])
#        rı=gridV[lat,lon]
        xi,yi,zi=geocentric2xyz(phi,lon,rı)

        x.append(xi)
        y.append(yi)
        z.append(zi)
    print(count)

x=np.array(x)
y=np.array(y)
z=np.array(z)

n2=dt.datetime.now()
print("geçen süre:",(n2-n1))
del GM,M,N,Snm,pnm,Ni,T,V,X,Y,Z,lat,line,lon,lonr,m,n,n1,n2,phi,satır,t,veri,xi,yi,zi

del C2n,Cnm,CnmT,CnmU,J2,J2n,R,Tterm,Vterm,a,b,count,e,eı,f,gamaE,gamaP,gridN,gridT,gridV,qo,qoı,radius,rı,w

x=np.reshape(x,(181,361))
y=np.reshape(y,(181,361))
z=np.reshape(z,(181,361))
s=[]
for i in range(len(x)):
   s.append((x[i]**2+y[i]**2+z[i]**2)**0.5)
s=np.array(s)
  
from mayavi import mlab
mlab.mesh(x, y, z, scalars=s,colormap="jet")









""" PLOTLY 3D"""


#import chart_studio.plotly as py
#import plotly.graph_objs as go
#
#trace1 = go.Scatter3d(
#    x=xi,
#    y=yi,
#    z=zi,
#    mode='markers',
#    marker=dict(
#        size=1,
#        color=zi,  # set color to an array/list of desired values
#        colorscale='Viridis',   # choose a colorscale
#        opacity=0.8
#    )
#)
#
#data = [trace1]
#layout = go.Layout(
#    margin=dict(
#        l=0,
#        r=0,
#        b=0,
#        t=0
#    )
#)
#import plotly
#fig = go.Figure(data=data, layout=layout)
#py.iplot(fig, filename='3d-scatter-colorscale')
#
#plotly.offline.plot({ "data": data, "layout": layout}, auto_open=True)





    
#from mpl_toolkits.mplot3d.axes3d import Axes3D
#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111,projection="3d")
#
#
#ax.scatter(x,y,z, c="blue",marker=".",s=0.1)
#ax.set_xlabel("x axis")
#ax.set_ylabel("y axis")
#ax.set_zlabel("z axis")


"""DOSYAYA YAZMA"""
#with open("nxyz.txt","w") as pot:
#    
#    pot.write("x\t y\t z\n") 
#    for i in range(65342):
#
#        pot.write("%s \t %s \t %s\n" % (xl[i],yl[i],zl[i]))
  
"""PLOT SURFACE"""
#lat1=-90
#lat2=90
#lon1=-180
#lon2=180
#
#latitude,longtitude=[],[]
#for lat in range(lat1,lat2+1):
#    latitude.append(lat)
#for lon in range(lon1,lon2+1):    
#    longtitude.append(lon)
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.axes3d import Axes3D
#from matplotlib import cm
#
#X,Y=np.meshgrid(latitude,longtitude)
#
#def potenti(lat,lon):
#    return gridN[lat,lon]

#Z=potenti(X,Y)

"""CONTOUR PLOTS"""
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z');

"""WIREFRAMES PLOTS"""
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_wireframe(X, Y, Z, color='black')
#ax.set_title('wireframe');

"""COLORMAP"""
#ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
#ax.set_title('surface');
#

"""DOSYAYA YAZMA"""
#with open("potential.txt","w") as pot:
#    
#    pot.write("lat\t lon\t potential_ell\n") 
#    for lat in range(lat1,lat2+1):
#        for lon in range(lon1,lon2+1):
#            pot.write("%s \t %s \t %s \n" % (lat,lon,grid[lat,lon]))
            


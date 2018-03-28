#-*- coding: UTF-8 -*-
import cv2
import numpy as np
import math
import copy
from PIL import Image
from pylab import *

I = Image.open('E:\\Study\\graduate\\python\\potato2.jpg')
I = np.array(I)
I =cv2.resize(I,(512,512),interpolation=cv2.INTER_CUBIC)
row=512
col=512

I=cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
OriginI=I

c=0
pi=3.14159
Npoint=150
x0=row/2
y0=col/2
r=y0/1.02
x=[x0+r*math.cos(c*2*pi/Npoint)]
y=[y0+r*math.sin(c*2*pi/Npoint)]
c+=1
while c<Npoint:
    xi=[x0+r*math.cos(c*2*pi/Npoint)]
    yi=[y0+r*math.sin(c*2*pi/Npoint)]

    x=np.column_stack((x,xi))
    y=np.column_stack((y,yi))
    c+=1

c=0
xg=[x0+r*math.cos(c*2*pi/Npoint)]
yg=[y0+r*math.sin(c*2*pi/Npoint)]
c+=1
while c<Npoint:
    xi=[x0+r*math.cos(c*2*pi/Npoint)];
    yi=[y0+r*math.sin(c*2*pi/Npoint)];

    xg=np.column_stack((xg,xi))
    yg=np.column_stack((yg,yi))
    c+=1

#高斯滤波
Igs = cv2.GaussianBlur(I, (9,9), 0, 0)
Igs=np.float64(Igs)

#求梯度 Gx,Gy
Gx = cv2.Sobel(Igs,cv2.CV_64F,1,0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
tGx = np.uint8(np.absolute(Gx))
Gy = cv2.Sobel(Igs,cv2.CV_64F,0,1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
tGy = np.uint8(np.absolute(Gy))

#黎曼距离
g=np.divide(1,(1+5*(multiply(Gx,Gx)+multiply(Gy,Gy))))

#求梯度gradGx,gradGy
gradGx = cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3)
gradGy = cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)

#迭代
epoch=50002
count=0
pcount=0
tcount=0
rx=np.uint64(x)
ry=np.uint64(y)
Igs=np.uint8(Igs)

while count<epoch:
    pcount=0
    show = copy(Igs)
    tempx=np.column_stack((x[0,Npoint-1],x))
    tempy=np.column_stack((y[0,Npoint-1],y))
    px=diff(tempx)
    py=diff(tempy)
    tempxx=np.column_stack((px,px[0,0]))
    tempyy=np.column_stack((py,py[0,0]))
    pxx=diff(tempxx)
    pyy=diff(tempyy)
    Nx=-np.divide(py,np.power((multiply(px,px)+multiply(py,py)),0.5));
    Ny=np.divide(px,np.power((multiply(px,px)+multiply(py,py)),0.5));  #法向量  np.power点幂  np.divide点除
    k=np.divide((multiply(px,pyy)- multiply(py,pxx)),np.power((multiply(px,px)+multiply(py,py)),1.5));  #曲率

    while pcount<Npoint:
        newg=g[rx[0,pcount],ry[0,pcount]]
        gradgx=gradGx[rx[0,pcount],ry[0,pcount]]
        gradgy=gradGy[rx[0,pcount],ry[0,pcount]]
        innermul=0
        innermul = gradgx*Nx[0,pcount]+gradgy*Ny[0,pcount]
        innermul=innermul/255
        if innermul>100:
            innermul = 0

        if k[0,pcount]<0:
            k[0,pcount]=abs(k[0,pcount])
        if k[0,pcount]>0.01:
            k[0,pcount]=0.01

        x[0,pcount]=x[0,pcount]+10*(newg*k[0,pcount]*Nx[0,pcount]-2*innermul*Nx[0,pcount])
        y[0,pcount]=y[0,pcount]+10*(newg*k[0,pcount]*Ny[0,pcount]-2*innermul*Ny[0,pcount])

        pcount+=1
    rx=np.uint64(x)
    ry=np.uint64(y)
    fcount=0
    while fcount<Npoint:
        if (fcount+1)<Npoint:
            pt1=(int(round(y[0,fcount])),int(round(x[0,fcount])))
            pt2=(int(round(y[0,fcount+1])),int(round(x[0,fcount+1])))
            cv2.line(show,pt1,pt2,(200,0,0),2)
        if (fcount+1)==Npoint:
            pt0=(int(round(y[0,0])),int(round(x[0,0])))
            pt1=(int(round(y[0,fcount])),int(round(x[0,fcount])))
            cv2.line(show,pt1,pt0,(200,0,0),2)
        fcount+=1
    if count%10==0:
        cv2.imshow("Processing...",show)
        cv2.waitKey(30)
    count+=1
cv2.destroyAllWindows()
fcount=0

while fcount<Npoint:
    #Igs[x[0,fcount], y[0,fcount]] = 255
    if (fcount+1)<Npoint:
        pt1=(int(round(y[0,fcount])),int(round(x[0,fcount])))
        pt2=(int(round(y[0,fcount+1])),int(round(x[0,fcount+1])))
        cv2.line(Igs,pt1,pt2,(200,0,0),2)
    if (fcount+1)==Npoint:
        pt0=(int(round(y[0,0])),int(round(x[0,0])))
        pt1=(int(round(y[0,fcount])),int(round(x[0,fcount])))
        cv2.line(Igs,pt1,pt0,(200,0,0),2)
    fcount+=1

#OriginI2=np.uint8(OriginI2)
#cv2.imshow("OriginalImage",OriginI2)
#cv2.waitKey(0)

cv2.imshow("Processed",Igs)
cv2.waitKey(0)









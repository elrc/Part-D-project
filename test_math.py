import math

pcx=1257
pcy=389
cx=1243
cy=426

hp=1920
vp=1080
fr=30
t=1/fr
d=0.55

pi=math.pi
tan=math.tan
cos=math.cos
sin=math.sin
sqrt=math.sqrt
atan=math.atan
rat=atan(9/16)
fv=78*(pi/180)
hfv=fv*cos(rat)
hhfv=hfv/2
vfv=fv*sin(rat)
hvfv=vfv/2
rpph=hfv/hp
rppv=vfv/vp
ang1=fv/2
w=2*d*tan(ang1)
x=w*cos(rat)
y=w*sin(rat)
prcx=rpph*pcx
prcy=rppv*pcy
rcx=rpph*cx
rcy=rppv*cy

if prcx<=hhfv:
    x1=(x/2)-(d*tan(hhfv-prcx))
elif prcx>hhfv:
    x1=(x/2)+(d*tan(prcx-hhfv))
if prcy<=hvfv:
    y1=(y/2)-(d*tan(hvfv-prcy))
elif prcy>hvfv:
    y1=(y/2)+(d*tan(prcy-hvfv))

if rcx<=hhfv:
    x2=(x/2)-(d*tan(hhfv-rcx))
elif rcx>hhfv:
    x2=(x/2)+(d*tan(rcx-hhfv))
if rcy<=hvfv:
    y2=(y/2)-(d*tan(hvfv-rcy))
elif rcy>hvfv:
    y2=(y/2)+(d*tan(rcy-hvfv))

ds=sqrt(((x1-x2)**2)+((y1-y2)**2))

v=ds/t

#print("x=",x)
#print("y=",y)
#print("prcx=",prcx)
#print("prcy=",prcy)
#print("rcx=",rcx)
#print("rcy=",rcy)
#print("x1=",x1)
#print("y1=",y1)
#print("x2=",x2)
#print("y2=",y2)
print("ds=",ds)
print("v=",v)
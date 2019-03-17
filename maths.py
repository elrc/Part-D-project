import math

pi = math.pi
tan = math.tan
cos = math.cos
sin = math.sin
sqrt = math.sqrt
atan = math.atan

def velocity_math(counter,time,speedarr,velocityarr,currentX,currentY,previouscX,previouscY,distance,angle,ratio,rpphorizontal,rppvertical,hhorizontalfv,hverticalfv,river_angle):
    if counter == 0:
        previouscX = currentX
        previouscY = currentY
        speedmax,speedavg,velocitymax,velocityavg = 0,0,0,0
        strspeed = "0"
        counter = 1
    else:
        w = 2 * distance * tan(angle)
        x = w * cos(ratio)
        y = w * sin(ratio)
        prcx = rpphorizontal * previouscX
        prcy = rppvertical * previouscY
        rcx= rpphorizontal * currentX
        rcy = rppvertical * currentY
        
        if prcx <= hhorizontalfv:
            x1 = (x / 2) - (distance * tan(hhorizontalfv - prcx))
        elif prcx > hhorizontalfv:
            x1 = (x / 2) + (distance * tan(prcx - hhorizontalfv))
        if prcy <= hverticalfv:
            y1 = (y / 2) - (distance * tan(hverticalfv - prcy))
        elif prcy > hverticalfv:
            y1 = (y / 2) + (distance * tan(prcy - hverticalfv))
                                
        if rcx <= hhorizontalfv:
            x2 = (x / 2) - (distance * tan(hhorizontalfv - rcx))
        elif rcx > hhorizontalfv:
            x2 = (x / 2) + (distance * tan(rcx - hhorizontalfv))
        if rcy <= hverticalfv:
            y2 = (y / 2) - (distance * tan(hverticalfv - rcy))
        elif rcy > hverticalfv:
            y2 = (y / 2) + (distance * tan(rcy - hverticalfv))

        ds = sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
        if x1 == x2:
            vel_ang = 0
        elif y1 == y2 and x2 > x1:
            vel_ang = pi / 2
        elif y1 == y2 and x2 < x1:
            vel_ang = (-1 * pi) / 2
        elif y1 == y2 and x1 == x2:
            vel_ang = 0
        else:
            vel_ang = atan((x1 - x2) / (y1 - y2))
        vel_ang_deg = vel_ang * (180 / pi)
        s = ds / time
        strspeed = "%f m/s" % s
        speedarr.append(s)
        speedmax = max(speedarr)
        speedavg = sum(speedarr) / len(speedarr)
        ang_vel = s * cos(river_angle - vel_ang)
        velocityarr.append(ang_vel)
        velocitymax = max(velocityarr)
        velocityavg = sum(velocityarr) / len(velocityarr)
        previouscX = currentX
        previouscY = currentY
    return (counter,speedarr,velocityarr,previouscX,previouscY,strspeed,speedmax,speedavg,velocitymax,velocityavg)
import math as m

def H(c1,c2,c3,c0):
    c0=float(c0)
    a=c1/c0

    b=c2/c0
    c=c3/c0
    H=-a*m.log(a,2)-b*m.log(b,2)-c*m.log(c,2)
    return H


print (H(2,3,1,6))
print (H(5,6,1,6))
print (H(6,4,2,6))

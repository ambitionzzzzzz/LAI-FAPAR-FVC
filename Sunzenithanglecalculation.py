import math as mt


'''
True solar time calculation
(https://faculty.eng.ufl.edu/jonathan-scheffe/wp-content/uploads/sites/100/2020/08/Solar-Time1419.pdf)

Calculation of solar zenith angle
(https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle)
'''


def TST(LOC, N, K, st):
    '''
    parameter:
    LOC: the longitude of the observation point(LOC has units degrees)
    N: day of the year
    K: K=1 or 2. If the longitude of the observation point is east longitude, then K=1; otherwise, K=2
    st: Observation Point Greenwich Mean Time (GMT)(st has units h)(24h)
    LST: the longitude of the central meridian in the time zone where the observation point is located(LST has units degrees)
    B: B has units radians
    E: E is the equation of time in minutes

    return:
    tst: True solar time((tst has units h))
    '''
    LST = mt.ceil((LOC-7.5)/15)*15
    B = mt.radians((N-1)*360/365)
    E = 229.2*(0.000075 + 0.001868*mt.cos(B) - 0.032077*mt.sin(B) - 0.014615*mt.cos(2*B) - 0.04089*mt.sin(2*B))
    if K == 1:
        tst = st - (4 * (LST - LOC) + E) / 60
    else:
        tst = st + (4 * (LST - LOC) + E) / 60
    return tst


def SZA(tst, LAT, N):
    '''
    parameter:
    tst: True solar time(tst has units h)
    SHA: Solar hour angle(SHA has units degrees)
    LAT: The latitude of the observation point \
         (the value range is -90°~90°, the positive or negative depends on the northern and southern hemispheres)(degrees)
    YRA: solar declination angle(YRA has units degrees)

    return:
    sza：solar zenith angle(sza has units degrees)
    csza：cosine of solar zenith angle
    '''
    SHA = 15*(tst-12)
    YRA = mt.radians(-23.45*mt.cos(mt.radians((360/365)*(N+10))))
    csza = mt.sin(YRA)*mt.sin(mt.radians(LAT)) + mt.cos(YRA)*mt.cos(mt.radians(LAT))*mt.cos(mt.radians(SHA))
    sza = mt.degrees(mt.acos(csza))
    '''
    if sza > 90:
        sza = 180 - sza
    else:
        sza = sza
    '''
    return sza, csza, SHA

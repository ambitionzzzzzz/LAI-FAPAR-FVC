from scipy.integrate import quad
from numpy import sin, cos, pi, tan, arctan, arccos
import numpy as np
from Sunzenithanglecalculation import SZA
# import warnings
import pandas as pd

def Aceta(sza):
    cotsza = 1 / abs(tan(sza))
    tant = cotsza
    tli = arctan(tant)  # 求出临界值
    return tli

def FAPAR_B(lidfa, LAI, sunzenith):
    sza = sunzenith * pi / 180

    def G1(t, lidfa):
        lidfa = lidfa * pi /180
        A_ceta = cos(t)
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * sin(t)) / (A * (((cos(t) ** 2) + (X ** 2) * (sin(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    def G2(t, lidfa):

        lidfa = lidfa * pi / 180
        A_ceta = cos(t) * cos(sza)
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * sin(t)) / (A * (((cos(t) ** 2) + (X ** 2) * (sin(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    def G3(t, lidfa):
        lidfa = lidfa * pi / 180
        fi = arccos((1 / tan(sza)) * (1 / tan(t)))
        A_ceta = cos(t) * cos(sza) * (1 + (2 / pi) * (tan(fi) - fi))
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * sin(t)) / (A * (((cos(t) ** 2) + (X ** 2) * (sin(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    if sza == 0:
        G = quad(G1, 0, pi / 2, args=lidfa)[0]
    else:
        tli = Aceta(sza)
        # print(tli)
        G = quad(G2, 0, tli, args=lidfa)[0] + quad(G3, tli, pi/2, args=lidfa)[0]
    lamda = 1
    P = np.exp((-1 * lamda * G * LAI) / cos(sza))
    fapar_b = 1 - P
    return fapar_b


def lim(x):
    # 求出临界值
    return arctan(1 / tan(x))


def FAPAR_W(lidfa, LAI):

    def G1(t, x, lidfa):
        lidfa = lidfa * pi / 180
        A_ceta = cos(t) * cos(x)
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * cos(t)) / (A * (((sin(t) ** 2) + (X ** 2) * (cos(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    def G2(t, x, lidfa):
        lidfa = lidfa * pi / 180
        fi = arccos((1 / tan(x)) * (1 / tan(t)))
        A_ceta = cos(t) * cos(x) * (1 + (2 / pi) * (tan(fi) - fi))
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * cos(t)) / (A * (((sin(t) ** 2) + (X ** 2) * (cos(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    lamda = 1
    fapar_w = quad(lambda x : (1 - np.exp((-1 * lamda * LAI * quad(lambda t: G1(t, x, lidfa), 0, lim(x))[0]) / cos(x)) * np.exp((-1 * lamda * LAI * quad(lambda t: G2(t, x, lidfa), lim(x), pi / 2)[0]) / cos(x))) * cos(x) * sin(x),
                   0, pi/2)[0]
    fapar_w = 2 * fapar_w
    return fapar_w


def FVC(lidfa, LAI):
    # 使用的是旋转椭球体叶角密度分布
    def G(t, lidfa):
        lidfa = lidfa * pi / 180
        A_ceta = cos(t)
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * sin(t)) / (A * (((cos(t) ** 2) + (X ** 2) * (sin(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    G = quad(G, 0, pi/2, args=lidfa)[0]
    lamda = 1
    P = np.exp(-1 * lamda * G * LAI)
    fvc = 1 - P
    return fvc



def FAPAR_B1(lidfa, LAI, date, latitude):
    sza = SZA(12, latitude, date)[0] * pi / 180

    def FZ(t):
        return cos(t)

    fenzi = quad(FZ, sza, pi/2)[0]

    def G1(t, x, lidfa):
        lidfa = lidfa * pi / 180
        A_ceta = cos(t) * cos(x)
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * cos(t)) / (A * (((sin(t) ** 2) + (X ** 2) * (cos(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    def G2(t, x, lidfa):
        lidfa = lidfa * pi / 180
        fi = arccos((1 / tan(x)) * (1 / tan(t)))
        A_ceta = cos(t) * cos(x) * (1 + (2 / pi) * (tan(fi) - fi))
        X = -3 + ((lidfa / 9.65) ** (-0.6061))
        if abs(X - 1) <= 1e-5:
            A = 2
        elif X - 1 > 1e-5:
            e = (1 - (X ** (-2))) ** 0.5
            A = X + np.log((1 + e) / (1 - e)) / (2 * e * X)
        else:
            e = (1 - (X ** 2)) ** 0.5
            A = X + ((np.arcsin(e)) / e)
        f_ceta = (2 * (X ** 3) * cos(t)) / (A * (((sin(t) ** 2) + (X ** 2) * (cos(t) ** 2)) ** 2))

        return A_ceta * f_ceta

    lamda = 1
    fenmu = quad(lambda x : (1 - np.exp((-1 * lamda * LAI * quad(lambda t: G1(t, x, lidfa), 0, lim(x))[0]) / cos(x)) * np.exp((-1 * lamda * LAI * quad(lambda t: G2(t, x, lidfa), lim(x), pi / 2)[0]) / cos(x))) * cos(x),
                   sza, pi/2)[0]


    fapar_b1 = fenmu / fenzi
    return fapar_b1


def calculate_FVC(row):
    print("Calculating FVC for row:", row.name)
    return FVC(float(row['lidfa']), float(row['LAI']))


def calculate_Fapar_b(row):
    print("Calculating FAPAR_B for row:", row.name)
    return FAPAR_B(float(row['lidfa']), float(row['LAI']), float(row['sunzenith']))

def calculate_Fapar_b1(row):
    print("Calculating FAPAR_B1 for row:", row.name)
    return FAPAR_B1(float(row['lidfa']), float(row['LAI']), int(row['date(365)']), float(row['latitude']))

def calculate_Fapar_w(row):
    print("Calculating FAPAR_W for row:", row.name)
    return FAPAR_W(float(row['lidfa']), float(row['LAI']))






df = pd.read_excel('E:\Prosail\project\全新数据(优化太阳天顶角)\Datas.xlsx')

df['FAPAR_B'] = df.apply(calculate_Fapar_b, axis=1)
# df['FAPAR_W'] = df.apply(calculate_Fapar_w, axis=1)
df['FVC'] = df.apply(calculate_FVC, axis=1)

df.to_excel('E:\Prosail\project\全新数据(优化太阳天顶角)\Datas_all_modified.xlsx', index=False)

print("------------------finished---------------------------------")



# a = FAPAR_B1(60.1918, 3, 15, 60)
# b = FAPAR_B(60.1918, 3, 15, 60)
# b = FAPAR_B(40, 0.1, 10, 55)
#b = FAPAR_W(56.8775645, 9.5)

# print(a)
# print(b)
# print(b)
#print(b)
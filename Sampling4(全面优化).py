import random
import numpy as np
import tkinter as tk
import pandas as pd
from scipy.stats import truncnorm
from itertools import product
from Sunzenithanglecalculation import SZA

'''
The sampling algorithm is based on https://step.esa.int/docs/extra/ATBD_S2ToolBox_V2.1.pdf
The parameters sampled by the script are for the ProsailD + 4sail model(https://github.com/jgomezdans/prosail)
                                                                       (https://www.sciencedirect.com/science/article/abs/pii/S0034425717300962)

The script will eventually generate two ".xlsx" files: Parameters_ranges.xlsx (parameter range information), 
                                                       Parameters_table.xlsx (complete parameter value table)

All parameters used satisfy the truncated Gaussian distribution                                                      
'''



def Gauss_value(lower, upper, mu, sigma, size):
    '''
    生成size个满足截断高斯分布的随机值函数
    :param lower:
    :param upper:
    :param mu:
    :param sigma:
    :param size:
    :return:
    '''
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)

def clicked():
    '''
    Response function of "start" button
    paralist: Information matrix of all parameter distribution characteristics
              [ [Minimum value of N parameter, Maximum value of N parameter, average of N parameters,
              Standard deviation of N parameters, Number of categories of N parameters],
                [Minimum value of Cab parameter, Maximum value of Cab parameter, average of Cab parameters,
              Standard deviation of Cab parameters, Number of categories of Cab parameters],
                                  .........................
              ]
    levels: The interval divided by each parameter according to the number of categories(Nb_class)
    para_range: Parameter range information and distribution characteristic matrix("Parameter", "Minimum", "Maximum", "Mode", "Std", "Nb_class", "Levels)
    combinations: The full permutation matrix obtained by combining parameter categories with each other, calculated by the function rand_class_value()


    :return:
    Parameters_ranges.xlsx (parameter range information),
    Parameters_table.xlsx (complete parameter value table)
    '''

    paralist = []
    paralist1 = []
    paralist2 = []
    paralist3 = []
    paralist4 = []
    paralist5 = []
    paralist6 = []
    paralist7 = []
    paralist8 = []
    paralist9 = []
    paralist10 = []
    paralist11 = []
    for i in range(1, 6):
        paralist1.append(float(globals()[f"txt1{i}"].get()))
        paralist2.append(float(globals()[f"txt2{i}"].get()))
        paralist3.append(float(globals()[f"txt3{i}"].get()))
        paralist4.append(float(globals()[f"txt4{i}"].get()))
        paralist5.append(float(globals()[f"txt5{i}"].get()))
        paralist6.append(float(globals()[f"txt6{i}"].get()))
        paralist7.append(float(globals()[f"txt7{i}"].get()))
        paralist8.append(float(globals()[f"txt8{i}"].get()))
        paralist9.append(float(globals()[f"txt9{i}"].get()))
        paralist10.append(float(globals()[f"txt10{i}"].get()))
        paralist11.append(float(globals()[f"txt11{i}"].get()))
    for i in range(1, 12):
        paralist.append(locals()[f"paralist{i}"])

    levels = []
    for i in range(0, np.array(paralist).shape[0]):
        levels.append(gauss(paralist[i]))

    para_range = []
    for i in range(1, np.array(paralist).shape[0] + 1):
        para_range.append([globals()[f"label{i}"].cget("text"), globals()[f"txt{i}1"].get(), \
                           globals()[f"txt{i}2"].get(), globals()[f"txt{i}3"].get(), \
                           globals()[f"txt{i}4"].get(), globals()[f"txt{i}5"].get(), \
                           levels[i - 1]])
    para_range.append(["soli_type", 1, int(txt122.get())])
    para_range.append(["Sunzenith(degree)"])
    para_range.append(["Viewzenith(degree)"])
    para_range.append(["Relazimuth(degree)"])
    columns = ["Parameter", "Minimum", "Maximum", "Mode", "Std", "Nb_class", "Levels"]
    df = pd.DataFrame(para_range, columns=columns)
    df.to_excel('E:\Prosail\project\全新数据(优化太阳天顶角)\Parameters_ranges.xlsx', index=False)

    print("The parameter range information table has been generated................")
    print("-------------------------------------------------------------------------------")
    print("Parameter table is being generated...............................")
    combinations = rand_class_value(paralist)[0]  # 所有的组合
    lengh = rand_class_value(paralist)[1]  # 所有组合的总数

    #----------------------------------------------------------------------------------------------------------------------------
    Single_class_lai = int(lengh / int(globals()["txt85"].get()))  # 每类LAI对应的样本数
    print("Scl:", Single_class_lai)

    # 生成LAI的随机抽样组合
    lower = float(globals()["txt81"].get())
    upper = float(globals()["txt82"].get())
    mu = float(globals()["txt83"].get())
    sigma = float(globals()["txt84"].get())

    LAI_distribution = [0.5, upper, mu, sigma, int(globals()["txt85"].get()) - 1]
    LAI_classes = gauss(LAI_distribution)
    LAI_values = []
    LAI_values.append([random.uniform(0, 0.5) for _ in range(Single_class_lai)])  # 第一类设置为[0, 0.5]

    for i in range(len(LAI_classes) - 1):
        if i != len(LAI_classes) - 2:
            lai02 = [random.uniform(LAI_classes[i], LAI_classes[i + 1]) for _ in range(Single_class_lai)]
            LAI_values.append(lai02)
        else:
            LAI_values.append(Gauss_value(LAI_classes[i], LAI_classes[i+1], mu, sigma, Single_class_lai).tolist())

    # 优化ant的随机抽样组合
    Single_class_ant = int(lengh / int(globals()["txt75"].get()))  # 每类ant对应的样本数
    lower_ant = float(globals()["txt71"].get())
    upper_ant = float(globals()["txt72"].get())
    mu_ant = float(globals()["txt73"].get())
    sigma_ant = float(globals()["txt74"].get())

    ant_distribution = [lower_ant, upper_ant, mu_ant, sigma_ant, int(globals()["txt75"].get())]
    ant_classes = gauss(ant_distribution)
    ant_values4 = Gauss_value(ant_classes[len(ant_classes) - 2], ant_classes[len(ant_classes) - 1], mu_ant, sigma_ant, Single_class_ant).tolist()  # 只优化最后1类

    # 优化Car的随机抽样组合
    Single_class_car = int(lengh / int(globals()["txt35"].get()))  # 每类Car对应的样本数
    lower_car = float(globals()["txt31"].get())
    upper_car = float(globals()["txt32"].get())
    mu_car = float(globals()["txt33"].get())
    sigma_car = float(globals()["txt34"].get())

    car_distribution = [lower_car, upper_car, mu_car, sigma_car, int(globals()["txt35"].get())]
    car_classes = gauss(car_distribution)
    car_values4 = Gauss_value(car_classes[len(car_classes) - 2], car_classes[len(car_classes) - 1], mu_car, sigma_car, Single_class_car).tolist()  # 只优化最后1类

    # 优化hspot的随机采样组合(hspot只有1类)
    Single_class_hot = int(lengh / int(globals()["txt105"].get()))  # 每类hspot对应的样本数
    lower_hot = float(globals()["txt101"].get())
    upper_hot = float(globals()["txt102"].get())
    mu_hot = float(globals()["txt103"].get())
    sigma_hot = float(globals()["txt104"].get())

    hot_values = Gauss_value(lower_hot, upper_hot, mu_hot, sigma_hot, Single_class_hot).tolist()

    # 优化rsoil随机采样组合(全部， gauss————uni)
    Single_class_rsoil = int(lengh / int(globals()["txt115"].get()))  # 每类hspot对应的样本数
    lower_rsoil = float(globals()["txt111"].get())
    upper_rsoil = float(globals()["txt112"].get())
    rsoil_classes = np.linspace(lower_rsoil, upper_rsoil, int(globals()["txt115"].get()) + 1)

    rsoil_values = []
    for i in range(len(rsoil_classes) - 1):
        rsoil_values.append([random.uniform(rsoil_classes[i], rsoil_classes[i + 1]) for _ in range(Single_class_rsoil)])

    #--------------------------------------------------------------------------------------------------------------------------------------


    para_table = []
    for i in range(0, lengh):
        nucla = Sampling(combinations, i, levels, LAI_values, car_values4, ant_values4, hot_values, rsoil_values)
        parameters = co_distribution(nucla)
        para_list = zenith(parameters)
        para_table.append(para_list)
    print("LAI_values:", LAI_values)
    columns = ["N", "Cab", "Car", "Cbrown", "Cw", "Cm", "ant", "LAI", "lidfa", "hspot", "rsoil", "soil_type",
               "date(365)", "solar_time", "latitude", "sunzenith", "viewzenith", "relazimuth"]
    df = pd.DataFrame(para_table, columns=columns)
    df.to_excel('E:\Prosail\project\全新数据(优化太阳天顶角)\Parameters_table.xlsx', index=False)
    print("Parameter table has been generated...............................")


def gauss(distribution_charact):
    '''
    Calculate the classification interval of the variable according to the normal distribution law so that the cumulative probability of each interval is the same.

    :param:
    distribution_charact: distribution information matrix----[Minimum value of parameter, Maximum value of parameter, average of parameters,
                                                                    Standard deviation of parameters, Number of categories of parameters]
    :return:
    intervals: Each classification interval calculated based on distribution information
    '''
    # Set the characteristic parameters of the distribution
    lower_bound = distribution_charact[0]
    upper_bound = distribution_charact[1]
    mean = distribution_charact[2]
    std_dev = distribution_charact[3]
    num_intervals = int(distribution_charact[4])
    # Create a truncated Gaussian distribution object
    a = (lower_bound - mean) / std_dev
    b = (upper_bound - mean) / std_dev
    trunc_dist = truncnorm(a, b, loc=mean, scale=std_dev)
    # Calculate cumulative probability
    cumulative_probs = np.linspace(0, 1, num_intervals + 1)
    # Calculate the value range of each interval
    intervals = [round(trunc_dist.ppf(prob), 4) for prob in cumulative_probs]

    return intervals


def rand_class_value(paralist):
    '''
    Generate an index matrix consisting of different categories of all parameters combined with each other

    :param:
    paralist: Information matrix of all parameter distribution characteristics

    :return:
    combinations: Index matrix for all category combinations
    lengh: Number of rows of index matrix
    '''
    # Get the number of categories. The 5th digit of each row of the distribution feature matrix represents the number of categories.
    nucla_N = int(paralist[0][4])
    nucla_Cab = int(paralist[1][4])
    nucla_Car = int(paralist[2][4])
    nucla_Cbrown = int(paralist[3][4])
    nucla_Cw = int(paralist[4][4])
    nucla_Cm = int(paralist[5][4])
    nucla_ant = int(paralist[6][4])
    nucla_LAI = int(paralist[7][4])
    nucla_lidfa = int(paralist[8][4])
    nucla_hspot = int(paralist[9][4])
    nucla_rsoil = int(paralist[10][4])

    # ranges-------------------------------------------
    rangeN = range(1, nucla_N + 1)
    rangeCab = range(1, nucla_Cab + 1)
    rangeCar = range(1, nucla_Car + 1)
    rangeCbrown = range(1, nucla_Cbrown + 1)
    rangeCw = range(1, nucla_Cw + 1)
    rangeCm = range(1, nucla_Cm + 1)
    rangeant = range(1, nucla_ant + 1)
    rangeLAI = range(1, nucla_LAI + 1)
    rangelidfa = range(1, nucla_lidfa + 1)
    rangehspot = range(1, nucla_hspot + 1)
    rangersoil = range(1, nucla_rsoil + 1)

    # Use itertools.product to generate permutations and combinations
    combinations = list(product(rangeN, rangeCab, rangeCar, rangeCbrown, rangeCw, rangeCm, rangeant, \
                                rangeLAI, rangelidfa, rangehspot, rangersoil))

    lengh = np.array(combinations).shape[0]
    return combinations, lengh


def Sampling(combinations, i, levels, LAI_values, car_values4, ant_values4, hot_values, rsoil_values):
    '''
    Randomly obtain parameter values in the corresponding categories according to the index matrix of category combinations

    :param combinations:Index matrix for all category combinations
    :param i: Number of cycles
    :param levels: The interval divided by each parameter according to the number of categories(Nb_class)
    :return:
    nucla: The matrix obtained by randomly selecting each parameter contains 18 parameters (18 columns)
                                            ("N", "Cab", "Car", "Cbrown", "Cw", "Cm", "ant", "LAI", "lidfa", "hspot", "rsoil",
                                            "soil_type","date(365)", "solar_time", "latitude", "sunzenith", "viewzenith", "relazimuth"),
                                            this function is only for the first 12 parameters
    '''
    cla_N = levels[0]
    cla_Cab = levels[1]
    cla_Car = levels[2]
    cla_Cbrown = levels[3]
    cla_Cw = levels[4]
    cla_Cm = levels[5]
    cla_ant = levels[6]
    cla_LAI = levels[7]
    cla_lidfa = levels[8]
    cla_hspot = levels[9]
    cla_rsoil = levels[10]

    nucla = [0] * 18

    # Randomly select values in the interval
    nu_N = combinations[i][0]
    nucla[0] = round(random.uniform(cla_N[nu_N - 1], cla_N[nu_N]), 4)

    nu_Cab = combinations[i][1]
    nucla[1] = round(random.uniform(cla_Cab[nu_Cab - 1], cla_Cab[nu_Cab]), 4)

    nu_Car = combinations[i][2]
    if nu_Car != int(globals()["txt35"].get()):
        nucla[2] = round(random.uniform(cla_Car[nu_Car - 1], cla_Car[nu_Car]), 4)
    else:
        nucla[2] = car_values4[0]
        del car_values4[0]


    nu_Cbrown = combinations[i][3]
    nucla[3] = round(random.uniform(cla_Cbrown[nu_Cbrown - 1], cla_Cbrown[nu_Cbrown]), 4)

    nu_Cw = combinations[i][4]
    nucla[4] = round(random.uniform(cla_Cw[nu_Cw - 1], cla_Cw[nu_Cw]), 4)

    nu_Cm = combinations[i][5]
    nucla[5] = round(random.uniform(cla_Cm[nu_Cm - 1], cla_Cm[nu_Cm]), 4)

    nu_ant = combinations[i][6]
    if nu_ant != int(globals()["txt75"].get()):
        nucla[6] = round(random.uniform(cla_ant[nu_ant - 1], cla_ant[nu_ant]), 4)
    else:
        nucla[6] = ant_values4[0]
        del ant_values4[0]


    nu_LAI = combinations[i][7]
    nucla[7] = LAI_values[nu_LAI - 1][0]
    del LAI_values[nu_LAI - 1][0]
    print("nu_lAI_lenth:", len(LAI_values[nu_LAI - 1]))

    # nucla[7] = round(random.uniform(cla_LAI[nu_LAI - 1], cla_LAI[nu_LAI]), 4)

    nu_lidfa = combinations[i][8]
    nucla[8] = round(random.uniform(cla_lidfa[nu_lidfa - 1], cla_lidfa[nu_lidfa]), 4)

    nu_hspot = combinations[i][9]  # hspot只有1类
    # nucla[9] = round(random.uniform(cla_hspot[nu_hspot - 1], cla_hspot[nu_hspot]), 4)
    nucla[9] = hot_values[0]
    del hot_values[0]


    nu_rsoil = combinations[i][10]
    # nucla[10] = round(random.uniform(cla_rsoil[nu_rsoil - 1], cla_rsoil[nu_rsoil]), 4)
    nucla[10] = rsoil_values[nu_rsoil - 1][0]
    del rsoil_values[nu_rsoil - 1][0]


    nucla[11] = random.randint(1, int(txt122.get()))  # Randomly select soil type

    return nucla


def co_distribution(nucla):
    '''
    Calculate the co-distribution values of parameters related to the LAI distribution

    co-distribution:
    LAI：0-15
    Cab: Vmax=90   Vmin=5/3*LAI+20   V=（V1-20）/70*(Vmax-Vmin)+Vmin
    Cbrown: Vmax=2-0.12*LAI	   Vmin=0   V=V1/2*(Vmax-Vmin)+Vmin
    Cw: Vmax=0.85-0.05/15*LAI  Vmin=0.1/15*LAI+0.6  V=(V1-0.6)/0.25*(Vmax-Vmin)+Vmin
    N: Vmax=1.8          Vmin=0.1/15*LAI+1.2      V=( V1-1.2)/0.6*(Vmax-Vmin)+Vmin
    Cm: Vmax=0.011    Vmin=0.002/15*LAI+0.003  V=( V1-0.003)/0.008*(Vmax-Vmin)+Vmin
    ALA/lidfa: Vmax=80-LAI  Vmin=5/3*LAI+30   V=( V1-30)/50*(Vmax-Vmin)+Vmin
    Hot: Vmax=0.5   Vmin=0.1  V=V1
    rsoil: Vmax=3.5-2.3/15*LAI   Vmin=0.5   V=(V1-0.5)/3*(Vmax-Vmin)+Vmin
    Car: Vmax=20      Vmin=4/15*LAI+2   V= (V1-2)/18*(Vmax-Vmin)+Vmin
    ant: Vmax=8       Vmin=0.5/15*LAI    V= V1/8*(Vmax-Vmin)+Vmin

    :param nucla: The matrix obtained by randomly selecting each parameter contains 18 parameters (18 columns),
                  this function is only for the first 12 parameters
    :return:
    parameters: The value matrix after each parameter is co-distributed
    '''
    LAI = nucla[7]
    parameters = [0] * len(nucla)

    VmaxN = 1.8
    VminN = 0.1 / 15 * LAI + 1.2
    parameters[0] = round((nucla[0] - 1.2) / 0.6 * (VmaxN - VminN) + VminN, 4)

    VmaxCab = 90
    VminCab = 5 / 3 * LAI + 20
    parameters[1] = round((nucla[1] - 20) / 70 * (VmaxCab - VminCab) + VminCab, 4)

    VmaxCar = 20
    VminCar = 4 / 15 * LAI + 2
    parameters[2] = round((nucla[2] - 2) / 18 * (VmaxCar - VminCar) + VminCar, 4)

    VmaxCbrown = 2 - 0.12 * LAI
    VminCbrown = 0
    parameters[3] = round(nucla[3] / 2 * (VmaxCbrown - VminCbrown) + VminCbrown, 4)

    VmaxCw = 0.85 - 0.05 / 15 * LAI
    VminCw = 0.1 / 15 * LAI + 0.6
    parameters[4] = round((nucla[4] - 0.6) / 0.25 * (VmaxCw - VminCw) + VminCw, 4)

    VmaxCm = 0.011
    VminCm = 0.002 / 15 * LAI + 0.03
    parameters[5] = round((nucla[5] - 0.003) / 0.008 * (VmaxCm - VminCm) + VminCm, 4)

    Vmaxant = 8
    Vminant = 0.5 / 15 * LAI
    parameters[6] = round(nucla[6] / 8 * (Vmaxant - Vminant) + Vminant, 4)

    Vmaxlidfa = 80 - LAI
    Vminlidfa = 5 / 3 * LAI + 30
    parameters[8] = round((nucla[8] - 30) / 50 * (Vmaxlidfa - Vminlidfa) + Vminlidfa, 4)

    Vmaxrsoil = 3.5 - 2.3 / 15 * LAI
    Vminrsoil = 0.5
    parameters[10] = round((nucla[10] - 0.5) / 3 * (Vmaxrsoil - Vminrsoil) + Vminrsoil, 4)

    parameters[7] = LAI
    parameters[9] = nucla[9]
    parameters[11] = nucla[11]
    # print(parameters)
    return parameters


def zenith(parameters):
    '''
    Random value function for the orientation parameter ("sunzenith", "viewzenith", "relazimuth")

    :param parameters: The value matrix after each parameter is co-distributed
    :return:
    para_raw: One line of complete parameter value information
    '''

    time = [1, 91, 182, 273, 365]  # Divide the year into four intervals and randomly select values
    random_time = random.randint(1, 4)
    n = random.randint(time[random_time - 1], time[random_time])
    LAT = random.uniform(-56, 66)

    while True:
        tst = random.uniform(10, 14)
        if SZA(tst, LAT, n)[0] < 90:
            parameters[12] = n  # Date (number of days in the year)
            parameters[13] = tst  # Randomly select local solar time(10-14)
            parameters[14] = LAT  # Randomly select the local latitude, negative values are south latitudes(-56-66)
            break

    szA = round(SZA(tst, LAT, n)[0], 4)  # Call the SZA function to calculate the solar zenith angle

    vzA = round(random.uniform(0, 40),
                4)  # Take random values according to the distribution of "viewzenith" and "relazimuth"(ATBD.pdf)
    RA = [0, 65, 115, 180]
    random_ra = random.randint(1, 2)
    if random_ra == 1:
        rA = round(random.uniform(RA[random_ra - 1], RA[random_ra]), 4)
    else:
        rA = round(random.uniform(RA[random_ra], RA[random_ra + 1]), 4)
    parameters[15] = szA
    parameters[16] = vzA
    parameters[17] = rA
    para_raw = parameters
    return para_raw


# -------------------------------------------------------------------------GUI interface construction----------------------------------------------------------------
window = tk.Tk()
window.geometry('400x600')
window.title("Generation of parameter table")

# 创建Canvas以容纳所有Frame
canvas = tk.Canvas(window)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 创建垂直滚动条
SVBar = tk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
SVBar.pack(side=tk.RIGHT, fill=tk.Y)

# 连接Canvas和滚动条
canvas.configure(yscrollcommand=SVBar.set)

# 创建一个Frame用于放置所有Frame
frame_container = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame_container, anchor="nw")

# 创建Frame并添加子部件
frames = []
for i in range(1, 12):
    frame = tk.Frame(frame_container, width=30, height=100, relief='groove', bd=2)
    frame.grid(row=i - 1, column=0)
    lbl1 = tk.Label(frame, text="Minimum:", width=10, anchor="w")
    lbl1.grid(column=0, row=1)
    lbl2 = tk.Label(frame, text="Maximum:", width=10, anchor="w")
    lbl2.grid(column=0, row=2)
    lbl3 = tk.Label(frame, text="Mode:", width=10, anchor="w")
    lbl3.grid(column=0, row=3)
    lbl4 = tk.Label(frame, text="Std:", width=10, anchor="w")
    lbl4.grid(column=0, row=4)
    lbl5 = tk.Label(frame, text="Nb_Class:", width=10, anchor="w")
    lbl5.grid(column=0, row=5)
    frames.append(frame)

label1 = tk.Label(frames[0], text="N(Leaf structure parameter)(N/A)", anchor="center")
label1.grid(column=0, row=0, columnspan=2)
label2 = tk.Label(frames[1], text="Cab(Chlorophyll a+b concentration)(ug/cm2)", anchor="center")
label2.grid(column=0, row=0, columnspan=2)
label3 = tk.Label(frames[2], text="Car(Carotenoid concentration)(ug/cm2)", anchor="center")
label3.grid(column=0, row=0, columnspan=2)
label4 = tk.Label(frames[3], text="Cbrown(Brown pigment)(N/A)", anchor="center")
label4.grid(column=0, row=0, columnspan=2)
label5 = tk.Label(frames[4], text="Cw(Equivalent water thickiness)(g/cm2)", anchor="center")
label5.grid(column=0, row=0, columnspan=2)
label6 = tk.Label(frames[5], text="Cm(Dry matter content)(g/cm2)", anchor="center")
label6.grid(column=0, row=0, columnspan=2)
label7 = tk.Label(frames[6], text="ant(Anthocyanins content)(ug/cm2)", anchor="center")
label7.grid(column=0, row=0, columnspan=2)
label8 = tk.Label(frames[7], text="LAI(Leaf Area Index)(N/A)", anchor="center")
label8.grid(column=0, row=0, columnspan=2)
label9 = tk.Label(frames[8], text="lidfa(Leaf angle distribution)(degree)", anchor="center")
label9.grid(column=0, row=0, columnspan=2)
label10 = tk.Label(frames[9], text="hspot(Hotspot parameter)(N/A)", anchor="center")
label10.grid(column=0, row=0, columnspan=2)
label11 = tk.Label(frames[10], text="rsoil(Soil brigthness factor)(N/A)", anchor="center")
label11.grid(column=0, row=0, columnspan=2)

frame1 = tk.Frame(frame_container, width=30, height=50, relief='groove', bd=2)
frame1.grid(row=12, column=0)
start_button = tk.Button(frame1, text="start", command=clicked, anchor="center")
start_button.grid(column=1, row=0, columnspan=2)

frame2 = tk.Frame(frame_container, width=30, height=50, relief='groove', bd=2)
frame2.grid(row=11, column=0)
lbl12 = tk.Label(frame2, text="Number of soil types:", width=18, anchor="w")
lbl12.grid(column=0, row=1)
txt122 = tk.Entry(frame2, width=17, state='normal')
txt122.insert(0, "7")
txt122.grid(column=1, row=1)

txt11 = tk.Entry(frames[0], width=25, state='normal')
txt11.insert(0, "1.2")
txt11.grid(column=1, row=1)
txt12 = tk.Entry(frames[0], width=25, state='normal')
txt12.insert(0, "1.8")
txt12.grid(column=1, row=2)
txt13 = tk.Entry(frames[0], width=25, state='normal')
txt13.insert(0, "1.5")
txt13.grid(column=1, row=3)
txt14 = tk.Entry(frames[0], width=25, state='normal')
txt14.insert(0, "0.3")
txt14.grid(column=1, row=4)
txt15 = tk.Entry(frames[0], width=25, state='normal')
txt15.insert(0, "3")
txt15.grid(column=1, row=5)

txt21 = tk.Entry(frames[1], width=25, state='normal')
txt21.insert(0, "20")
txt21.grid(column=1, row=1)
txt22 = tk.Entry(frames[1], width=25, state='normal')
txt22.insert(0, "90")
txt22.grid(column=1, row=2)
txt23 = tk.Entry(frames[1], width=25, state='normal')
txt23.insert(0, "45")
txt23.grid(column=1, row=3)
txt24 = tk.Entry(frames[1], width=25, state='normal')
txt24.insert(0, "30")
txt24.grid(column=1, row=4)
txt25 = tk.Entry(frames[1], width=25, state='normal')
txt25.insert(0, "4")
txt25.grid(column=1, row=5)

txt31 = tk.Entry(frames[2], width=25, state='normal')
txt31.insert(0, "2")
txt31.grid(column=1, row=1)
txt32 = tk.Entry(frames[2], width=25, state='normal')
txt32.insert(0, "20")
txt32.grid(column=1, row=2)
txt33 = tk.Entry(frames[2], width=25, state='normal')
txt33.insert(0, "6")
txt33.grid(column=1, row=3)
txt34 = tk.Entry(frames[2], width=25, state='normal')
txt34.insert(0, "3")
txt34.grid(column=1, row=4)
txt35 = tk.Entry(frames[2], width=25, state='normal')
txt35.insert(0, "4")
txt35.grid(column=1, row=5)

txt41 = tk.Entry(frames[3], width=25, state='normal')
txt41.insert(0, "0")
txt41.grid(column=1, row=1)
txt42 = tk.Entry(frames[3], width=25, state='normal')
txt42.insert(0, "2")
txt42.grid(column=1, row=2)
txt43 = tk.Entry(frames[3], width=25, state='normal')
txt43.insert(0, "0")
txt43.grid(column=1, row=3)
txt44 = tk.Entry(frames[3], width=25, state='normal')
txt44.insert(0, "0.3")
txt44.grid(column=1, row=4)
txt45 = tk.Entry(frames[3], width=25, state='normal')
txt45.insert(0, "3")
txt45.grid(column=1, row=5)

txt51 = tk.Entry(frames[4], width=25, state='normal')
txt51.insert(0, "0.6")
txt51.grid(column=1, row=1)
txt52 = tk.Entry(frames[4], width=25, state='normal')
txt52.insert(0, "0.85")
txt52.grid(column=1, row=2)
txt53 = tk.Entry(frames[4], width=25, state='normal')
txt53.insert(0, "0.75")
txt53.grid(column=1, row=3)
txt54 = tk.Entry(frames[4], width=25, state='normal')
txt54.insert(0, "0.08")
txt54.grid(column=1, row=4)
txt55 = tk.Entry(frames[4], width=25, state='normal')
txt55.insert(0, "4")
txt55.grid(column=1, row=5)

txt61 = tk.Entry(frames[5], width=25, state='normal')
txt61.insert(0, "0.003")
txt61.grid(column=1, row=1)
txt62 = tk.Entry(frames[5], width=25, state='normal')
txt62.insert(0, "0.011")
txt62.grid(column=1, row=2)
txt63 = tk.Entry(frames[5], width=25, state='normal')
txt63.insert(0, "0.005")
txt63.grid(column=1, row=3)
txt64 = tk.Entry(frames[5], width=25, state='normal')
txt64.insert(0, "0.005")
txt64.grid(column=1, row=4)
txt65 = tk.Entry(frames[5], width=25, state='normal')
txt65.insert(0, "4")
txt65.grid(column=1, row=5)

txt71 = tk.Entry(frames[6], width=25, state='normal')
txt71.insert(0, "0")
txt71.grid(column=1, row=1)
txt72 = tk.Entry(frames[6], width=25, state='normal')
txt72.insert(0, "8")
txt72.grid(column=1, row=2)
txt73 = tk.Entry(frames[6], width=25, state='normal')
txt73.insert(0, "0.5")
txt73.grid(column=1, row=3)
txt74 = tk.Entry(frames[6], width=25, state='normal')
txt74.insert(0, "2")
txt74.grid(column=1, row=4)
txt75 = tk.Entry(frames[6], width=25, state='normal')
txt75.insert(0, "4")
txt75.grid(column=1, row=5)

txt81 = tk.Entry(frames[7], width=25, state='normal')
txt81.insert(0, "0")
txt81.grid(column=1, row=1)
txt82 = tk.Entry(frames[7], width=25, state='normal')
txt82.insert(0, "15")
txt82.grid(column=1, row=2)
txt83 = tk.Entry(frames[7], width=25, state='normal')
txt83.insert(0, "2")
txt83.grid(column=1, row=3)
txt84 = tk.Entry(frames[7], width=25, state='normal')
txt84.insert(0, "3")
txt84.grid(column=1, row=4)
txt85 = tk.Entry(frames[7], width=25, state='normal')
txt85.insert(0, "7")
txt85.grid(column=1, row=5)

txt91 = tk.Entry(frames[8], width=25, state='normal')
txt91.insert(0, "30")
txt91.grid(column=1, row=1)
txt92 = tk.Entry(frames[8], width=25, state='normal')
txt92.insert(0, "80")
txt92.grid(column=1, row=2)
txt93 = tk.Entry(frames[8], width=25, state='normal')
txt93.insert(0, "60")
txt93.grid(column=1, row=3)
txt94 = tk.Entry(frames[8], width=25, state='normal')
txt94.insert(0, "30")
txt94.grid(column=1, row=4)
txt95 = tk.Entry(frames[8], width=25, state='normal')
txt95.insert(0, "3")
txt95.grid(column=1, row=5)

txt101 = tk.Entry(frames[9], width=25, state='normal')
txt101.insert(0, "0.1")
txt101.grid(column=1, row=1)
txt102 = tk.Entry(frames[9], width=25, state='normal')
txt102.insert(0, "0.5")
txt102.grid(column=1, row=2)
txt103 = tk.Entry(frames[9], width=25, state='normal')
txt103.insert(0, "0.2")
txt103.grid(column=1, row=3)
txt104 = tk.Entry(frames[9], width=25, state='normal')
txt104.insert(0, "0.5")
txt104.grid(column=1, row=4)
txt105 = tk.Entry(frames[9], width=25, state='normal')
txt105.insert(0, "1")
txt105.grid(column=1, row=5)

txt111 = tk.Entry(frames[10], width=25, state='normal')
txt111.insert(0, "0.5")
txt111.grid(column=1, row=1)
txt112 = tk.Entry(frames[10], width=25, state='normal')
txt112.insert(0, "3.5")
txt112.grid(column=1, row=2)
txt113 = tk.Entry(frames[10], width=25, state='normal')
txt113.insert(0, "1.2")
txt113.grid(column=1, row=3)
txt114 = tk.Entry(frames[10], width=25, state='normal')
txt114.insert(0, "2.0")
txt114.grid(column=1, row=4)
txt115 = tk.Entry(frames[10], width=25, state='normal')
txt115.insert(0, "4")
txt115.grid(column=1, row=5)

# 更新Canvas尺寸
frame_container.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

window.mainloop()
# -------------------------------------------------------------------------GUI interface construction----------------------------------------------------------------






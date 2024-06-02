import pandas as pd
import numpy as np
import prosail
'''
Calculate the brightness of each channel based on the value information of each parameter.
Prosail-----------(https://github.com/jgomezdans/prosail)

soil: Soil reflectance data read from .xlsx file (modify path)
data: The spectral response function value of each channel read from PSF.xlsx.(Modify path)
parameters: complete parameter value table(Modify path)

return:
Training_table.xlsx(Complete table with brightness information for each channel)
'''


def con(a, b):
    c = [0] * 2101
    for i in range(0, len(b)):
            c[int(a[i]-400)] = b[i]
    return c


def smooth(c):
    '''
    The smoothing function is used to smooth the original reflectance data by interpolation \
    and extend it to the length of 2101 (because the Prosail model supports input reflectance data \
    from 400nm-2500nm with an interval of 1nm).

    :return:
    smoothed_array: Smoothed reflectance array (400nm-2500nm)
    '''
    # Find the index of a non-zero element
    non_zero_indices = np.where(np.array(c) != 0)[0]
    # Perform linear interpolation
    smoothed_array = c.copy()

    for i in range(len(non_zero_indices) - 1):
        start_index = non_zero_indices[i]
        end_index = non_zero_indices[i + 1]
        # Get start value and end value
        start_value = c[start_index]
        end_value = c[end_index]
        # Calculate interpolation step size
        step = (end_value - start_value) / (end_index - start_index)
        # Use interpolation to fill the 0 value part
        for j in range(start_index + 1, end_index):
            smoothed_array[j] = smoothed_array[j - 1] + step
    return smoothed_array


def soil_libra():
    # Specify the path to the Excel file to be read(.xlsx)
    excel_file_path = 'E:\Prosail\土壤反射率\干湿土壤反射率数据\反射率数据/all.xlsx'    # !!!! Pay attention to modifying the path(Soil reflectance)
    # Use pandas to read Excel files without specifying column names
    df = pd.read_excel(excel_file_path, header=None)
    # Get the number of columns
    num_columns = df.shape[1]
    column = []
    # Iterate through each column and save it into the corresponding array
    for column_index in range(num_columns):
        column_data = df.iloc[:, column_index].tolist()
        column.append(column_data)
    # The "column" two-dimensional array contains each column of data in the .xlsx file, \
    # where the first column is the wavelength (nm) corresponding to the corresponding reflectance.
    wavelength = [round(num) for num in column[0]]
    spectral_soil = []
    for i in range(1, num_columns):
        c = con(wavelength, column[i])
        spectral_soil.append(smooth(c))
    return spectral_soil


def PSD(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts, tto, psi, ant, soil_type, rsoil):
    rsoil0 = np.array(soil[soil_type - 1]) * rsoil
    spectral = prosail.run_prosail(n=n, cab=cab, car=car, cbrown=cbrown, cw=cw, cm=cm, lai=lai, lidfa=lidfa, hspot=hspot, \
                                   tts=tts, tto=tto, psi=psi, ant=ant, alpha=40.0, prospect_version='D', typelidf=2, lidfb=0.0, \
                                   factor='SDR', rsoil0=rsoil0, rsoil=None, psoil=None, soil_spectrum1=None, soil_spectrum2=None)

    psfpro1221 = []
    psfpro2221 = []
    psfpro3221 = []
    psfpro1222 = []
    psfpro2222 = []
    psfpro3222 = []
    #  R = Sum(PROSAIL(i)*PSF(i))/Sum(PSF(i)), where i are the wavelengths for the current channel.
    #  Sum(PROSAIL(i)*PSF(i))
    for j in range(0, len(band_lengh)):
        psfpro1221.append(Psf1221[j] * spectral[int(band_lengh[j] - 400)])
        psfpro2221.append(Psf2221[j] * spectral[int(band_lengh[j] - 400)])
        psfpro3221.append(Psf3221[j] * spectral[int(band_lengh[j] - 400)])
        psfpro1222.append(Psf1222[j] * spectral[int(band_lengh[j] - 400)])
        psfpro2222.append(Psf2222[j] * spectral[int(band_lengh[j] - 400)])
        psfpro3222.append(Psf3222[j] * spectral[int(band_lengh[j] - 400)])

    CH01221 = np.sum(np.array(psfpro1221)) / psf1221
    CH02221 = np.sum(np.array(psfpro2221)) / psf2221
    CH03221 = np.sum(np.array(psfpro3221)) / psf3221
    CH01222 = np.sum(np.array(psfpro1222)) / psf1222
    CH02222 = np.sum(np.array(psfpro2222)) / psf2222
    CH03222 = np.sum(np.array(psfpro3222)) / psf3222

    return [CH01221, CH02221, CH03221, CH01222,  CH02222, CH03222]
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculated_CH01221(row):
    print("Calculating CH01221 for row:", row.name)
    results = PSD(float(row['N']), float(row['Cab']), float(row['Car']), float(row['Cbrown']), float(row['Cw']), float(row['Cm']), \
               float(row['LAI']), float(row['lidfa']), float(row['hspot']), float(row['sunzenith']), float(row['viewzenith']), float(row['relazimuth']),\
               float(row['ant']), int(row['soil_type']), float(row['rsoil']))
    return results[0]


def calculated_CH02221(row):
    print("Calculating CH02221 for row:", row.name)
    results = PSD(float(row['N']), float(row['Cab']), float(row['Car']), float(row['Cbrown']), float(row['Cw']), float(row['Cm']), \
                  float(row['LAI']), float(row['lidfa']), float(row['hspot']), float(row['sunzenith']),
                  float(row['viewzenith']), float(row['relazimuth']), \
                  float(row['ant']), int(row['soil_type']), float(row['rsoil']))
    return results[1]

def calculated_CH03221(row):
    print("Calculating CH03221 for row:", row.name)
    results = PSD(float(row['N']), float(row['Cab']), float(row['Car']), float(row['Cbrown']), float(row['Cw']),
                  float(row['Cm']), \
                  float(row['LAI']), float(row['lidfa']), float(row['hspot']), float(row['sunzenith']),
                  float(row['viewzenith']), float(row['relazimuth']), \
                  float(row['ant']), int(row['soil_type']), float(row['rsoil']))
    return results[2]

def calculated_CH01222(row):
    print("Calculating CH01222 for row:", row.name)
    results = PSD(float(row['N']), float(row['Cab']), float(row['Car']), float(row['Cbrown']), float(row['Cw']),
                  float(row['Cm']), \
                  float(row['LAI']), float(row['lidfa']), float(row['hspot']), float(row['sunzenith']),
                  float(row['viewzenith']), float(row['relazimuth']), \
                  float(row['ant']), int(row['soil_type']), float(row['rsoil']))
    return results[3]

def calculated_CH02222(row):
    print("Calculating CH02222 for row:", row.name)
    results = PSD(float(row['N']), float(row['Cab']), float(row['Car']), float(row['Cbrown']), float(row['Cw']),
                  float(row['Cm']), \
                  float(row['LAI']), float(row['lidfa']), float(row['hspot']), float(row['sunzenith']),
                  float(row['viewzenith']), float(row['relazimuth']), \
                  float(row['ant']), int(row['soil_type']), float(row['rsoil']))
    return results[4]

def calculated_CH03222(row):
    print("Calculating CH03222 for row:", row.name)
    results = PSD(float(row['N']), float(row['Cab']), float(row['Car']), float(row['Cbrown']), float(row['Cw']),
                  float(row['Cm']), \
                  float(row['LAI']), float(row['lidfa']), float(row['hspot']), float(row['sunzenith']),
                  float(row['viewzenith']), float(row['relazimuth']), \
                  float(row['ant']), int(row['soil_type']), float(row['rsoil']))
    return results[5]


def calculated_all(row):
    print("Calculating num for row:", row.name)
    results = PSD(float(row['N']), float(row['Cab']), float(row['Car']), float(row['Cbrown']), float(row['Cw']),
                  float(row['Cm']), \
                  float(row['LAI']), float(row['lidfa']), float(row['hspot']), float(row['sunzenith']),
                  float(row['viewzenith']), float(row['relazimuth']), \
                  float(row['ant']), int(row['soil_type']), float(row['rsoil']))
    return results
# --------------------------------------------------------------------------------------------------------------------
soil = soil_libra()

# Read the spectral response function
data = pd.read_excel('E:\Prosail\project\PSF.xlsx', header = None)

# Sum(PSF(i))
psf1221 = data.iloc[:, 1].sum()
psf2221 = data.iloc[:, 2].sum()
psf3221 = data.iloc[:, 3].sum()
psf1222 = data.iloc[:, 4].sum()
psf2222 = data.iloc[:, 5].sum()
psf3222 = data.iloc[:, 6].sum()

band_lengh = round(data.iloc[:, 0])
Psf1221 = data.iloc[:, 1]
Psf2221 = data.iloc[:, 2]
Psf3221 = data.iloc[:, 3]
Psf1222 = data.iloc[:, 4]
Psf2222 = data.iloc[:, 5]
Psf3222 = data.iloc[:, 6]

# Read parameter value table(Obtained from Sampling.py)
parameters = pd.read_excel('E:\Prosail\project\全新数据(优化太阳天顶角)/Parameters_table.xlsx')    # !!!! Pay attention to modifying the path
# training_table = parameters.copy().assign(CH01_221=None, CH02_221=None, CH03_221=None, CH01_222=None, CH02_222=None, CH03_222=None)
# print(training_table)
'''
parameters['CH01221'] = parameters.apply(calculated_CH01221, axis=1)
parameters['CH02221'] = parameters.apply(calculated_CH02221, axis=1)
parameters['CH03221'] = parameters.apply(calculated_CH03221, axis=1)
parameters['CH01222'] = parameters.apply(calculated_CH01222, axis=1)
parameters['CH02222'] = parameters.apply(calculated_CH02222, axis=1)
parameters['CH03222'] = parameters.apply(calculated_CH03222, axis=1)
'''
new_columns = parameters.apply(calculated_all, axis=1, result_type='expand')
parameters[['CH01221', 'CH02221', 'CH03221', 'CH01222', 'CH02222', 'CH03222']] = new_columns

parameters.to_excel('E:\Prosail\project\全新数据(优化太阳天顶角)/Datas.xlsx', index=False)    # !!!! Pay attention to modifying the path





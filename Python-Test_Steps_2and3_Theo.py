#cd "C:\Python Test"
#python -m pip install --upgrade pip
#python -m pip install -U pip setuptools
#python -m pip install scipy
#python -c "import scipy.stats"
#python -m pip install matplotlib
#python Python-Test_Step2and3_Theo.py
import pandas as pd
import numpy as np
from numpy import inf
import scipy as sp
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

# Main Dataframe
file = 'Global Template.xlsx'
df = pd.read_excel(file)
print(df)

#solve for div/0 (returns 'inf')
df = df.replace([np.inf], 0)
df.sort_values(by='MOS', ascending=False)

#Dataframes for regions
df_shire = df.loc[(df['SNL1'] == 'Shire')]
df_polombia = df.loc[(df['SNL1'] == 'Polombia')]
df_sangala = df.loc[(df['SNL1'] == 'Sangala')]
df_turgistan = df.loc[(df['SNL1'] == 'Turgistan')]
df_urkesh = df.loc[(df['SNL1'] == 'Urkesh')]
df_nukula = df.loc[(df['SNL1'] == "Nuku'la Atoll")]
df_molvania = df.loc[(df['SNL1'] == 'Molvanîa')]
df_hogwarts = df.loc[(df['SNL1'] == 'Hogwarts')]
df_flausenthurm = df.loc[(df['SNL1'] == 'Flausenthurm')]
df_westerose = df.loc[(df['SNL1'] == 'Westerose')]

#Get numpy arrays of MOS by region
shire_mos_array = np.array(df_shire['MOS'])
polombia_mos_array = np.array(df_polombia['MOS'])
sangala_mos_array = np.array(df_sangala['MOS'])
turgistan_mos_array = np.array(df_turgistan['MOS'])
urkesh_mos_array = np.array(df_urkesh['MOS'])
nukula_mos_array = np.array(df_nukula['MOS'])
molvania_mos_array = np.array(df_molvania['MOS'])
hogwarts_mos_array = np.array(df_hogwarts['MOS'])
flausenthurm_mos_array = np.array(df_flausenthurm['MOS'])
westerose_mos_array = np.array(df_westerose['MOS'])
mos_array = np.array(df['MOS'])

# Mean & Median MOS by region
shire_mos = "Average MOS in Shire: " + str(np.mean(shire_mos_array)) + "; Median MOS in Shire: " + str(np.median(shire_mos_array)) + "; lowest MOS in Shire: " + str(np.min(shire_mos_array)) + "; highest MOS in Shire: " + str(np.max(shire_mos_array)) + "; standard deviation of MOS in Shire: " + str(np.std(shire_mos_array))
polombia_mos = "Average MOS in Polombia: " + str(np.mean(polombia_mos_array)) + "; Median MOS in Polombia: " + str(np.median(polombia_mos_array)) + "; lowest MOS in Polombia: " + str(np.min(polombia_mos_array)) + "; highest MOS in Polombia: " + str(np.max(polombia_mos_array)) + "; standard deviation of MOS in Polombia: " + str(np.std(polombia_mos_array))
sangala_mos = "Average MOS in Sangala: " + str(np.mean(sangala_mos_array)) + "; Median MOS in Sangala: " + str(np.median(sangala_mos_array)) + "; lowest MOS in Sangala: " + str(np.min(sangala_mos_array)) + "; highest MOS in Sangala: " + str(np.max(sangala_mos_array)) + "; standard deviation of MOS in Sangala: " + str(np.std(sangala_mos_array))
turgistan_mos = "Average MOS in Turgistan: " + str(np.mean(turgistan_mos_array)) + "; Median MOS in Turgistan: " + str(np.median(turgistan_mos_array)) + "; lowest MOS in Turgistan: " + str(np.min(turgistan_mos_array)) + "; highest MOS in Turgistan: " + str(np.max(turgistan_mos_array)) + "; standard deviation of MOS in Turgistan: " + str(np.std(turgistan_mos_array))
urkesh_mos = "Average MOS in Urkesh: " + str(np.mean(urkesh_mos_array)) + "; Median MOS in Urkesh: " + str(np.median(urkesh_mos_array)) + "; lowest MOS in Urkesh: " + str(np.min(urkesh_mos_array)) + "; highest MOS in Urkesh: " + str(np.max(urkesh_mos_array)) + "; standard deviation of MOS in Urkesh: " + str(np.std(urkesh_mos_array))
nukula_mos = "Average MOS in Nuku'la Atoll: " + str(np.mean(nukula_mos_array)) + "; Median MOS in Nuku'la Atoll: " + str(np.median(nukula_mos_array)) + "; lowest MOS in Nuku'la Atoll: " + str(np.min(nukula_mos_array)) + "; highest MOS in Nuku'la Atoll: " + str(np.max(nukula_mos_array)) + "; standard deviation of MOS in Nuku'la Atoll: " + str(np.std(nukula_mos_array))
molvania_mos = "Average MOS in Molvanîa: " + str(np.mean(molvania_mos_array)) + "; Median MOS in Molvanîa: " + str(np.median(molvania_mos_array)) + "; lowest MOS in Molvanîa: " + str(np.min(molvania_mos_array)) + "; highest MOS in Molvanîa: " + str(np.max(molvania_mos_array)) + "; standard deviation of MOS in Molvanîa: " + str(np.std(molvania_mos_array))
hogwarts_mos = "Average MOS in Hogwarts: " + str(np.mean(hogwarts_mos_array)) + "; Median MOS in Hogwarts: " + str(np.median(hogwarts_mos_array)) + "; lowest MOS in Hogwarts: " + str(np.min(hogwarts_mos_array)) + "; highest MOS in Hogwarts: " + str(np.max(hogwarts_mos_array)) + "; standard deviation of MOS in Hogwarts: " + str(np.std(hogwarts_mos_array))
flausenthurm_mos = "Average MOS in Flausenthurm: " + str(np.mean(flausenthurm_mos_array)) + "; Median MOS in Flausenthurm: " + str(np.median(flausenthurm_mos_array)) + "; lowest MOS in Flausenthurm: " + str(np.min(flausenthurm_mos_array)) + "; highest MOS in Flausenthurm: " + str(np.max(flausenthurm_mos_array)) + "; standard deviation of MOS in Flausenthurm: " + str(np.std(flausenthurm_mos_array))
westerose_mos = "Average MOS in Westerose: " + str(np.mean(westerose_mos_array)) + "; Median MOS in Westerose: " + str(np.median(westerose_mos_array)) + "; lowest MOS in Westerose: " + str(np.min(westerose_mos_array)) + "; highest MOS in Westerose: " + str(np.max(westerose_mos_array)) + "; standard deviation of MOS in Westerose: " + str(np.std(westerose_mos_array))
countrywide_mos = "Average MOS in-country: " + str(np.mean(mos_array)) + "; Median MOS in-country: " + str(np.median(mos_array)) + "; lowest MOS in-country: " + str(np.min(mos_array)) + "; highest MOS in-country: " + str(np.max(mos_array)) + "; standard deviation of MOS in-country: " + str(np.std(mos_array))

#Print outputs
print(shire_mos)
print(polombia_mos)
print(sangala_mos)
print(turgistan_mos)
print(urkesh_mos)
print(nukula_mos)
print(molvania_mos)
print(hogwarts_mos)
print(flausenthurm_mos)
print(westerose_mos)
print(countrywide_mos)


#Skewness
print("MOS skewness: " + str(sp.stats.skew(mos_array)))
print("MOS kurtosis: " + str(sp.stats.kurtosis(mos_array)))
#Normal distribution random samples
norm_dist = np.random.normal(3.97956215880793, 30.51691636001731, 6496)
print("Normal Distribution skewness: " + str(sp.stats.skew(norm_dist)))
print("Normal Distribution kurtosis: " + str(sp.stats.kurtosis(norm_dist)))

# Outliers - 3*StdDev
mos_stddev = "MOS standard deviation" + str(np.std(mos_array))
mos_3stddev = "MOS outlier floor: " + str(3*(np.std(mos_array)))
print(mos_stddev)
print(mos_3stddev)

#Remove outliers
mos_array = mos_array[(mos_array < 91.55)]
shire_mos_array = shire_mos_array[(shire_mos_array < 91.55)]
polombia_mos_array = polombia_mos_array[(polombia_mos_array < 91.55)]
sangala_mos_array = sangala_mos_array[(sangala_mos_array < 91.55)]
turgistan_mos_array = turgistan_mos_array[(turgistan_mos_array < 91.55)]
urkesh_mos_array = urkesh_mos_array[(urkesh_mos_array < 91.55)]
nukula_mos_array = nukula_mos_array[(nukula_mos_array < 91.55)]
molvania_mos_array = molvania_mos_array[(molvania_mos_array < 91.55)]
hogwarts_mos_array = hogwarts_mos_array[(hogwarts_mos_array < 91.55)]
flausenthurm_mos_array = flausenthurm_mos_array[(flausenthurm_mos_array < 91.55)]
westerose_mos_array = westerose_mos_array[(westerose_mos_array < 91.55)]
shire_mos_array = shire_mos_array[(shire_mos_array < 91.55)]

shire_mos = "Average MOS in Shire: " + str(np.mean(shire_mos_array)) + "; Median MOS in Shire: " + str(np.median(shire_mos_array)) + "; lowest MOS in Shire: " + str(np.min(shire_mos_array)) + "; highest MOS in Shire: " + str(np.max(shire_mos_array)) + "; standard deviation of MOS in Shire: " + str(np.std(shire_mos_array))
polombia_mos = "Average MOS in Polombia: " + str(np.mean(polombia_mos_array)) + "; Median MOS in Polombia: " + str(np.median(polombia_mos_array)) + "; lowest MOS in Polombia: " + str(np.min(polombia_mos_array)) + "; highest MOS in Polombia: " + str(np.max(polombia_mos_array)) + "; standard deviation of MOS in Polombia: " + str(np.std(polombia_mos_array))
sangala_mos = "Average MOS in Sangala: " + str(np.mean(sangala_mos_array)) + "; Median MOS in Sangala: " + str(np.median(sangala_mos_array)) + "; lowest MOS in Sangala: " + str(np.min(sangala_mos_array)) + "; highest MOS in Sangala: " + str(np.max(sangala_mos_array)) + "; standard deviation of MOS in Sangala: " + str(np.std(sangala_mos_array))
turgistan_mos = "Average MOS in Turgistan: " + str(np.mean(turgistan_mos_array)) + "; Median MOS in Turgistan: " + str(np.median(turgistan_mos_array)) + "; lowest MOS in Turgistan: " + str(np.min(turgistan_mos_array)) + "; highest MOS in Turgistan: " + str(np.max(turgistan_mos_array)) + "; standard deviation of MOS in Turgistan: " + str(np.std(turgistan_mos_array))
urkesh_mos = "Average MOS in Urkesh: " + str(np.mean(urkesh_mos_array)) + "; Median MOS in Urkesh: " + str(np.median(urkesh_mos_array)) + "; lowest MOS in Urkesh: " + str(np.min(urkesh_mos_array)) + "; highest MOS in Urkesh: " + str(np.max(urkesh_mos_array)) + "; standard deviation of MOS in Urkesh: " + str(np.std(urkesh_mos_array))
nukula_mos = "Average MOS in Nuku'la Atoll: " + str(np.mean(nukula_mos_array)) + "; Median MOS in Nuku'la Atoll: " + str(np.median(nukula_mos_array)) + "; lowest MOS in Nuku'la Atoll: " + str(np.min(nukula_mos_array)) + "; highest MOS in Nuku'la Atoll: " + str(np.max(nukula_mos_array)) + "; standard deviation of MOS in Nuku'la Atoll: " + str(np.std(nukula_mos_array))
molvania_mos = "Average MOS in Molvanîa: " + str(np.mean(molvania_mos_array)) + "; Median MOS in Molvanîa: " + str(np.median(molvania_mos_array)) + "; lowest MOS in Molvanîa: " + str(np.min(molvania_mos_array)) + "; highest MOS in Molvanîa: " + str(np.max(molvania_mos_array)) + "; standard deviation of MOS in Molvanîa: " + str(np.std(molvania_mos_array))
hogwarts_mos = "Average MOS in Hogwarts: " + str(np.mean(hogwarts_mos_array)) + "; Median MOS in Hogwarts: " + str(np.median(hogwarts_mos_array)) + "; lowest MOS in Hogwarts: " + str(np.min(hogwarts_mos_array)) + "; highest MOS in Hogwarts: " + str(np.max(hogwarts_mos_array)) + "; standard deviation of MOS in Hogwarts: " + str(np.std(hogwarts_mos_array))
flausenthurm_mos = "Average MOS in Flausenthurm: " + str(np.mean(flausenthurm_mos_array)) + "; Median MOS in Flausenthurm: " + str(np.median(flausenthurm_mos_array)) + "; lowest MOS in Flausenthurm: " + str(np.min(flausenthurm_mos_array)) + "; highest MOS in Flausenthurm: " + str(np.max(flausenthurm_mos_array)) + "; standard deviation of MOS in Flausenthurm: " + str(np.std(flausenthurm_mos_array))
westerose_mos = "Average MOS in Westerose: " + str(np.mean(westerose_mos_array)) + "; Median MOS in Westerose: " + str(np.median(westerose_mos_array)) + "; lowest MOS in Westerose: " + str(np.min(westerose_mos_array)) + "; highest MOS in Westerose: " + str(np.max(westerose_mos_array)) + "; standard deviation of MOS in Westerose: " + str(np.std(westerose_mos_array))
countrywide_mos = "Average MOS in-country: " + str(np.mean(mos_array)) + "; Median MOS in-country: " + str(np.median(mos_array)) + "; lowest MOS in-country: " + str(np.min(mos_array)) + "; highest MOS in-country: " + str(np.max(mos_array)) + "; standard deviation of MOS in-country: " + str(np.std(mos_array))
print(shire_mos)
print(polombia_mos)
print(sangala_mos)
print(turgistan_mos)
print(urkesh_mos)
print(nukula_mos)
print(molvania_mos)
print(hogwarts_mos)
print(flausenthurm_mos)
print(westerose_mos)
print(countrywide_mos)



### PART 3 ###
#Molvania t-test
print(sp.stats.ttest_ind(molvania_mos_array, mos_array))
#Turgistan t-test
print(sp.stats.ttest_ind(turgistan_mos_array, mos_array))

#Linear Regression
ami_array = np.array(df['AMI'])
mos_array = np.array(df['MOS'])
print(stats.linregress(ami_array, mos_array))




"""

# log
mos_nlog = np.log(mos_array)
mos_nlog[mos_nlog == -inf] = 0

shire_mos_nlog = np.log(shire_mos_array)
shire_mos_nlog[shire_mos_nlog == -inf] = 0

polombia_mos_nlog = np.log(polombia_mos_array)
polombia_mos_nlog[polombia_mos_nlog == -inf] = 0

sangala_mos_nlog = np.log(sangala_mos_array)
sangala_mos_nlog[sangala_mos_nlog == -inf] = 0

turgistan_mos_nlog = np.log(turgistan_mos_array)
turgistan_mos_nlog[turgistan_mos_nlog == -inf] = 0

urkesh_mos_nlog = np.log(urkesh_mos_array)
urkesh_mos_nlog[urkesh_mos_nlog == -inf] = 0

nukula_mos_nlog = np.log(nukula_mos_array)
nukula_mos_nlog[nukula_mos_nlog == -inf] = 0

molvania_mos_nlog = np.log(molvania_mos_array)
molvania_mos_nlog[molvania_mos_nlog == -inf] = 0

hogwarts_mos_nlog = np.log(hogwarts_mos_array)
hogwarts_mos_nlog[hogwarts_mos_nlog == -inf] = 0

flausenthurm_mos_nlog = np.log(flausenthurm_mos_array)
flausenthurm_mos_nlog[flausenthurm_mos_nlog == -inf] = 0

westerose_mos_nlog = np.log(westerose_mos_array)
westerose_mos_nlog[westerose_mos_nlog == -inf] = 0


#T-test (nLogs)
print("Country-wide MOS variance: " + str((np.var(mos_array))))
print("Country-wide MOS nLog variance: " + str((np.var(mos_nlog))))

print("Molvania MOS variance: " + str((np.var(molvania_mos_array))))
print("Molvania MOS nLog variance: " + str((np.var(molvania_mos_nlog))))

print(sp.stats.ttest_ind(shire_mos_nlog, mos_nlog, equal_var=True))
print(sp.stats.ttest_ind(polombia_mos_nlog, mos_nlog, equal_var=True))
print(sp.stats.ttest_ind(sangala_mos_nlog, mos_nlog, equal_var=True))
print(sp.stats.ttest_ind(turgistan_mos_nlog, mos_nlog, equal_var=True))
print(sp.stats.ttest_ind(urkesh_mos_nlog, mos_nlog, equal_var=True))
print(sp.stats.ttest_ind(nukula_mos_nlog, mos_nlog, equal_var=True))
print(sp.stats.ttest_ind(molvania_mos_nlog, mos_nlog, equal_var=False))
print(sp.stats.ttest_ind(hogwarts_mos_nlog, mos_nlog, equal_var=False))
print(sp.stats.ttest_ind(flausenthurm_mos_nlog, mos_nlog, equal_var=True))
print(sp.stats.ttest_ind(westerose_mos_nlog, mos_nlog, equal_var=True))

print(sp.stats.ttest_ind(polombia_mos_array, molvania_mos_array))

"""
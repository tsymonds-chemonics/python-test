#cd "C:\Python Test"
#python -m pip install --upgrade pip
#python Python-Test_Step_1_Theo.py
#python
import pandas as pd
import numpy as np

file = 'test_Country_X_Test_Data.xls'

shire_df = pd.read_excel(file, sheet_name="Shire", header=2)

#Data Coding for future reference
product_namlist = list(shire_df.columns[1:])
sohproduct_namlist = list(shire_df.iloc[[1],[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63]])
amiproduct_namlist = list(shire_df.iloc[[1],[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64]])
product_shortnames = ['26 Zidovudine/Lamivudine /Nevirapine 300/150/200mg', '24 Zidovudine/ Lamivudine 300/150mg', '20 Tenofovir/Lamivudine 300/300mg', '21 Tenofovir/Lamivudine/ Efavirenz 300/300/600mg', '538 Tenofovir/Lamivudine/Efavirenz(30) 300/300/400', '539 Tenofovir/Lamivudine/Efavirenz(90) 300/300/400', '15 Nevirapine 200mg', '07 Efavirenz 600mg', '10 Lamivudine 150mg', '494 Zidovudine 300mg', '04 Abacavir/Lamivudine 600/300mg', '03 Abacavir/Lamivudine 60/30mg', '27 Zidovudine/Lamivudine/Nevirapine 60/30/50mg', '25 Zidovudine/Lamivudine 60/30mg', '08 Efavirenz 200mg', '16 Nevirapine suspension 10mg/ml', '01 Abacavir 300mg', '05 Atazanavir 300mg', '13 Lopinavir/Ritonavir  200/50mg', '17 Raltegravir 400mg', '18 Ritonavir 100mg', '19 Tenofovir 300mg', '12 Lopinavir/Ritonavir 100/25mg', '14 Lopinavir/Ritonavir 80mg/20ml', '09 Etravirine 100mg', '06 Durunavir 600mg', '51 Condoms: Female condoms', '53 Condoms: Male condoms', '73 Determine HIV 1/2 kit', '74 SD Bioline', '75 UniGold', '48 Isoniazid 300mg']


def build_facility_df(region_df, index_add, facility, region) :
    soh_facility = np.array(region_df.iloc[[index_add],[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63]])
    ami_facility = np.array(region_df.iloc[[index_add],[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64]])
    #possible col index list: np.array(range(1,64,2)) and np.array(range(1,65,2))

    df_meds = pd.DataFrame(product_shortnames)
    df_soh_flip = pd.DataFrame(soh_facility)
    df_soh = df_soh_flip.transpose()
    df_ami_flip = pd.DataFrame(ami_facility)
    df_ami = df_ami_flip.transpose()

    df_meds['soh'] = df_soh
    df_meds['ami'] = df_ami
    df_meds['mos'] = ((df_soh)/(df_ami))
    df_medicine = df_meds

    df_medicine.insert(0, 'Facility', facility)
    df_medicine.insert(0, 'SNL1', region)
    df_medicine.insert(0, 'Period', '2018 Oct')
    df_medicine.insert(0, 'Country', 'X')
    df_region_facility = df_medicine
    #region_df.append(df_region_facility)
    #print below to debug
    #print(df_region_facility)
    return df_region_facility

df_col_list = ['Country', 'Period', 'SNL1', 'Facility']
df = pd.DataFrame(columns=df_col_list)
#print(df)

# list of regions in country
regions = ["Shire", "Polombia", "Sangala", "Turgistan", "Urkesh", "Nuku'la Atoll", "Molvanîa", "Hogwarts", "Flausenthurm", "Westerose"]

# repeat for each region
for region in regions:
    # create df for region
    region_df = pd.read_excel(file, sheet_name=region, header=2)
    region_df = region_df.drop([region_df.shape[0]-1])

    # get facilities in region
    facilities = region_df['Organisation unit'].tolist()
    #facilities = facilities[:-1]

    # call function to append to region_df
    index = 0
    for facility in facilities:
        df = df.append(build_facility_df(region_df, index, facility, region))
        index+=1

df.columns = ['Country', 'Period', 'SNL1', 'Facility', 'Product', 'SOH', 'AMI', 'MOS']
#print(df)
df.to_excel("Global Template.xlsx", index=False)
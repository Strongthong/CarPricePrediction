import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import pickle
import requests
import json

class DataPipeLine :

    def __init__(self, data_source):
        self.data_path = data_source
        self.data = None
        self.regressor_data = None
    

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def clean_data(self):
        warnings.filterwarnings("ignore")
        df_clean = self.data.drop(['LISTING_URL', 'SCRAPE_DATE'],axis=1)
        dropped_data=df_clean.dropna()
        df_clean=dropped_data
        
        self.data['SCRAPE_DATE'] = pd.to_datetime(self.data['SCRAPE_DATE'])
        df_clean['MANUFACTURED_YEAR'] = df_clean['MANUFACTURED_YEAR'].astype(int)
        df_clean['CAR_AGE'] = self.data['SCRAPE_DATE'].dt.year  - df_clean['MANUFACTURED_YEAR']
        
        df_clean['POST_DATE'] = pd.to_datetime(df_clean['POST_DATE'])
        df_clean['POST_AGE'] = (self.data['SCRAPE_DATE'] - df_clean['POST_DATE'])
        df_clean['POST_AGE']=df_clean['POST_AGE'].dt.days

        df_clean['TRANSMISSION_CONVERT'] = df_clean['TRANSMISSION'].apply(lambda x: 1 if x == 'Auto' else 0)
        df_clean.drop('TRANSMISSION',axis=1,inplace=True)
        df_clean.rename(columns={'TRANSMISSION_CONVERT':"TRANSMISSION"}, inplace=True) 

        veh_list=[]
        for veh in df_clean['VEHICLE_TYPE'].unique():
            veh_list.append(veh)

        veh_list.sort()
        out = map(lambda x:x.lower(), veh_list)
        veh_list = list(out) 

        df_clean['VEHICLE_TYPE']
        df_encoded = pd.get_dummies(df_clean['VEHICLE_TYPE'], prefix='VEHICLE_TYPE')

        # Concatenating the new columns to the original DataFrame
        df_encoded = df_encoded.astype(int)
        df_clean = pd.concat([df_clean, df_encoded], axis=1)

        df_clean2=df_clean
        df_clean2.loc[df_clean2['BRAND'] == 'Aston','BRAND'] = 'Aston Martin'
        df_clean2.loc[df_clean2['BRAND'] == 'Land','BRAND'] = 'Land Rover'
        df_clean2.loc[df_clean2['BRAND'] == 'Alfa', 'BRAND'] = 'Alfa Romeo'
        # Cleaning whitespaces from the values in "Brand" to prevent any messup later
        df_clean2['BRAND'].apply(str.strip)

        category_brands = {
            'EXOTIC': ['Koenigsegg','Bugatti','Ferrari', 'Lamborghini','Aston Martin','McLaren','Hummer'],
            'ULTRA_LUXURY': ['Porsche','Maserati','Rolls-Royce', 'Land Rover','Bentley','Maybach'],
            'LUXURY': ['MINI','Mini','Alfa Romeo','Mercedes','Mercedes-Benz', 'BMW', 'Audi', 'Lexus','Jeep','Lotus','Volvo','Peugeot','Tesla','BYD','Acura','Cadillac','Jaguar','Infiniti','Chrysler','Lincoln','Genesis'],
            'MID_LEVEL': ['Volkswagen','Renault','Ford', 'Chevrolet'],
            'ECONOMY': ['Toyota','Honda','Hyundai', 'Kia', 'Nissan', 'Mazda','Mitsubishi','Subaru','Suzuki','Citroen','Proton','Ssangyong','Daihatsu','Fiat','Skoda','Opel','MG','SEAT','Perodua'],
            'OTHERS': []  # An empty list for unspecified brands
        }

        # Reversing the categorization based on the 'Brand' column
        df_clean2['CAR_CATEGORY'] = df_clean2['BRAND'].apply(lambda x: next((category for category, brands in category_brands.items() if x in brands), 'OTHERS'))

        df_encoded = pd.get_dummies(df_clean2['CAR_CATEGORY'], prefix='CAR_CATEGORY')

        # Concatenating the new columns to the original DataFrame
        df_encoded = df_encoded.astype(int)
        df_clean2 = pd.concat([df_clean2, df_encoded], axis=1)

        df_clean_log=df_clean2.copy()
        df_clean_log["MILEAGE_KM"] = df_clean_log['MILEAGE_KM'].apply(np.log)

        df_clean_log["DEPRE_VALUE_PER_YEAR"] = df_clean_log['DEPRE_VALUE_PER_YEAR'].apply(np.log)
        df_clean_log["ROAD_TAX_PER_YEAR"] = df_clean_log['ROAD_TAX_PER_YEAR'].apply(np.log)
        df_clean_log["DEREG_VALUE_FROM_SCRAPE_DATE"] = df_clean_log['DEREG_VALUE_FROM_SCRAPE_DATE'].apply(np.log)
        df_clean_log["OMV"] = df_clean_log['OMV'].apply(np.log)
        df_clean_log["ARF"] = df_clean_log['ARF'].apply(np.log)
        df_clean_log["COE_FROM_SCRAPE_DATE"] = df_clean_log['COE_FROM_SCRAPE_DATE'].apply(np.log)
        #df_clean_log["DAYS_OF_COE_LEFT"] = df_clean_log['DAYS_OF_COE_LEFT'].apply(np.log)
        df_clean_log["ENGINE_CAPACITY_CC"] = df_clean_log['ENGINE_CAPACITY_CC'].apply(np.log)
        df_clean_log["CURB_WEIGHT_KG"] = df_clean_log['CURB_WEIGHT_KG'].apply(np.log)

        df_clean_log.drop('ARF', axis=1, inplace=True)
        df_clean_log.drop('ROAD_TAX_PER_YEAR', axis=1, inplace=True)

        #Dropping columns not needed in our regression as they have been encoded 
        df_clean_log.drop('BRAND', axis=1, inplace=True)
        df_clean_log.drop('REG_DATE', axis=1, inplace=True)
        df_clean_log.drop('MANUFACTURED_YEAR', axis=1, inplace=True)
        df_clean_log.drop('VEHICLE_TYPE', axis=1, inplace=True)
        df_clean_log.drop('POST_DATE', axis=1, inplace=True)
        df_clean_log.drop('CAR_CATEGORY', axis=1, inplace=True)
        df_clean_log.drop(columns=df_clean_log.columns[0], axis=1,  inplace=True)
        self.data= df_clean_log
        

    def regression(self):
        warnings.filterwarnings("ignore")
        x_var=self.data.drop('PRICE', axis=1)
        y= self.data['PRICE']

        test_size = 0.20
        seed = 7

        X_train, X_test, y_train, y_test = train_test_split(x_var, y, test_size=test_size, random_state=seed)
        
        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        regressor = ExtraTreesRegressor(n_estimators=400)
        regressor.fit(rescaledX, y_train)
        # transform the validation dataset
        rescaledValidationX = scaler.transform(X_test)
        predictions = regressor.predict(rescaledValidationX)
        self.regressor_data=round(r2_score(y_test,predictions), 3)

        pickle.dump(regressor, open('model.pkl','wb'))
        pickle.dump(scaler, open('scaler.pkl','wb'))

        model = pickle.load(open('model.pkl','rb'))
        scaler= pickle.load(open('scaler.pkl','rb'))

    def run_pipeline(self):
        self.load_data()
        self.clean_data()
        self.regression()
        return self.regressor_data
      


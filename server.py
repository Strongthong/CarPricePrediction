# Import libraries
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
@app.route('/api',methods=['POST'])
def predict():

    # Get the data from the POST request.
    data = request.get_json(force=True)
    
    if data['DEPRE_VALUE_PER_YEAR']:
        depre_value_per_year=np.log(float(data['DEPRE_VALUE_PER_YEAR']))
    else:
        depre_value_per_year=0

    if data['DEREG_VALUE_FROM_SCRAPE_DATE']:
        dereg_value=np.log(float(data['DEREG_VALUE_FROM_SCRAPE_DATE']))
    else:
        dereg_value=0

    if data['OMV']:
        omv=np.log(float(data['OMV']))
    else:
        omv=0

    if data['COE_FROM_SCRAPE_DATE']:
        coe=np.log(float(data['COE_FROM_SCRAPE_DATE']))
    else:
        coe=0

    if data['DAYS_OF_COE_LEFT']:
        days_of_coe_left=data['DAYS_OF_COE_LEFT']
    else:
        days_of_coe_left=0

    if data['ENGINE_CAPACITY_CC']:
        engine_capacity=np.log(float(data['ENGINE_CAPACITY_CC']))
    else:
        engine_capacity=0

    if data['CURB_WEIGHT_KG']:
        curb_weight=np.log(float(data['CURB_WEIGHT_KG']))
    else:
        curb_weight=0

    if data['NO_OF_OWNERS']:
        no_of_owners=data['NO_OF_OWNERS']
    else:
        no_of_owners=0   

    if data['MILEAGE_KM']:
        mileage_km=np.log(float(data['MILEAGE_KM']))
    else:
        mileage_km=0
    
    if data['CAR_AGE']:
        car_age=data['CAR_AGE']
    else:
        car_age=0

    # if data['POST_AGE']:
    #     post_age=data['POST_AGE']
    # else:
    #     post_age=0
    
    if data['TRANSMISSION']:
        transmission=data['TRANSMISSION'].lower()

        if transmission=='auto':
            transmission=1
        else:
            transmission=0
    else:
        transmission=0

    veh_list=['hatchback','luxury sedan', 'mpv', 'mid-sized sedan', 'others','suv','sports car','stationwagon']
    veh_type_list=[]
    if data['VEHICLE_TYPE']:
        veh_type=data['VEHICLE_TYPE'].lower()
        for veh in veh_list:
            if veh_type==veh:
                veh_type_list.append(1)
            else:
                veh_type_list.append(0)
        
    
    #if there are missing values in vehicle type, will use this array
    else:
        veh_type_list=[0,0,0,0,1,0,0,0]  


    category_brands = {
        'ECONOMY': ['Toyota', 'Honda', 'Hyundai', 'Kia', 'Nissan', 'Mazda', 'Mitsubishi', 'Subaru', 'Suzuki', 'Citroen', 'Proton', 'Ssangyong', 'Daihatsu', 'Fiat', 'Skoda', 'Opel', 'MG', 'SEAT', 'Perodua'],
        'EXOTIC': ['Koenigsegg', 'Bugatti', 'Ferrari', 'Lamborghini', 'Aston Martin', 'McLaren', 'Hummer'],
        'LUXURY': ['MINI', 'Mini', 'Alfa Romeo', 'Mercedes', 'Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Jeep', 'Lotus', 'Volvo', 'Peugeot', 'Tesla', 'BYD', 'Acura', 'Cadillac', 'Jaguar', 'Infiniti', 'Chrysler', 'Lincoln', 'Genesis'],
        'MID_LEVEL': ['Volkswagen', 'Renault', 'Ford', 'Chevrolet'],
        #'OTHERS': [],  # An empty list for unspecified brands
        'ULTRA_LUXURY': ['Porsche', 'Maserati', 'Rolls-Royce', 'Land Rover', 'Bentley', 'Maybach']
        }

    cat_type_list=[]
    if data['BRAND']:
        check=0
        brand=data['BRAND'].lower()
        for category, brands in category_brands.items():
            if brand in brands:
                cat_type_list.append(1)
                check+=1
            else:
                cat_type_list.append(0)

        if check==0:
            cat_type_list.insert(4,1)
        else:
            cat_type_list.insert(4,0)  

    else:
        cat_type_list=[0,0,0,0,1,0]

    final_array=[]
    final_array.append(depre_value_per_year)
    final_array.append(dereg_value)
    final_array.append(omv)
    final_array.append(coe)
    final_array.append(days_of_coe_left)
    final_array.append(engine_capacity)
    final_array.append(curb_weight)
    final_array.append(no_of_owners)
    final_array.append(mileage_km)
    final_array.append(car_age)
    # final_array.append(post_age)
    final_array.append(transmission)
    final_array.extend(veh_type_list)
    final_array.extend(cat_type_list)

    features = np.array([final_array])
    rescaled=scaler.transform(features)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(rescaled)
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=5000, debug=True)
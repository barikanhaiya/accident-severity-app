import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -------------------------------
# Load trained model pipeline
# -------------------------------
pipe = pickle.load(open("pipe.pkl", "rb"))

st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚧", layout="centered")

st.title("🚧 Road Accident Severity Predictor")
st.write("Fill in the accident details below to predict the severity of injury.")

st.markdown("---")

# -------------------------------
# Input fields
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    Day_of_week = st.selectbox("Day of Week", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
    Age_band_of_driver = st.selectbox("Age Band of Driver", ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown'])
    Sex_of_driver = st.selectbox("Sex of Driver", ['Male', 'Female', 'Unknown'])
    Educational_level = st.selectbox("Educational Level", ['Illiterate', 'Writing & reading', 'Elementary school',
                                                             'Junior high school', 'High school', 'Above high school', 'Unknown'])
    Vehicle_driver_relation = st.selectbox("Vehicle-Driver Relation", ['Owner', 'Employee', 'Other', 'Unknown'])
    Driving_experience = st.selectbox("Driving Experience", ['Below 1yr', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'No Licence', 'unknown'])
    Type_of_vehicle = st.selectbox("Type of Vehicle", ['Automobile', 'Public (13–45 seats)', 'Lorry (41–100Q)',
                                                          'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen',
                                                          'Lorry (11–40Q)', 'Public (> 45 seats)', 'Long lorry', 'Motorcycle',
                                                          'Special vehicle', 'Ridden horse', 'Bajaj', 'Turbo', 'Bicycle', 'Other'])
    Owner_of_vehicle = st.selectbox("Owner of Vehicle", ['Owner', 'Governmental', 'Organization', 'Other'])
    Service_year_of_vehicle = st.selectbox("Service Year of Vehicle", ['Below 1yr', '1-2yr', '2-5yrs', '5-10yrs', 'Above 10yr', 'Unknown'])
    Defect_of_vehicle = st.selectbox("Defect of Vehicle", ['No defect', '5', '7'])
    Area_accident_occured = st.selectbox("Area Accident Occurred", ['Residential areas', 'Office areas', 'Recreational areas',
                                                                       'Industrial areas', 'Church areas', 'Market areas',
                                                                       'Rural village areas', 'Outside rural areas', 'Hospital areas',
                                                                       'School areas', 'Other', 'Unknown'])
    Lanes_or_Medians = st.selectbox("Lanes or Medians", ['Undivided Two way', 'One way',
                                                            'Two-way (divided with broken lines road marking)',
                                                            'Two-way (divided with solid lines road marking)',
                                                            'Double carriageway (median)', 'other', 'Unknown'])
    Road_allignment = st.selectbox("Road Alignment", ['Tangent road with flat terrain', 'Tangent road with mild grade and flat terrain',
                                                         'Tangent road with mountainous terrain and', 'Tangent road with rolling terrain',
                                                         'Gentle horizontal curve', 'Sharp reverse curve', 'Escarpments',
                                                         'Steep grade upward with mountainous terrain', 'Steep grade downward with mountainous terrain'])

with col2:
    Types_of_Junction = st.selectbox("Types of Junction", ['No junction', 'T Shape', 'Y Shape', 'Crossing', 'O Shape', 'X Shape', 'Other', 'Unknown'])
    Road_surface_type = st.selectbox("Road Surface Type", ['Asphalt roads', 'Asphalt roads with some distress', 'Earth roads', 'Gravel roads', 'Other'])
    Road_surface_conditions = st.selectbox("Road Surface Conditions", ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'])
    Light_conditions = st.selectbox("Light Conditions", ['Daylight', 'Darkness - lights lit', 'Darkness - lights unlit', 'Darkness - no lighting'])
    Weather_conditions = st.selectbox("Weather Conditions", ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Windy', 'Snow', 'Fog or mist', 'Other', 'Unknown'])
    Type_of_collision = st.selectbox("Type of Collision", ['Vehicle with vehicle collision', 'Collision with roadside objects',
                                                              'Collision with pedestrians', 'Collision with animals', 'Rollover',
                                                              'Fall from vehicles', 'With Train', 'Collision with roadside-parked vehicles',
                                                              'Other', 'Unknown'])
    Number_of_vehicles_involved = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=2)
    Number_of_casualties = st.number_input("Number of Casualties", min_value=1, max_value=10, value=1)
    Vehicle_movement = st.selectbox("Vehicle Movement", ['Going straight', 'Turnover', 'U-Turn', 'Reversing', 'Waiting to go',
                                                            'Getting off', 'Moving Backward', 'Entering a junction', 'Parked',
                                                            'Stopping', 'Overtaking', 'Other', 'Unknown'])
    Casualty_class = st.selectbox("Casualty Class", ['Driver or rider', 'Pedestrian', 'Passenger', 'na'])
    Sex_of_casualty = st.selectbox("Sex of Casualty", ['Male', 'Female', 'na'])
    Age_band_of_casualty = st.selectbox("Age Band of Casualty", ['Under 18', '18-30', '31-50', 'Over 51', 'na'])
    Casualty_severity = st.selectbox("Casualty Severity", [1, 2, 3, 'na'])

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    Work_of_casuality = st.selectbox("Work of Casualty", ['Driver', 'Employee', 'Self-employed', 'Student', 'Unemployed', 'Other', 'Unknown'])
    Fitness_of_casuality = st.selectbox("Fitness of Casualty", ['Normal', 'Deaf', 'Blind', 'Other', 'Unknown'])

with col4:
    Pedestrian_movement = st.selectbox("Pedestrian Movement", ['Not a Pedestrian', "Crossing from driver's nearside",
                                                                  "Crossing from nearside - masked by parked vehicle",
                                                                  'Crossing from offside', 'In carriageway, stationary',
                                                                  'Walking along in carriageway, back to traffic',
                                                                  'Walking along in carriageway, facing traffic', 'Other', 'Unknown'])
    Cause_of_accident = st.selectbox("Cause of Accident", ['No distancing', 'Changing lane to the left', 'Changing lane to the right',
                                                              'Overtaking', 'Overspeed', 'Overloading', 'Improper parking',
                                                              'Driving carelessly', 'Driving at high speed', 'Driving under the influence of drugs',
                                                              'Drunk driving', 'Getting off the vehicle improperly', 'Turnover',
                                                              'Other', 'Unknown'])

Hour_of_Day = st.slider("Hour of Day", min_value=0, max_value=23, value=12)

st.markdown("---")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Severity"):
    features = pd.DataFrame([[Day_of_week, Age_band_of_driver, Sex_of_driver, Educational_level, Vehicle_driver_relation,
                               Driving_experience, Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle,
                               Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians, Road_allignment,
                               Types_of_Junction, Road_surface_type, Road_surface_conditions, Light_conditions,
                               Weather_conditions, Type_of_collision, Number_of_vehicles_involved,
                               Number_of_casualties, Vehicle_movement, Casualty_class, Sex_of_casualty,
                               Age_band_of_casualty, Casualty_severity, Work_of_casuality, Fitness_of_casuality,
                               Pedestrian_movement, Cause_of_accident, Hour_of_Day]])

    result = pipe.predict(features)[0]

    if result == 2:
        st.success("🟢 Predicted Severity: **Slight Injury**")
    elif result == 1:
        st.warning("🟠 Predicted Severity: **Serious Injury**")
    else:
        st.error("🔴 Predicted Severity: **Fatal Injury**")

st.markdown("---")
st.caption("Built with Streamlit | Random Forest Model | Road Traffic Accident Severity Prediction")

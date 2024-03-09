from flask import Flask, request, render_template,send_file
import mammoth
import joblib
import numpy as np
import pandas as pd
app = Flask(__name__)


model1 = joblib.load('final_svc.pkl')
# model2 = joblib.load('final_rf.pkl')
model_2 = joblib.load('final_svc.pkl')
model3 = joblib.load('final_nn.pkl')
model4 = joblib.load('final_log.pkl')
model5 = joblib.load('final_nb.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graphs')
def view_graphs():
    return render_template('graphs.html')
@app.route('/feedback')
def view_feedback():
    return render_template('feedback.html')
@app.route('/report')
def view_report():
    with open('static/Analysis_Report.docx', 'rb') as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html = result.value
    return render_template('report.html', html=html)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # time = int(request.form['time'])
        road_class = request.form['road_class']
        district = request.form['district']
        loccoord = request.form['loccoord']
        traffctl = request.form['traffctl']
        visibility = request.form['visibility']
        light = request.form['light']
        rdsfcond = request.form['rdsfcond']
        invage = request.form['invage']
        injury = request.form['injury']
        drivact = request.form['drivact']
        drivcond = request.form['drivcond']
        pedestrian = int(request.form['pedestrian'])
        cyclist = int(request.form['cyclist'])
        automobile = int(request.form['automobile'])
        motorcycle = int(request.form['motorcycle'])
        truck = int(request.form['truck'])
        trsn_city_veh = int(request.form['trsn_city_veh'])
        emerg_veh = int(request.form['emerg_veh'])
        passenger = int(request.form['passenger'])
        speeding = int(request.form['speeding'])
        ag_driv = int(request.form['ag_driv'])
        redlight = int(request.form['redlight'])
        alcohol = int(request.form['alcohol'])
        disability = int(request.form['disability'])
        data = {'TIME': 0.124521,
                road_class: 1,
                district: 1,
                loccoord: 1,
                traffctl: 1,
                visibility: 1,
                light: 1,
                rdsfcond: 1,
                invage: 1,
                injury: 1,
                drivact: 1,
                drivcond: 1,
               'PEDESTRIAN': pedestrian,
                'CYCLIST': cyclist,
                'AUTOMOBILE': automobile,
                'MOTORCYCLE': motorcycle,
                'TRUCK': truck,
                'TRSN_CITY_VEH': trsn_city_veh,
                'EMERG_VEH': emerg_veh,
                'PASSENGER': passenger,
                'SPEEDING': speeding,
                'AG_DRIV': ag_driv,
                'REDLIGHT': redlight,
                'ALCOHOL': alcohol,
                'DISABILITY':disability}
        data_cleaned = clean(data)
        df = pd.DataFrame([data_cleaned])
        model = model_selected()
        prediction = result(df)
        prediction2 = result2(df,model)
        return render_template('result.html',  input = data_cleaned, id = df, result_prediction = prediction, model = model,model_prediction = prediction2)
            
def clean(data):
    col_mapping ={'TIME': 0,
            'ROAD_CLASS_Arterial': 0,
            'ROAD_CLASS_Collector': 0,
            'ROAD_CLASS_Expressway': 0,
            'ROAD_CLASS_Local': 0,
            'ROAD_CLASS_other': 0,
            'DISTRICT_Etobicoke York': 0,
            'DISTRICT_North York': 0,
            'DISTRICT_Scarborough': 0,
            'DISTRICT_Toronto and East York': 0,
            'LOCCOORD_at Intersection': 0,
            'LOCCOORD_not at intersection': 0,
            'TRAFFCTL_control': 0,
            'TRAFFCTL_no control': 0,
            'VISIBILITY_Clear': 0,
            'VISIBILITY_Drifting Snow': 0,
            'VISIBILITY_Fog, Mist, Smoke, Dust': 0,
            'VISIBILITY_Freezing Rain': 0,
            'VISIBILITY_Other': 0,
            'VISIBILITY_Rain': 0,
            'VISIBILITY_Snow': 0,
            'VISIBILITY_Strong wind': 0,
            'LIGHT_Dark': 0,
            'LIGHT_Dark, artificial': 0,
            'LIGHT_Dawn': 0,
            'LIGHT_Dawn, artificial': 0,
            'LIGHT_Daylight': 0,
            'LIGHT_Daylight, artificial': 0,
            'LIGHT_Dusk': 0,
            'LIGHT_Dusk, artificial': 0,
            'LIGHT_Other': 0,
            'RDSFCOND_Dry': 0,
            'RDSFCOND_Ice': 0,
            'RDSFCOND_Loose Sand or Gravel': 0,
            'RDSFCOND_Loose Snow': 0,
            'RDSFCOND_Other': 0,
            'RDSFCOND_Packed Snow': 0,
            'RDSFCOND_Slush': 0,
            'RDSFCOND_Spilled liquid': 0,
            'RDSFCOND_Wet': 0,
            'INVAGE_0 to 4': 0,
            'INVAGE_10 to 14': 0,
            'INVAGE_15 to 19': 0,
            'INVAGE_20 to 24': 0,
            'INVAGE_25 to 29': 0,
            'INVAGE_30 to 34': 0,
            'INVAGE_35 to 39': 0,
            'INVAGE_40 to 44': 0,
            'INVAGE_45 to 49': 0,
            'INVAGE_5 to 9': 0,
            'INVAGE_50 to 54': 0,
            'INVAGE_55 to 59': 0,
            'INVAGE_60 to 64': 0,
            'INVAGE_65 to 69': 0,
            'INVAGE_70 to 74': 0,
            'INVAGE_75 to 79': 0,
            'INVAGE_80 to 84': 0,
            'INVAGE_85 to 89': 0,
            'INVAGE_90 to 94': 0,
            'INVAGE_Over 95': 0,
            'INVAGE_unknown': 0,
            'INJURY_Fatal': 0,
            'INJURY_Major': 0,
            'INJURY_Minimal': 0,
            'INJURY_Minor': 0,
            'INJURY_None': 0,
            'DRIVACT_Normal': 0,
            'DRIVACT_Not Normal': 0,
            'DRIVCOND_Normal': 0,
            'DRIVCOND_Not Normal': 0,
            'PEDESTRIAN': 0,
            'CYCLIST': 0,
            'AUTOMOBILE': 0,
            'MOTORCYCLE': 0,
            'TRUCK': 0,
            'TRSN_CITY_VEH': 0,
            'EMERG_VEH': 0,
            'PASSENGER': 0,
            'SPEEDING': 0,
            'AG_DRIV': 0,
            'REDLIGHT': 0,
            'ALCOHOL': 0,
            'DISABILITY': 0
        }
    dict3 = {k: data[k] if k in data else col_mapping[k] for k in col_mapping}
    return dict3

def result(data):
    prediction = model1.predict(data)
    return prediction

def model_selected():
    if request.method == 'POST':
        m = request.form['model123']
    return m

def result2(data,model):
    if(model == "SUPPORT VECTOR CLASSIFIER"):
        prediction = model1.predict(data)
    elif(model == "RANDOM FOREST"):
        prediction = model2.predict(data)
    elif(model == "NEURAL NETWORK"):
        prediction = model3.predict(data)
    elif(model == "LOGISTIC REGRESSION"):
        prediction = model4.predict(data)
    elif(model == "NAIVE BAYES"):
        prediction = model5.predict(data)
    return prediction

if __name__ == '__main__':
    app.run(debug=True, port = 5000)


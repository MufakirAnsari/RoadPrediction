from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the best model and scaler
model = joblib.load('gradient_boosting_model_best.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'PCR_Year_-1': [float(request.form['PCR_Year_-1'])],
        'Y-1_StrDeduct': [float(request.form['Y-1_StrDeduct'])],
        'Y-1_OtherDeduct': [float(request.form['Y-1_OtherDeduct'])],
        'Avg. Treatment Thickness': [float(request.form['Avg_Treatment_Thickness'])],
        'ADT_TOTAL_NBR': [float(request.form['ADT_TOTAL_NBR'])],
        'ADT_TRUCK_NBR': [float(request.form['ADT_TRUCK_NBR'])],
        'ADT_PSNGR_CAR_NBR': [float(request.form['ADT_PSNGR_CAR_NBR'])],
        'Avg. Design AC Cont Pct (%)': [float(request.form['Avg_Design_AC_Cont_Pct'])],
        'Avg. Rap AC': [float(request.form['Avg_Rap_AC'])],
        'Avg. Virgin AC': [float(request.form['Avg_Virgin_AC'])],
        'Avg. Stability': [float(request.form['Avg_Stability'])],
        'Avg. Flow': [float(request.form['Avg_Flow'])],
        'Avg. VMA': [float(request.form['Avg_VMA'])],
        'Avg. Air Void Pct (%)': [float(request.form['Avg_Air_Void_Pct'])],
        'Soundness Loss': [float(request.form['Soundness_Loss'])],
        'MicroDeval Loss': [float(request.form['MicroDeval_Loss'])],
        'ABSORPTION_PCT': [float(request.form['ABSORPTION_PCT'])],
        'SSD_SPEC_GRAV': [float(request.form['SSD_SPEC_GRAV'])],
        'LA Abrasion': [float(request.form['LA_Abrasion'])],
        'SHALE_A_PCT':        [float(request.form['SHALE_A_PCT'])],
        'PCRP': [float(request.form['PCRP'])],
        'SNOW': [float(request.form['SNOW'])],
        'TAVG': [float(request.form['TAVG'])]
    }

    # Convert form data to DataFrame
    df = pd.DataFrame(data)
    
    # Preprocess the data
    df['ADT_TRUCK_NBR'] = (df['ADT_TRUCK_NBR'] > 150).astype(int)
    if 'SHALE_A_PCT' not in df.columns:
        df['SHALE_A_PCT'] = 0
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(df)
    data_imputed_df = pd.DataFrame(data_imputed, columns=df.columns)
    
    # Standardize the data
    data_scaled = scaler.transform(data_imputed_df)
    
    # Make predictions
    prediction = model.predict(data_scaled)
    
    # Return the prediction as a JSON response
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)



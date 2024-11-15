from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow
from fastapi import Request

app = FastAPI()

MLFLOW_TRACKING_URI = "http://24.144.69.175:5000"
RUN_ID = "cf6fecc5c83b4447809e1875a1669ff3"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    try:
        model_uri = f"runs:/{RUN_ID}/model"
        
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load model")

model = load_model()


feature_order = [
    "_RMSenergy_Mean", "_Lowenergy_Mean", "_Fluctuation_Mean", "_Tempo_Mean",
    "_MFCC_Mean_1", "_MFCC_Mean_2", "_MFCC_Mean_3", "_MFCC_Mean_4", "_MFCC_Mean_5",
    "_MFCC_Mean_6", "_MFCC_Mean_7", "_MFCC_Mean_8", "_MFCC_Mean_9", "_MFCC_Mean_10",
    "_MFCC_Mean_11", "_MFCC_Mean_12", "_MFCC_Mean_13", "_Roughness_Mean",
    "_Roughness_Slope", "_Zero-crossingrate_Mean", "_AttackTime_Mean",
    "_AttackTime_Slope", "_Rolloff_Mean", "_Eventdensity_Mean", "_Pulseclarity_Mean",
    "_Brightness_Mean", "_Spectralcentroid_Mean", "_Spectralspread_Mean",
    "_Spectralskewness_Mean", "_Spectralkurtosis_Mean", "_Spectralflatness_Mean",
    "_EntropyofSpectrum_Mean", "_Chromagram_Mean_1", "_Chromagram_Mean_2",
    "_Chromagram_Mean_3", "_Chromagram_Mean_4", "_Chromagram_Mean_5",
    "_Chromagram_Mean_6", "_Chromagram_Mean_7", "_Chromagram_Mean_8",
    "_Chromagram_Mean_9", "_Chromagram_Mean_10", "_Chromagram_Mean_11",
    "_Chromagram_Mean_12", "_HarmonicChangeDetectionFunction_Mean",
    "_HarmonicChangeDetectionFunction_Std", "_HarmonicChangeDetectionFunction_Slope",
    "_HarmonicChangeDetectionFunction_PeriodFreq",
    "_HarmonicChangeDetectionFunction_PeriodAmp",
    "_HarmonicChangeDetectionFunction_PeriodEntropy"
]

@app.post("/predict")
async def predict(request: Request):
    try:
        input_data = await request.json()
        missing_features = [f for f in feature_order if f not in input_data]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Columnas faltantes: {', '.join(missing_features)}")
        
        data = pd.DataFrame([input_data])
        data = data[feature_order]
        
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
    
@app.get("/health")
async def health():
    return {"status": "ok"}

# Pasos para correr API de Inferencia

### 1. Generar la imagen de docker y correrla
```bash
cd app
docker build -t inference_api .
docker run -d -p 8000:8000 inference_api
```
### 2. Petición al endpoint

Ejecutar una petición HTTP con estas características:

- Tipo de petición: POST
- URL: http://localhost:8000/predict
- Content-Type: application/json
- Body: raw/JSON

#### Ejemplo de cuerpo de petición:
```
{
  "_RMSenergy_Mean": -1.285627868272798,
  "_Lowenergy_Mean": 0.7377750510941161,
  "_Fluctuation_Mean": 0.8738742211739952,
  "_Tempo_Mean": 0.18603967045781147,
  "_MFCC_Mean_1": 1.9299149804355153,
  "_MFCC_Mean_2": 0.5419099423413448,
  "_MFCC_Mean_3": 1.3558202806027653,
  "_MFCC_Mean_4": 0.17254478047571503,
  "_MFCC_Mean_5": 0.2159256684774218,
  "_MFCC_Mean_6": 0.3916115213138109,
  "_MFCC_Mean_7": -1.1670055214261204,
  "_MFCC_Mean_8": -1.0575262314191676,
  "_MFCC_Mean_9": 0.666435743877914,
  "_MFCC_Mean_10": 0.8300708423647596,
  "_MFCC_Mean_11": 1.803154230547734,
  "_MFCC_Mean_12": 1.6774663444832651,
  "_MFCC_Mean_13": 1.6644918551890546,
  "_Roughness_Mean": -0.9146552785839978,
  "_Roughness_Slope": 1.4531108377728108,
  "_Zero-crossingrate_Mean": -1.1333054849155422,
  "_AttackTime_Mean": -0.1969576683691288,
  "_AttackTime_Slope": -0.07419875430629524,
  "_Rolloff_Mean": -1.6792620521988668,
  "_Eventdensity_Mean": -1.0932599012212199,
  "_Pulseclarity_Mean": -1.0789406159438433,
  "_Brightness_Mean": -1.9882139169480615,
  "_Spectralcentroid_Mean": -1.6926381637720591,
  "_Spectralspread_Mean": -1.4504020884624953,
  "_Spectralskewness_Mean": 1.9885302064395152,
  "_Spectralkurtosis_Mean": 1.2412659017063752,
  "_Spectralflatness_Mean": -0.7000499466394786,
  "_EntropyofSpectrum_Mean": -1.7898850716482193,
  "_Chromagram_Mean_1": 0.4445453720908125,
  "_Chromagram_Mean_2": -0.8806293475392796,
  "_Chromagram_Mean_3": -0.9812850577698573,
  "_Chromagram_Mean_4": -0.7038732673680674,
  "_Chromagram_Mean_5": -0.1201170924056609,
  "_Chromagram_Mean_6": -0.7726991750496839,
  "_Chromagram_Mean_7": -0.7943255064467435,
  "_Chromagram_Mean_8": 1.6922119585643225,
  "_Chromagram_Mean_9": 0.21331921274233626,
  "_Chromagram_Mean_10": 1.1440180201683037,
  "_Chromagram_Mean_11": -1.0600054730237471,
  "_Chromagram_Mean_12": -0.8186233324925698,
  "_HarmonicChangeDetectionFunction_Mean": -0.2202413192817325,
  "_HarmonicChangeDetectionFunction_Std": 1.4458369238026045,
  "_HarmonicChangeDetectionFunction_Slope": 0.17357053145027734,
  "_HarmonicChangeDetectionFunction_PeriodFreq": -0.7827124788615227,
  "_HarmonicChangeDetectionFunction_PeriodAmp": -2.4534495294369516,
  "_HarmonicChangeDetectionFunction_PeriodEntropy": 0.8568684680257574
}
```

#### Ejemplo de respuesta:
```
{
    "prediction": [
        2
    ]
}
```
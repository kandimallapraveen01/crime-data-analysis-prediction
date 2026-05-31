### CRIME DATA FORECASTING & HOTSPOT DETECTION

A machine learning application to forecast crime trends in india and visualize the predections.
This uses Recursive Support Vector Regression (SVR) to predict future trends of years 2024-2030.

### Project Structure

```text
crime_forecasting/
├── app/
│   └── visualization.py    
├── data/
│   ├── raw/                  
│   └── processed/            
├── json/
│   ├── india_state.geojson    
│   └── cleaned.geojson        
├── models/                    
├── notebooks/
│   ├── processing.ipynb       
│   └── forecast_model.ipynb   
├── dsi.py                   
├── requirements.txt           
└── run_project.ps1            
```


### Running Project (Automated Script)

1. Open **PowerShell** in the project root folder.
2. Run the auto-setup script:
   ```powershell
   .\run_project.ps1

### Manual Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


2. **Download & Clean Data:**
```bash
python dsi.py

```


3. **Generate Data & Models:**
* Run `notebooks/processing.ipynb` (Creates processed CSV)
* Run `notebooks/forecast_model.ipynb` (Trains SVR models)


4. **Launch Dashboard:**
```bash
cd app
streamlit run visualization.py

```
#### Usable Links

1. **DataSet**
    * NCRB Statistics

2. **Indian GeoJSON Maps:**
    * [GeoJson - 1](https://github.com/geohacker/india/blob/master/state/india_state.geojson)
    * [GeoJson - 2](https://github.com/udit-001/india-maps-data/blob/main/geojson/india.geojson)    

### Note on Accuracy

The model was trained on data from years **2020-2023** uses **Lag Features** (history of previous months) to predict the future. Forecasts beyond 2-3 years (2026+) rely purely on the model's own previous guesses ("recursive estimation") and carry higher uncertainty. The app will display a warning when predicting these long-term timelines.

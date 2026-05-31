
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   STARTING PROJECT SETUP" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# INSTALL REQUIREMENTS
Write-Host "`n[STEP 1] Installing requirements from requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
    Write-Host "Requirements installed." -ForegroundColor Green
} else {
    Write-Host "requirements.txt not found! Skipping install..." -ForegroundColor Red
}

# CHECKING FOR FILES
Write-Host "`n[STEP 2] Checking Raw Data and GeoJSON..." -ForegroundColor Yellow

$rawDataPath = "data\raw\crime_dataset_india.csv"
$geojsonPath = "json\cleaned.geojson"
$dsiScript = "dsi.py"

if (-not (Test-Path $rawDataPath) -or -not (Test-Path $geojsonPath)) {
    Write-Host "Missing Raw Data or Cleaned GeoJSON." -ForegroundColor Magenta
    Write-Host "Running dsi.py to download and process..." -ForegroundColor Magenta
    
    if (Test-Path $dsiScript) {
        python $dsiScript
        Write-Host "dsi.py execution finished." -ForegroundColor Green
    } else {
        Write-Error "dsi.py is missing in the root folder! Cannot download dataset."
    }
} else {
    Write-Host "Raw Data and GeoJSON present." -ForegroundColor Green
}

# CHECKIGN FOR PROCESSED DATA
Write-Host "`n[STEP 3] Checking Processed Data..." -ForegroundColor Yellow
$processedData = "data\processed\crime_data_processed.csv"

if (-not (Test-Path $processedData)) {
    Write-Host "Processed data is missing ($processedData)." -ForegroundColor Red
    Write-Host "ACTION REQUIRED: Please open and run 'notebooks/processing.ipynb' to generate this file." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit
} else {
    Write-Host "Processed data found." -ForegroundColor Green
}

# CHECKING MODELS
Write-Host "`n[STEP 4] Checking Models..." -ForegroundColor Yellow
$modelFile = "models\svr_model.pkl"

if (-not (Test-Path $modelFile)) {
    Write-Host "Trained models are missing ($modelFile)." -ForegroundColor Red
    Write-Host "ACTION REQUIRED: Please open and run 'notebooks/forecast_model.ipynb' to train the models." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit
} else {
    Write-Host "Models found." -ForegroundColor Green
}

# RUN STREAMLIT APP
Write-Host "`n[STEP 5] Launching Application..." -ForegroundColor Cyan

if (Test-Path "app\visualization.py") {
    Push-Location "app" 
    
    try {
        Write-Host "Starting Streamlit via Python..." -ForegroundColor Green
        python -m streamlit run visualization.py
    } catch {
        Write-Error "Streamlit crashed. See error details above."
    } finally {
        Pop-Location 
        Write-Host "Returned to project root." -ForegroundColor Gray
    }
} else {
    Write-Error "The 'app' directory does not exist. Cannot launch visualization.py."
}
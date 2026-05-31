import os
import shutil
import kagglehub
import geopandas as gpd
from pathlib import Path

project_root = Path(os.getcwd()).resolve()

# Destination paths
raw_data_path = project_root / "data" / "raw"
json_folder = project_root / "json"
input_geojson = json_folder / "india.json"
output_geojson = json_folder / "cleaned.geojson"

os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)

print(f"Project Root detected at: {project_root}")

print("\n--- Starting Dataset Download ---")
try:
    # Download latest version
    path = kagglehub.dataset_download("sudhanvahg/indian-crimes-dataset")
    print(f"Kaggle download path: {path}")

    # Copy dataset contents into project folder
    print(f"Copying files to {raw_data_path}...")
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(raw_data_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print("Dataset copied successfully.")

except Exception as e:
    print(f"Error downloading/copying dataset: {e}")

print("\n--- Starting GeoJSON Processing ---")

if not input_geojson.exists():
    print(f"Warning: Input GeoJSON not found at {input_geojson}")
    print("Please ensure 'india.json' is present in the 'json' folder.")
    print("Download: https://github.com/adarshbiradar/maps-geojson/blob/master/india.json")
else:
    try:
        # Load original geojson
        print("Loading GeoJSON...")
        gdf = gpd.read_file(input_geojson)

        # print(gdf.columns)

        # Clean geometry values
        print("Cleaning geometries (buffer & simplify)...")
        gdf["geometry"] = gdf["geometry"].buffer(0)
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)

        # Save cleaned file
        gdf.to_file(output_geojson, driver="GeoJSON")
        print(f"Cleaned file saved to: {output_geojson}")

    except Exception as e:
        print(f"Error processing GeoJSON: {e}")

print("\n--- DSI Setup Complete ---")
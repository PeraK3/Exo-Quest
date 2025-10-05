import pandas as pd
import os
from astroquery.mast import Catalogs

# Define desired features for Sam's 8-parameter model
desired_features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_snr', 'koi_insol', 'koi_srad']

# Manual step: Download Kepler dataset from Kaggle
print("🚀 Step 1: Manually download 'cumulative.csv' from Kaggle NASA Kepler dataset.")
print("   → URL: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results")
print("   → Rename to 'data/kepler_koi_cumulative.csv'")
print("   → Or use Kaggle CLI: kaggle datasets download -d nasa/kepler-exoplanet-search-results -p data/")

# Create data dir if missing
os.makedirs('data', exist_ok=True)

# Verify/load Kepler data
csv_path = 'data/kepler_koi_cumulative.csv'
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} KOIs!")
        
        # Check for desired features
        available_features = [f for f in desired_features if f in df.columns]
        missing_features = [f for f in desired_features if f not in df.columns]
        print(f"✅ Available features: {available_features}")
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            print("   → Ensure dataset includes koi_snr, koi_insol, koi_srad. Try downloading from MAST or Kaggle.")
            print("   → MAST URL: https://archive.stsci.edu/kepler/data_search/search.php")
        else:
            print("✅ All 8 features found: ready for Sam's model!")
        
        # Display sample data for key features
        sample_columns = [f for f in desired_features if f in df.columns] + ['koi_disposition']
        print("\nSample data:")
        print(df[sample_columns].head())
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
else:
    print("❌ CSV missing—grab it from Kaggle or MAST!")

# Generate mock TESS TIC mapping (expanded for realism)
mock_kepids = [11442733, 11442734, 10000001, 10000002, 10000003]
mock_tics = ['TIC 123456789', 'TIC 987654321', 'TIC 456789123', 'TIC 789123456', 'TIC 321654987']
tess_sample = pd.DataFrame({
    'kepid': mock_kepids,
    'tic_id': mock_tics
})

# Optional: Fetch real Kepler-to-TESS mapping via astroquery (uncomment if time allows)
"""
try:
    kepids = df['kepid'].dropna().unique()[:5]  # Limit for hackathon speed
    tic_data = []
    for kepid in kepids:
        result = Catalogs.query_criteria(catalog="Tic", KIC=kepid)
        if len(result) > 0:
            tic_data.append({'kepid': kepid, 'tic_id': f"TIC {result['ID'][0]}"})
    tess_sample = pd.DataFrame(tic_data)
except Exception as e:
    print(f"❌ Astroquery failed: {e}, using mock TESS mapping")
"""

tess_sample.to_csv('data/tess_koi_map.csv', index=False)
print("✅ TESS map saved (mock or real)—ready for light curve fetches!")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Define desired features for Sam's 8-parameter model
desired_features = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_snr', 'koi_insol', 'koi_srad']

# Verify CSV exists
csv_path = 'data/kepler_koi_cumulative.csv'
if not os.path.exists(csv_path):
    print("❌ Error: 'kepler_koi_cumulative.csv' not found in 'data/' directory!")
    print("🚀 Please download the Kepler dataset:")
    print("   1. Visit: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results")
    print("   2. Download 'cumulative.csv'")
    print("   3. Rename to 'data/kepler_koi_cumulative.csv'")
    print("   4. Or use Kaggle CLI: kaggle datasets download -d nasa/kepler-exoplanet-search-results -p data/")
    print("   5. Alternative: Get from MAST (ensure all 8 features): https://archive.stsci.edu/kepler/data_search/search.php")
    print("⚠️ Cannot train without CSV. Exiting.")
    exit(1)

# Load data
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} KOIs!")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    print("   → Verify 'kepler_koi_cumulative.csv' is valid.")
    exit(1)

# Check available features
available_features = [f for f in desired_features if f in df.columns]
features = available_features if len(available_features) >= 5 else ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
missing_features = [f for f in desired_features if f not in df.columns]

# Log feature status
print(f"✅ Available features: {available_features}")
if missing_features:
    print(f"⚠️ Missing features for Sam's model: {missing_features}")
    print("   → Using fallback features: {features}")
else:
    print(f"✅ All 8 features found for Sam's model!")

# Clean data
try:
    df_clean = df[features + ['koi_disposition']].dropna()
    print(f"✅ After cleaning: {len(df_clean)} samples with features: {features}")
except Exception as e:
    print(f"❌ Error cleaning data: {e}")
    print("   → Ensure dataset includes required features and koi_disposition.")
    exit(1)

# Prep
X = df_clean[features]
y = df_clean['koi_disposition']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split & train (80/20; RF for quick/high acc)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save feature names in model for dash_app.py compatibility
clf.feature_names_in_ = features

# Eval
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Model trained! Accuracy: {acc:.2%} (~85%+ expected on KOIs: CONFIRMED/CANDIDATE/FALSE POSITIVE)")

# Save
joblib.dump(clf, "exoquest_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Saved exoquest_model.pkl & label_encoder.pkl—AI ready for scans!")
# exoquest_app.py

import streamlit as st
import pandas as pd
import joblib, json, io

# --- Load model + assets ---
clf = joblib.load('exo_model/exoquest_model.pkl')
le = joblib.load('exo_model/label_encoder.pkl')
imputer = joblib.load('exo_model/imputer.joblib')
features = json.load(open('exo_model/features.json'))

# --- Page setup ---
st.set_page_config(page_title="ExoQuest AI", page_icon="🪐", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0b0c10;
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #0b0c10;
        color: white;
    }
    .upload-box {
        border: 2px dashed #4b9cd3;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        background-color: #1e1e1e;
    }
    .prediction {
        font-size: 20px;
        margin: 5px 0;
    }
    .planet-name {
        font-weight: bold;
        color: #9bdcff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🌌 ExoQuest AI – Exoplanet Classifier")
st.markdown("Upload your CSV and let the AI reveal which worlds are real. 🚀")

# --- Upload box ---
uploaded_file = st.file_uploader("📂 Upload your exoplanet CSV", type=["csv"], label_visibility="collapsed")

# --- Prediction logic ---
if uploaded_file is not None:
    try:
        df_new = pd.read_csv(uploaded_file)
        st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
        st.write("Preview:", df_new.head())

        # Check columns
        if not all(f in df_new.columns for f in features):
            missing = [f for f in features if f not in df_new.columns]
            st.error(f"❌ Missing required columns: {missing}")
        else:
            # Predict
            X_new = pd.DataFrame(imputer.transform(df_new[features]), columns=features)
            preds = clf.predict(X_new)
            labels = le.inverse_transform(preds)
            df_new['AI_Prediction'] = labels

            # --- Display results ---
            st.markdown("### 🧠 AI Predictions")
            for i, row in df_new.iterrows():
                name = row.get('kepoi_name', f"Planet #{i+1}")
                label = row['AI_Prediction']

                if label == "CONFIRMED":
                    emoji, color = "✅", "lime"
                elif label == "CANDIDATE":
                    emoji, color = "🔭", "skyblue"
                else:
                    emoji, color = "❌", "red"

                st.markdown(
                    f"<div class='prediction' style='color:{color};'>"
                    f"<span class='planet-name'>{name}</span>: {label} {emoji}"
                    "</div>",
                    unsafe_allow_html=True
                )

            # Optional: downloadable results
            csv = df_new.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Results CSV", csv, "exoquest_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📁 Upload a CSV file to begin analysis.")

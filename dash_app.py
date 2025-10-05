import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import joblib
from lightkurve import search_lightcurve
from astroquery.mast import Catalogs
import time
import base64
from io import StringIO
import os

# Verify model and CSV files
model_path = "exoquest_model.pkl"
le_path = "label_encoder.pkl"
csv_path = "kepler_koi_cumulative.csv"
if not os.path.exists(model_path) or not os.path.exists(le_path):
    print("❌ Error: Model files missing! Run train.py to generate exoquest_model.pkl and label_encoder.pkl.")
    exit(1)
if not os.path.exists(csv_path):
    print("❌ Error: 'kepler_koi_cumulative.csv' not found! Run download_data.py and ensure CSV is in data/.")
    exit(1)

# Load pre-trained model + label encoder
try:
    clf = joblib.load(model_path)
    le = joblib.load(le_path)
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    exit(1)

# Get model features dynamically
features = clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
print(f"✅ Model loaded with features: {features}")

# Load CSV dataset for navigation
try:
    df = pd.read_csv(csv_path)
    df_clean = df[features + ['koi_disposition', 'kepid']].dropna()
    print(f"✅ Loaded {len(df_clean)} clean KOIs for navigation.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit(1)

# Fetch RA/Dec and distance for Kepler stars
tic_data = []
for kepid in df_clean['kepid'].sample(min(100, len(df_clean))):
    try:
        result = Catalogs.query_criteria(catalog="Tic", KIC=kepid)
        if len(result) > 0:
            dist = result['dstArcSec'][0] / 3600 * 1000 / 3.262 if 'dstArcSec' in result.colnames and result['dstArcSec'][0] is not None else np.nan
            tic_data.append({'kepid': kepid, 'ra': result['ra'][0], 'dec': result['dec'][0], 'distance_ly': dist})
    except Exception as e:
        print(f"❌ TIC query failed for KIC {kepid}: {e}")
        tic_data.append({'kepid': kepid, 'ra': np.nan, 'dec': np.nan, 'distance_ly': np.nan})
tic_df = pd.DataFrame(tic_data)
tic_df['ra'] = tic_df['ra'].fillna(np.random.uniform(0, 360))
tic_df['dec'] = tic_df['dec'].fillna(np.random.uniform(-90, 90))
tic_df['distance_ly'] = tic_df['distance_ly'].fillna(np.random.uniform(100, 5000))
tic_df.to_csv('data/kepler_coords.csv', index=False)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    {'href': 'https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap', 'rel': 'stylesheet'},
    '/assets/custom.css'
])

# State variables
INDEX = 0
CUSTOM_INPUT = False

# Build input form dynamically based on model features
input_fields = [
    dcc.Input(id=f"input-{f}", type="number", placeholder=f"{f.replace('koi_', '').capitalize()}", style={'marginRight': '10px', 'fontFamily': '"Press Start 2P", monospace'})
    for f in features
]
input_fields.append(html.Button("Submit Planet", id="btn-submit-input", n_clicks=0, style={'fontFamily': '"Press Start 2P", monospace'}))

app.layout = html.Div(
    style={'backgroundColor': 'black', 'color': 'lime', 'fontFamily': '"Press Start 2P", monospace'},
    children=[
        html.Div(
            id="header",
            className="header",
            children=[
                html.Img(src='/assets/ship1.png', style={'width': '100px'}),
                html.H1("🚀 ExoQuest: Hunt for Exoplanets", style={'color': 'yellow', 'textShadow': '2px 2px red', 'fontFamily': '"Press Start 2P", monospace'})
            ]
        ),
        html.Button("Chart Path ☆", id="btn-input-data", n_clicks=0, style={'fontFamily': '"Press Start 2P", monospace', 'marginTop': '10px'}),
        html.Div(
            id="input-form",
            style={'display': 'none', 'marginTop': '10px'},
            children=input_fields
        ),
        dcc.Upload(
            id='upload-csv',
            children=html.Button('Upload CSV for Batch Analysis', style={'fontFamily': '"Press Start 2P", monospace', 'marginTop': '10px'}),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'marginTop': '10px'
            }
        ),
        html.Div(id='csv-output', style={'marginTop': '10px'}),
        html.Div(
            className="controls",
            children=[
                html.Button("← Fly Left", id="btn-left", n_clicks=0, style={'fontFamily': '"Press Start 2P", monospace'}),
                html.Button("Fly Right →", id="btn-right", n_clicks=0, style={'fontFamily': '"Press Start 2P", monospace'}),
                html.Button("Chart Path ☆", id="btn-input-data-dup", n_clicks=0, style={'fontFamily': '"Press Start 2P", monospace', 'marginLeft': '10px'})
            ],
            style={'marginTop': '10px', 'display': 'flex', 'alignItems': 'center'}
        ),
        html.Div(
            className="main-plot",
            children=[dcc.Graph(id="flux-graph", config={'scrollZoom': True})]
        ),
        html.Div(
            id="prediction-text",
            style={'marginTop': '20px', 'fontSize': '20px', 'fontFamily': '"Press Start 2P", monospace'}
        ),
        html.Button("Export Discovery", id="btn-export", style={'fontFamily': '"Press Start 2P", monospace'}),
        html.Button("Enter VR Mode", id="btn-vr", n_clicks=0, style={'fontFamily': '"Press Start 2P", monospace', 'marginTop': '10px'}),
        dcc.Graph(id="vr-graph", style={'display': 'none', 'height': '600px'}),
        html.Div(id="export-status", style={'marginTop': '10px', 'color': 'yellow', 'fontFamily': '"Press Start 2P", monospace'}),
        html.Div(
            id="popup-message",
            children="",
            style={
                'display': 'none',
                'position': 'fixed',
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'backgroundColor': 'rgba(0, 0, 0, 0.8)',
                'color': 'lime',
                'padding': '20px',
                'border': '2px solid yellow',
                'fontFamily': '"Press Start 2P", monospace',
                'fontSize': '16px',
                'zIndex': '1000',
                'textAlign': 'center'
            }
        ),
        dcc.Store(id='popup-trigger', data=time.time()),
        dcc.Graph(id="importance-graph")
    ]
)

@app.callback(
    Output("input-form", "style"),
    Input("btn-input-data", "n_clicks"),
    Input("btn-input-data-dup", "n_clicks")
)
def toggle_input_form(n_clicks, n_clicks_dup):
    if (n_clicks + n_clicks_dup) % 2 == 1:
        return {'display': 'block', 'marginTop': '10px'}
    return {'display': 'none', 'marginTop': '10px'}

@app.callback(
    Output("csv-output", "children"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    State("upload-csv", "last_modified")
)
def parse_csv(contents, filename, date):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df_upload = pd.read_csv(StringIO(decoded.decode('utf-8')))
            if set(features).issubset(df_upload.columns):
                X_upload = df_upload[features]
                pred_upload = clf.predict(X_upload)
                pred_label_upload = le.inverse_transform(pred_upload)
                df_upload['Prediction'] = pred_label_upload
                confirmed_count = sum(pred_label_upload == 'CONFIRMED')
                candidate_count = sum(pred_label_upload == 'CANDIDATE')
                false_count = sum(pred_label_upload == 'FALSE')
                df_upload.to_csv("discovery_upload.csv", index=False)
                return html.Div([
                    html.H4(f"Analysis Complete: {confirmed_count} CONFIRMED, {candidate_count} CANDIDATE, {false_count} FALSE"),
                    dcc.Graph(figure=go.Figure(data=go.Bar(x=['CONFIRMED', 'CANDIDATE', 'FALSE'], y=[confirmed_count, candidate_count, false_count], marker_color='lime'))),
                    html.Table([
                        html.Thead(html.Tr([html.Th(col) for col in df_upload.columns])),
                        html.Tbody([html.Tr([html.Td(df_upload.iloc[i][col]) for col in df_upload.columns]) for i in range(min(10, len(df_upload)))])
                    ])
                ], style={'color': 'lime', 'fontFamily': '"Press Start 2P", monospace'})
            else:
                return html.Div(f"CSV must have columns: {', '.join(features)}", style={'color': 'red'})
        except Exception as e:
            return html.Div(f"Error parsing CSV: {e}", style={'color': 'red'})
    return "No file uploaded"

@app.callback(
    [Output("flux-graph", "figure"),
     Output("prediction-text", "children"),
     Output("popup-trigger", "data")],
    [Input("btn-left", "n_clicks"),
     Input("btn-right", "n_clicks"),
     Input("btn-submit-input", "n_clicks")],
    [State(f"input-{f}", "value") for f in features] +
    [State("flux-graph", "figure"),
     State("popup-trigger", "data")]
)
def update_graph(left_clicks, right_clicks, submit_clicks, *args):
    global INDEX, CUSTOM_INPUT
    ctx = dash.callback_context
    inputs = args[:-2]
    existing, popup_count = args[-2:]

    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    popup_count = time.time()

    if button_id == "btn-left":
        INDEX = max(0, INDEX - 1)
        CUSTOM_INPUT = False
    elif button_id == "btn-right":
        INDEX = min(len(df_clean)-1, INDEX + 1)
        CUSTOM_INPUT = False
    elif button_id == "btn-submit-input":
        CUSTOM_INPUT = True

    if CUSTOM_INPUT and button_id == "btn-submit-input":
        try:
            inputs = [float(x) if x is not None else 0 for x in inputs]
            if any(x <= 0 for x in inputs):
                raise ValueError("Inputs must be positive")
            sample = np.array(inputs).reshape(1, -1)
            pred = clf.predict(sample)
            pred_label = le.inverse_transform(pred)[0]
            period = inputs[features.index('koi_period')] if 'koi_period' in features else 10
            duration = inputs[features.index('koi_duration')] if 'koi_duration' in features else 2
            depth = inputs[features.index('koi_depth')] if 'koi_depth' in features else 1000
            time_vals = np.linspace(0, duration, 100)
            flux_vals = 1 - np.sin(time_vals / period) * (depth / 10000)
            title = f"Custom Planet: {pred_label}"
            distance_ly = "Unknown"
            if pred_label == "CONFIRMED":
                popup_message = "Exoplanet: CONFIRMED! Your path charts to discovery"
            elif pred_label == "CANDIDATE":
                popup_message = "Exoplanet: CANDIDATE! A curious find on your path Captain!"
            else:
                popup_message = "Exoplanet: FALSE POSITIVE! Chart again for a better find next time, Captain"
        except:
            pred_label = "Invalid Input"
            time_vals = np.linspace(0, 10, 100)
            flux_vals = np.ones(100)
            title = "Custom Planet: Invalid Input"
            distance_ly = "Unknown"
            popup_message = "Invalid input, Captain! Try positive numbers."
    else:
        row = df_clean.iloc[INDEX]
        sample = row[features].values.reshape(1, -1)
        pred = clf.predict(sample)
        pred_label = le.inverse_transform(pred)[0]

        tic_map = pd.read_csv('data/tess_koi_map.csv')
        kepid_val = row['kepid']
        tic_match = tic_map[tic_map['kepid'] == kepid_val]
        tic_id = tic_match['tic_id'].iloc[0] if not tic_match.empty else None

        dist_match = tic_df[tic_df['kepid'] == kepid_val]
        distance_ly = dist_match['distance_ly'].iloc[0] if not dist_match.empty else np.random.uniform(100, 5000)
        distance_ly = round(distance_ly, 1) if not np.isnan(distance_ly) else "Unknown"

        time_vals, flux_vals = None, None
        if tic_id:
            try:
                lc = search_lightcurve(tic_id, mission="TESS").download(flux_column='pdcsap_flux')
                if lc and lc.flux is not None:
                    time_vals = lc.time.value
                    flux_vals = lc.flux / np.nanmedian(lc.flux)
            except Exception as e:
                print(f"❌ Lightkurve failed for {tic_id}: {e}")

        if time_vals is None or flux_vals is None:
            period = row['koi_period'] if 'koi_period' in row else 10
            duration = row['koi_duration'] if 'koi_duration' in row else 2
            depth = row['koi_depth'] if 'koi_depth' in row else 1000
            time_vals = np.linspace(0, duration, 100)
            flux_vals = 1 - np.sin(time_vals / period) * (depth / 10000)
        title = f"Planet Spotted: {pred_label} (TIC: {tic_id or 'Mock'})"
        popup_message = f"You've found an {pred_label} Captain!\nDistance: {distance_ly} ly" if pred_label in ["CONFIRMED", "CANDIDATE"] else f"Keep Exploring Captain, Discovery Awaits Us!\nDistance: {distance_ly} ly"

    fig = go.Figure(data=go.Scatter(x=time_vals, y=flux_vals, mode='lines+markers', line=dict(color='lime')))
    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='lime', family='"Press Start 2P", monospace'),
        title_text=title,
        xaxis_title="Time (days)",
        yaxis_title="Normalized Flux"
    )

    return fig, f"Prediction: {pred_label}", popup_count

@app.callback(
    Output("popup-message", "children"),
    Output("popup-message", "style"),
    Input("popup-trigger", "data")
)
def show_popup(trigger_timestamp):
    global INDEX, CUSTOM_INPUT
    style = {
        'display': 'block',
        'position': 'fixed',
        'top': '50%',
        'left': '50%',
        'transform': 'translate(-50%, -50%)',
        'backgroundColor': 'rgba(0, 0, 0, 0.8)',
        'color': 'lime',
        'padding': '20px',
        'border': '2px solid yellow',
        'fontFamily': '"Press Start 2P", monospace',
        'fontSize': '16px',
        'zIndex': '1000',
        'textAlign': 'center'
    }

    if CUSTOM_INPUT:
        try:
            row = df_clean.iloc[INDEX]
            sample = row[features].values.reshape(1, -1)
            pred = clf.predict(sample)
            pred_label = le.inverse_transform(pred)[0]
            if pred_label == "CONFIRMED":
                message = "Exoplanet: CONFIRMED! Your path charts to discovery"
            elif pred_label == "CANDIDATE":
                message = "Exoplanet: CANDIDATE! A curious find on your path Captain!"
            else:
                message = "Exoplanet: FALSE POSITIVE! Chart again for a better find next time, Captain"
        except:
            message = "Invalid input, Captain! Try positive numbers."
    else:
        row = df_clean.iloc[INDEX]
        sample = row[features].values.reshape(1, -1)
        pred = clf.predict(sample)
        pred_label = le.inverse_transform(pred)[0]
        dist_match = tic_df[tic_df['kepid'] == row['kepid']]
        distance_ly = dist_match['distance_ly'].iloc[0] if not dist_match.empty else np.random.uniform(100, 5000)
        distance_ly = round(distance_ly, 1) if not np.isnan(distance_ly) else "Unknown"
        message = f"You've found an {pred_label} Captain!\nDistance: {distance_ly} ly" if pred_label in ["CONFIRMED", "CANDIDATE"] else f"Keep Exploring Captain, Discovery Awaits Us!\nDistance: {distance_ly} ly"

    return message, style

@app.callback(
    [Output("vr-graph", "figure"),
     Output("vr-graph", "style")],
    Input("btn-vr", "n_clicks"),
    Input("btn-left", "n_clicks"),
    Input("btn-right", "n_clicks"),
    prevent_initial_call=True
)
def vr_mode(n_clicks, left_clicks, right_clicks):
    global INDEX, CUSTOM_INPUT

    if n_clicks > 0 and n_clicks % 2 == 1:
        if INDEX >= len(df_clean):
            INDEX = 0  # Prevent index out of range

        n_stars = min(100, len(df_clean))
        row = df_clean.iloc[INDEX]

        if len(tic_df) < n_stars:
            n_stars = len(tic_df)

        ra = tic_df['ra'].iloc[:n_stars]
        dec = tic_df['dec'].iloc[:n_stars]
        temp = df_clean['koi_teq'].iloc[:n_stars].values

        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)

        distance = temp / np.nanmax(temp) * 1000
        x *= distance
        y *= distance
        z *= distance

        dispositions = df_clean['koi_disposition'].iloc[:n_stars].values
        colors = []
        for d in dispositions:
            if d == 'CONFIRMED':
                colors.append('lime')
            elif d == 'CANDIDATE':
                colors.append('magenta')
            elif d == 'FALSE POSITIVE':
                colors.append('orange')
            else:
                colors.append('white')

        radii = df_clean['koi_prad'].iloc[:n_stars].values
        if len(radii) == 0:
            radii = np.ones(n_stars)

        sizes = (5 + 15 * (radii - np.min(radii)) / (np.max(radii) - np.min(radii) + 1e-6)) * 5

        cur_ra = np.deg2rad(ra.iloc[0])
        cur_dec = np.deg2rad(dec.iloc[0])
        cur_temp = row['koi_teq']

        cur_x = np.cos(cur_dec) * np.cos(cur_ra)
        cur_y = np.cos(cur_dec) * np.sin(cur_ra)
        cur_z = np.sin(cur_dec)

        cur_distance = cur_temp / np.nanmax(df_clean['koi_teq']) * 2
        cur_x *= cur_distance
        cur_y *= cur_distance
        cur_z *= cur_distance

        current_size = [20]
        current_color = ['yellow']

        fig_vr = go.Figure(data=[
            go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(size=sizes, color=colors, opacity=0.7)
            ),
            go.Scatter3d(
                x=[cur_x], y=[cur_y], z=[cur_z], mode='markers',
                marker=dict(size=current_size, color=current_color, opacity=1, symbol='diamond')
            )
        ])

        fig_vr.update_layout(
            scene=dict(
                bgcolor='black',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            title=dict(text="🌌 VR Kepler Star Map", font=dict(family='"Press Start 2P", monospace', color='lime')),
            paper_bgcolor='black',
            font=dict(color='lime', family='"Press Start 2P", monospace'),
            scene_aspectmode='data',
            width=1200,
            height=800,
            scene_camera=dict(eye=dict(x=2, y=2, z=2)),
        )

        return fig_vr, {'display': 'block', 'height': '600px', 'backgroundColor': 'black'}
    else:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
        return empty_fig, {'display': 'none', 'backgroundColor': 'black'}




@app.callback(
    Output("importance-graph", "figure"),
    Input("btn-left", "n_clicks"),
    Input("btn-right", "n_clicks")
)
def update_importance(left_clicks, right_clicks):
    importances = clf.feature_importances_
    fig = go.Figure(data=go.Bar(
        x=features,
        y=importances,
        marker_color='lime'
    ))
    fig.update_layout(
        title="Feature Importance",
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='lime', family='"Press Start 2P", monospace'),
        xaxis_title="Features",
        yaxis_title="Importance"
    )
    return fig

@app.callback(
    Output("export-status", "children"),
    Input("btn-export", "n_clicks"),
    State("prediction-text", "children")
)
def export_discovery(n_clicks, prediction_text):
    global INDEX, CUSTOM_INPUT
    if n_clicks and not CUSTOM_INPUT:
        row = df_clean.iloc[INDEX]
        pred_label = prediction_text.replace("Prediction: ", "")
        export_data = pd.DataFrame({
            'kepid': [row['kepid']],
            **{f: [row[f]] for f in features},
            'Prediction': [pred_label]
        })
        export_data.to_csv('discovery.csv', mode='a', header=not os.path.exists('discovery.csv'), index=False)
        return "Exported to discovery.csv!"
    return "Export only for navigated planets."

if __name__ == '__main__':
    app.run(debug=True)
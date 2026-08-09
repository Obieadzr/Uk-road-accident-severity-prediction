import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load model artifacts ────────────────────────────────────────────────
# These three files must sit in a "Models" folder next to this app.py
model = joblib.load('Models/xgb_model.pkl')
scaler = joblib.load('Models/scaler.pkl')
feature_columns = joblib.load('Models/feature_columns.pkl')

THRESHOLD = 0.0091  # chosen for Fatal-class recall ~0.60 (see project writeup)

st.set_page_config(page_title="UK Road Accident Severity Predictor", layout="centered")
st.title("UK Road Accident Severity Predictor")
st.caption(
    "Portfolio demo — predicts accident severity (Fatal / Serious / Slight) from "
    "UK collision-report-style inputs. Model is tuned to prioritize catching Fatal "
    "cases (recall ≈ 0.60) at the cost of more false alarms — see README for the "
    "full precision/recall tradeoff and limitations."
)

st.divider()
st.subheader("Accident details")

# ── Official STATS19 code lookups (from DfT road-safety-open-dataset-data-guide) ──
DAY_OF_WEEK = {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"}
URBAN_RURAL = {1:"Urban",2:"Rural",3:"Unallocated"}
FIRST_ROAD_CLASS = {1:"Motorway",2:"A(M)",3:"A",4:"B",5:"C",6:"Unclassified"}
SECOND_ROAD_CLASS = {0:"Not at junction or within 20 metres",1:"Motorway",2:"A(M)",3:"A",4:"B",5:"C",6:"Unclassified",9:"Unknown"}
ROAD_TYPE = {1:"Roundabout",2:"One way street",3:"Dual carriageway",6:"Single carriageway",7:"Slip road",9:"Unknown",12:"One way street/Slip road"}
JUNCTION_DETAIL = {0:"Not at junction or within 20 metres",13:"T or staggered junction",16:"Crossroads",
                    17:"Junction with more than 4 arms (not roundabout)",18:"Using private drive or entrance",
                    19:"Other junction",99:"Unknown"}
PEDESTRIAN_CROSSING = {0:"No physical crossing facility within 50m",11:"Human control - school patrol",
                        12:"Human control - other authorised person",13:"Zebra crossing",
                        14:"Pedestrian light crossing (pelican/puffin/toucan)",15:"Pedestrian phase at traffic signal",
                        16:"Footbridge or subway",17:"Central refuge - no other controls",99:"Unknown"}
LIGHT_CONDITIONS = {1:"Daylight",4:"Darkness - lights lit",5:"Darkness - lights unlit",
                     6:"Darkness - no lighting",7:"Darkness - lighting unknown"}
WEATHER_CONDITIONS = {1:"Fine, no high winds",2:"Raining, no high winds",3:"Snowing, no high winds",
                       4:"Fine + high winds",5:"Raining + high winds",6:"Snowing + high winds",
                       7:"Fog or mist",8:"Other",9:"Unknown"}
ROAD_SURFACE = {1:"Dry",2:"Wet or damp",3:"Snow",4:"Frost or ice",5:"Flood over 3cm deep",
                 6:"Oil or diesel",7:"Mud",9:"Unknown"}
SPECIAL_CONDITIONS = {0:"None",1:"Auto traffic signal - out",2:"Auto signal part defective",
                       3:"Road sign/marking defective or obscured",4:"Roadworks",5:"Road surface defective",
                       6:"Oil or diesel",7:"Mud",9:"Unknown"}
CARRIAGEWAY_HAZARDS = {0:"None",11:"Defective traffic signals",12:"Signing/markings defective or obscured",
                        13:"Roadworks",14:"Oil or diesel",15:"Mud",16:"Dislodged vehicle load in carriageway",
                        17:"Other object in carriageway",18:"Involvement with previous collision",
                        19:"Pedestrian in carriageway - not injured",20:"Animal in carriageway",
                        21:"Poor or defective road surface",99:"Unknown"}

def selectbox_from_dict(label, d, default_key=None, help=None):
    keys = list(d.keys())
    idx = keys.index(default_key) if default_key in keys else 0
    return st.selectbox(label, options=keys, index=idx, format_func=lambda k: f"{d[k]}", help=help)

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-1.5, format="%.5f")
    latitude = st.number_input("Latitude", value=52.5, format="%.5f")
    number_of_vehicles = st.number_input("Number of vehicles", min_value=1, max_value=20, value=2)
    number_of_casualties = st.number_input("Number of casualties", min_value=1, max_value=50, value=1)
    speed_limit = st.selectbox("Speed limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
    day_of_week = selectbox_from_dict("Day of week", DAY_OF_WEEK, default_key=2)
    urban_or_rural_area = selectbox_from_dict("Area type", URBAN_RURAL, default_key=1)

with col2:
    date = st.date_input("Date of accident")
    time = st.time_input("Time of accident")
    junction_detail = selectbox_from_dict("Junction detail", JUNCTION_DETAIL, default_key=0)
    first_road_class = selectbox_from_dict("First road class", FIRST_ROAD_CLASS, default_key=3)
    second_road_class = selectbox_from_dict("Second road class", SECOND_ROAD_CLASS, default_key=0)
    road_type = selectbox_from_dict("Road type", ROAD_TYPE, default_key=6)
    pedestrian_crossing = selectbox_from_dict("Pedestrian crossing", PEDESTRIAN_CROSSING, default_key=0)

st.divider()
st.subheader("Conditions")

col3, col4 = st.columns(2)
with col3:
    light_conditions = selectbox_from_dict("Light conditions", LIGHT_CONDITIONS, default_key=1)
    weather_conditions = selectbox_from_dict("Weather conditions", WEATHER_CONDITIONS, default_key=1)
with col4:
    road_surface_conditions = selectbox_from_dict("Road surface conditions", ROAD_SURFACE, default_key=1)
    special_conditions_at_site = selectbox_from_dict("Special conditions at site", SPECIAL_CONDITIONS, default_key=0)
    carriageway_hazards = selectbox_from_dict("Carriageway hazards", CARRIAGEWAY_HAZARDS, default_key=0)

st.divider()

if st.button("Predict severity", type="primary"):
    # ── Replicate Step 3 feature engineering exactly ────────────────────
    hour = time.hour
    month = date.month

    season = 0 if month in [12, 1, 2] else 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8] else 3
    is_rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    is_weekend = 1 if day_of_week in [1, 7] else 0
    is_dark = 1 if light_conditions in [4, 5, 6, 7] else 0
    is_bad_weather = 1 if weather_conditions in [2, 3, 5, 6, 7] else 0
    is_highspeed = 1 if speed_limit >= 60 else 0
    is_urban = 1 if urban_or_rural_area == 1 else 0
    is_junction = 0 if junction_detail == 0 else 1
    is_hazards = 0 if carriageway_hazards == 0 else 1

    row = {
        'longitude': longitude,
        'latitude': latitude,
        'number_of_vehicles': number_of_vehicles,
        'number_of_casualties': number_of_casualties,
        'day_of_week': day_of_week,
        'first_road_class': first_road_class,
        'road_type': road_type,
        'speed_limit': speed_limit,
        'junction_detail': junction_detail,
        'second_road_class': second_road_class,
        'pedestrian_crossing': pedestrian_crossing,
        'light_conditions': light_conditions,
        'weather_conditions': weather_conditions,
        'road_surface_conditions': road_surface_conditions,
        'special_conditions_at_site': special_conditions_at_site,
        'carriageway_hazards': carriageway_hazards,
        'urban_or_rural_area': urban_or_rural_area,
        'hour': hour,
        'month': month,
        'season': season,
        'is_rush_hour': is_rush_hour,
        'is_weekend': is_weekend,
        'is_dark': is_dark,
        'is_bad_weather': is_bad_weather,
        'is_highspeed': is_highspeed,
        'is_urban': is_urban,
        'is_junction': is_junction,
        'is_hazards': is_hazards,
    }

    X_input = pd.DataFrame([row])
    # enforce exact column order the model was trained on
    X_input = X_input[feature_columns]

    # ── Replicate Step 5 scaling exactly (same columns, fitted scaler) ──
    cols_to_scale = ['longitude', 'latitude', 'speed_limit', 'number_of_vehicles',
                      'hour', 'month', 'number_of_casualties']
    X_input[cols_to_scale] = scaler.transform(X_input[cols_to_scale])

    # ── Predict with thresholded Fatal decision ─────────────────────────
    proba = model.predict_proba(X_input)[0]
    p_fatal, p_serious, p_slight = proba[0], proba[1], proba[2]

    if p_fatal >= THRESHOLD:
        pred_label = "Fatal"
    else:
        pred_label = "Serious" if p_serious >= p_slight else "Slight"

    st.divider()
    if pred_label == "Fatal":
        st.error(f"Predicted severity: **{pred_label}**")
    elif pred_label == "Serious":
        st.warning(f"Predicted severity: **{pred_label}**")
    else:
        st.success(f"Predicted severity: **{pred_label}**")

    st.write("Predicted probabilities:")
    st.write({
        "Fatal": f"{p_fatal:.4f}",
        "Serious": f"{p_serious:.4f}",
        "Slight": f"{p_slight:.4f}",
    })
    st.caption(
        f"Fatal is flagged if P(Fatal) ≥ {THRESHOLD} (a deliberately low bar, chosen "
        "to catch more real fatal cases at the cost of more false alarms — see project "
        "writeup for the precision/recall tradeoff)."
    )

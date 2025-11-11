# app.py
import streamlit as st
import joblib
import json
import numpy as np
import os

st.set_page_config(page_title="Wellness Personas", layout="wide")

# --- Optional: load training CSV for realistic slider ranges & medians ---
import pandas as pd
_train_stats = {}
try:
    _df_stats = pd.read_csv("Data Collection for ML mini project (Responses) - Form Responses 1.csv")
    # numeric min, max, median for later use
    _train_stats['mins'] = _df_stats.min(numeric_only=True)
    _train_stats['maxs'] = _df_stats.max(numeric_only=True)
    _train_stats['meds'] = _df_stats.median(numeric_only=True)
except Exception:
    _train_stats = None

st.title("Wellness Personas by Md.Nafiul Islam")
st.write("Enter your habits (4 numbers) and the app tells which persona you match.")
st.sidebar.header("How It Works 🧠")
st.sidebar.write("""
This app uses a K-Means clustering model trained on student survey data.
1. Fill the 4 inputs on the left.
2. Click Predict.
3. See which Wellness Persona you match and why.
""")


# --- Load artifacts ---
@st.cache_resource
def load_files():
    required = ['scaler.pkl','kmeans.pkl','features.json','persona_mapping.json','persona_descriptions.json']
    for f in required:
        if not os.path.exists(f):
            st.error(f"Missing file: {f}. Put it in the same folder as app.py.")
            st.stop()
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans.pkl')
    with open('features.json','r') as fh:
        features = json.load(fh)
        
       # Load persona mapping safely
    try:
        with open('persona_mapping.json','r') as fh:
            persona_map = json.load(fh)
    except Exception as e:
        st.error(f"Failed to load persona_mapping.json: {e}")
        st.stop()

    # Normalize persona_map keys to strings (important)
    try:
        persona_map = {str(k): v for k, v in persona_map.items()}
    except Exception as e:
        st.error(f"persona_mapping.json has unexpected format: {e}")
        st.stop()

    # Load persona descriptions safely
    try:
        with open('persona_descriptions.json','r') as fh:
            persona_desc = json.load(fh)
    except Exception as e:
        st.error(f"Failed to load persona_descriptions.json: {e}")
        st.stop()

    return scaler, kmeans, features, persona_map, persona_desc

scaler, kmeans, features, persona_map, persona_desc = load_files()

# --- Helper: clean label for display (shorten long feature names) ---
def pretty_label(name):
    return name.replace('_',' ').replace('  ',' ').strip().capitalize()

# --- Input widgets (left column) ---
left, right = st.columns([1,1])
with left:
    st.header("Your inputs")
    inputs = []
    # Heuristic: map common features to sensible widgets by checking keywords
        # Choose widget by keyword
            # get fallback min/max/median
    def stat_for(feat_name, key, fallback):
        if _train_stats is None: return fallback
        # names in CSV might differ slightly; try direct then fallback to 0/20 etc.
        try:
            return int(_train_stats[key].get(feat_name, fallback))
        except Exception:
            return fallback

    for feat in features:
        key = feat
        label = pretty_label(feat)
        # sensible defaults from training data if available
        minv = stat_for(feat, 'mins', 0)
        maxv = stat_for(feat, 'maxs', 20)
        medv = stat_for(feat, 'meds', 3)

        if 'eat' in feat.lower() or 'eating' in feat.lower():
            val = st.slider(label, min_value=max(0,minv), max_value=maxv if maxv>0 else 21, value=min(maxv, medv if medv>=0 else 3))
        elif 'budget' in feat.lower() or 'price' in feat.lower() or 'rupees' in feat.lower():
            # budgets may be large; enforce reasonable cap
            val = st.number_input(label + " (₹)", min_value=0, max_value=max(100000, maxv), value=max(100, medv))
        elif 'sweet' in feat.lower() or 'tooth' in feat.lower():
            val = st.slider(label + " (1-5)", min_value=1, max_value=5, value=int(min(max(1,medv),5)))
        elif 'hobby' in feat.lower() or 'hours' in feat.lower():
            val = st.slider(label + " (hours/week)", min_value=0, max_value=max(40, maxv), value=min(40, medv))
        else:
            # fallback numeric input
            val = st.number_input(label, value=float(medv))
        inputs.append(val)


    if st.button("Predict"):
        X = np.array(inputs).reshape(1, -1)
        X_scaled = scaler.transform(X)
        cluster = int(kmeans.predict(X_scaled)[0])
        persona_name = persona_map.get(str(cluster), persona_map.get(cluster, f"Cluster {cluster}"))
        desc = persona_desc.get(str(cluster), persona_desc.get(cluster, "No description found."))
        st.session_state['result'] = (cluster, persona_name, desc, X_scaled[0].tolist())

with right:
    st.header("Result")
    if 'result' in st.session_state:
        cluster, persona_name, desc, scaled_values = st.session_state['result']
            # emoji + success message
        emoji_map = {
            "Balanced Mainstream": "⚖️",
            "Passionate Hobbyist": "🎨",
            "High-End Spender": "💸",
            "Everyday Eater": "🍔"
        }
        emoji = emoji_map.get(persona_name, "🧩")
        st.success(f"You belong to the {persona_name} group! {emoji} 🎉")
        st.subheader(f"{emoji} {persona_name}")
        st.write(desc)
        st.markdown("---")
        st.write("**Cluster id:**", cluster)
        # show scaled feature values (small table)
        import pandas as pd
        df = pd.DataFrame([scaled_values], columns=[pretty_label(f) for f in features])
        st.write("Scaled input (how model sees you):")
        st.dataframe(df.round(3))
        # show distances to all centroids (why this persona)
        distances = kmeans.transform(np.array(inputs).reshape(1, -1))[0]
        st.write("### Distance from each Persona Center (smaller = closer):")
        for i, d in enumerate(distances):
            st.write(f"Cluster {i}: {d:.3f}")
            # cluster centers (scaled values) to compare
        import pandas as pd
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=[pretty_label(f) for f in features])
        centroids.index = [f"Cluster {i}" for i in range(len(centroids))]
        st.write("### Cluster Centers (Scaled Values):")
        st.dataframe(centroids.round(2))

    else:
        st.info("Fill inputs on the left and click Predict.")
        
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()



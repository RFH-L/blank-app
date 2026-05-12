import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Weather Detector",
    layout="wide"
)

st.title("🌍 AI Weather Detector")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙ Kontrol Sistem")

# =========================================================
# MENU
# =========================================================

menu = st.sidebar.radio(
    "Pilih Analisa",
    [
        "Cuaca",
        "Udara"
    ]
)

# =========================================================
# PREDIKSI WAKTU
# =========================================================

waktu = st.sidebar.selectbox(
    "Prediksi Waktu",
    [
        "Hari Ini",
        "1 Minggu",
        "1 Bulan",
        "1 Tahun",
        "10 Tahun"
    ]
)

# =========================================================
# TEMA
# =========================================================

theme_mode = st.sidebar.selectbox(
    "Tema Peta",
    [
        "dark",
        "light"
    ]
)

# =========================================================
# KEPADATAN TITIK
# =========================================================

st.sidebar.subheader("📍 Kepadatan")

density = st.sidebar.slider(
    "Jumlah Titik",
    min_value=50,
    max_value=1000,
    value=250,
    step=50
)

spread = st.sidebar.slider(
    "Luas Persebaran",
    min_value=0.01,
    max_value=0.25,
    value=0.05,
    step=0.01
)

# =========================================================
# ANIMASI
# =========================================================

st.sidebar.subheader("🎞 Animasi")

enable_animation = st.sidebar.toggle(
    "Aktifkan Animasi",
    value=False
)

# =========================================================
# MAP STYLE
# =========================================================

if theme_mode == "light":

    map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

else:

    map_style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

# =========================================================
# GENERATE DATA
# =========================================================

np.random.seed(42)

center_lat = -6.20
center_lon = 106.81

# =========================================================
# FUNCTION GENERATE DATA
# =========================================================

def generate_weather_data():

    df = pd.DataFrame({

        "lat": np.random.randn(density) * spread + center_lat,

        "lon": np.random.randn(density) * spread + center_lon,

        "suhu": np.random.uniform(24, 36, density),

        "kelembapan": np.random.uniform(40, 100, density),

        "angin": np.random.uniform(0, 35, density),

        "tekanan": np.random.uniform(980, 1030, density),

        "pm25": np.random.uniform(10, 200, density),
    })

    return df

# =========================================================
# CREATE DATA
# =========================================================

df = generate_weather_data()

# =========================================================
# FUTURE SIMULATION
# =========================================================

if waktu == "1 Minggu":

    df["suhu"] += np.random.uniform(0, 1, len(df))

elif waktu == "1 Bulan":

    df["suhu"] += np.random.uniform(1, 2, len(df))

elif waktu == "1 Tahun":

    df["suhu"] += np.random.uniform(2, 4, len(df))

elif waktu == "10 Tahun":

    df["suhu"] += np.random.uniform(4, 8, len(df))

# =========================================================
# WEATHER DETECTION
# =========================================================

def kategori_cuaca(row):

    if row["angin"] > 25:
        return "Badai"

    elif row["kelembapan"] > 85:
        return "Hujan"

    elif row["suhu"] > 32:
        return "Cerah"

    else:
        return "Mendung"

df["cuaca"] = df.apply(kategori_cuaca, axis=1)

# =========================================================
# AIR QUALITY
# =========================================================

def kualitas_udara(pm):

    if pm < 50:
        return "Segar"

    elif pm < 100:
        return "Normal"

    elif pm < 150:
        return "Buruk"

    else:
        return "Sangat Buruk"

df["udara"] = df["pm25"].apply(kualitas_udara)

# =========================================================
# DISASTER DETECTION
# =========================================================

def deteksi_bencana(row):

    if row["angin"] > 30:
        return "Tornado"

    elif row["kelembapan"] > 95:
        return "Banjir"

    elif row["tekanan"] < 985:
        return "Badai Ekstrem"

    else:
        return "Aman"

df["bencana"] = df.apply(deteksi_bencana, axis=1)

# =========================================================
# AI ANN
# =========================================================

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["cuaca"])

X = df[[
    "suhu",
    "kelembapan",
    "angin",
    "tekanan",
    "pm25"
]]

y = df["label"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_scaled)

df["prediksi"] = encoder.inverse_transform(pred)

# =========================================================
# DYNAMIC COLORS
# =========================================================

colors = []

for _, row in df.iterrows():

    # =====================================================
    # WEATHER
    # =====================================================

    if menu == "Cuaca":

        if row["prediksi"] == "Cerah":
            colors.append([255, 215, 0])

        elif row["prediksi"] == "Hujan":
            colors.append([30, 144, 255])

        elif row["prediksi"] == "Mendung":
            colors.append([180, 180, 180])

        else:
            colors.append([255, 0, 0])

    # =====================================================
    # AIR
    # =====================================================

    elif menu == "Udara":

        if row["udara"] == "Segar":
            colors.append([0, 255, 127])

        elif row["udara"] == "Normal":
            colors.append([255, 215, 0])

        elif row["udara"] == "Buruk":
            colors.append([255, 140, 0])

        else:
            colors.append([255, 0, 0])

    # =====================================================
    # DISASTER
    # =====================================================

    else:

        if row["bencana"] == "Aman":
            colors.append([0, 255, 127])

        elif row["bencana"] == "Banjir":
            colors.append([30, 144, 255])

        elif row["bencana"] == "Tornado":
            colors.append([255, 0, 0])

        else:
            colors.append([148, 0, 211])

df["color"] = colors

# =========================================================
# TOOLTIP
# =========================================================

tooltip = {
    "html": """
    <b>Cuaca:</b> {prediksi}<br/>
    <b>Suhu:</b> {suhu} °C<br/>
    <b>Kelembapan:</b> {kelembapan}%<br/>
    <b>Udara:</b> {udara}<br/>
    <b>Bencana:</b> {bencana}
    """
}

# =========================================================
# VIEW STATE
# =========================================================

zoom_level = max(7, 12 - (spread * 40))

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom_level,
    pitch=0
)

# =========================================================
# MAP CONTAINER
# =========================================================

map_container = st.empty()

# =========================================================
# FUNCTION RENDER MAP
# =========================================================

def render_map(dataframe):

    radius = max(120, 900 - density)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=dataframe,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=radius,
        opacity=0.7,
        pickable=True
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip
    )

    map_container.pydeck_chart(deck)

# =========================================================
# ANIMATION LOOP
# =========================================================

if enable_animation:

    for _ in range(5):

        animated_df = df.copy()

        animated_df["lat"] += np.random.uniform(
            -0.01,
            0.01,
            len(animated_df)
        )

        animated_df["lon"] += np.random.uniform(
            -0.01,
            0.01,
            len(animated_df)
        )

        render_map(animated_df)

        time.sleep(1)

    st.rerun()

# =========================================================
# NORMAL MAP
# =========================================================

else:

    render_map(df)

# =========================================================
# METRICS
# =========================================================

st.subheader(f"📊 Statistik {menu}")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Suhu Rata-rata",
        f"{df['suhu'].mean():.1f} °C"
    )

with col2:

    st.metric(
        "Kelembapan",
        f"{df['kelembapan'].mean():.1f}%"
    )

with col3:

    st.metric(
        "PM2.5",
        f"{df['pm25'].mean():.1f}"
    )

# =========================================================
# HISTORY 1 MONTH
# =========================================================

st.subheader("📈 Histori 1 Bulan")

today = datetime.now()

dates = [
    today - timedelta(days=i)
    for i in range(30)
]

dates.reverse()

history_df = pd.DataFrame({

    "Tanggal": dates,

    "Suhu": np.random.uniform(24, 35, 30),

    "Kelembapan": np.random.uniform(50, 95, 30)
})

st.line_chart(
    history_df.set_index("Tanggal")
)

# =========================================================
# FORECAST 10 YEARS
# =========================================================

st.subheader("🌍 Prediksi Suhu Selama 10 Tahun")

years = np.arange(2025, 2036)

future_temp = []

base_temp = df["suhu"].mean()

for i in range(len(years)):

    future_temp.append(
        base_temp + (i * 0.25)
    )

forecast_df = pd.DataFrame({

    "Tahun": years,

    "Prediksi Suhu": future_temp
})

st.line_chart(
    forecast_df.set_index("Tahun")
)

# =========================================================
# TABLE
# =========================================================

st.subheader("📋 Data AI")

st.dataframe(df)

# =========================================================
# FOOTER
# =========================================================

st.caption(
    "AI Weather Detector • ANN + Streamlit + PyDeck"
)

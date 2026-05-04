import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random
import time

st.set_page_config(page_title="Future Predictor Maps", layout="wide", initial_sidebar_state="collapsed")
# Masukkan CSS ini di bagian paling atas script
st.markdown("""
    <style>
    /* Mengatur kontainer utama tombol (kolom) agar terlihat seperti satu baris menu */
    [data-testid="stHorizontalBlock"] {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px) saturate(180%);
        border-radius: 50px;
        border: 1px solid rgba(255, 255, 255, 0.125);
        padding: 10px 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        width: fit-content !important;
        margin: 20px auto !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Menghilangkan border standar tombol Streamlit */
    div.stButton > button {
        background-color: transparent !important;
        border: none !important;
        color: #f8fafc !important;
        font-size: 26px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }

    /* Efek Hover */
    div.stButton > button:hover {
        color: #38bdf8 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.6);
        transform: translateY(-3px);
    }

    /* Mengatur spasi antar kolom agar rapat */
    [data-testid="column"] {
        display: flex;
        justify-content: center;
        align-items: center;
        width: fit-content !important;
        flex: unset !important;
        min-width: 60px !important;
    }
    
    /* Garis pemisah khusus untuk col4 */
    .menu-divider {
        width: 1.5px;
        height: 25px;
        background: rgba(255, 255, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)
# --- 1. FUNGSI LOGIKA ANN ---
def prediksi_sinyal_ann(data_tower, faktor_jam):
    # Logika ANN sederhana untuk beban sinyal
    weights = np.array([0.7, 0.3])
    bias = 0.5
    hasil = []
    for index, row in data_tower.iterrows():
        # Simulasi beban berdasarkan jam sibuk
        beban = (random.random() * 50) + (faktor_jam * 15)
        beban = min(100, beban)
        
        status = "Overload" if beban > 85 else "Padat" if beban > 60 else "Stabil"
        hasil.append({'beban': beban, 'status_sinyal': status})
    return pd.DataFrame(hasil)

def generate_detail_wilayah(data_induk, jumlah):
    bank_nama = {
        "Jakarta Barat": ["Slipi", "Tomang", "Grogol", "Kemanggisan", "Palmerah"],
        "Jakarta Selatan": ["Tebet", "Kemang", "Senopati", "Cilandak", "Jagakarsa"],
        "Jakarta Pusat": ["Menteng", "Gambir", "Senen", "Tanah Abang", "Sudirman"],
        "Jakarta Timur": ["Rawamangun", "Jatinegara", "Cakung", "Ciracas", "Cipayung"]
    }
    detail_data = []
    for _, row in data_induk.iterrows():
        opsi_nama = bank_nama.get(row['city'], ["Area Umum"])
        for i in range(jumlah):
            nama_lokasi = random.choice(opsi_nama)
            detail_data.append({
                'city': f"{nama_lokasi}, {row['city']}", 
                'lat': row['lat'] + np.random.uniform(-0.015, 0.015),
                'lon': row['lon'] + np.random.uniform(-0.015, 0.015)
            })
    return pd.DataFrame(detail_data)

def prediksi_cuaca_ann(data_lokasi):
    weights = np.array([0.5, -0.2]) 
    bias = 0.1
    hasil_prediksi = []
    for index, row in data_lokasi.iterrows():
        inputs = np.array([row['lat'], row['lon']])
        summation = np.dot(inputs, weights) + bias
        probabilitas = 1 / (1 + np.exp(-abs(summation) / 10)) 
        intensitas = probabilitas * 1000 + np.random.randint(-100, 100)
        status = "Hujan Lebat" if intensitas > 750 else "Mendung" if intensitas > 450 else "Cerah"
        hasil_prediksi.append({'size': max(10, intensitas), 'status': status})
    return pd.DataFrame(hasil_prediksi)

def prediksi_cuaca_ann(df):
    # Simulasi logika ANN yang lebih seimbang
    # Kita gunakan distribusi normal agar tidak semua hasilnya ekstrem
    np.random.seed(42) # Agar hasil konsisten
    
    # Membuat variasi nilai dasar antara 20 - 70 (tidak langsung 100)
    base_prediction = np.random.normal(loc=50, scale=15, size=len(df))
    
    # Memastikan nilai tetap di rentang 0-100
    base_prediction = np.clip(base_prediction, 10, 95)
    
    return pd.DataFrame({'size': base_prediction})

def prediksi_cuaca_masa_depan(data_lokasi, faktor):
    hasil = prediksi_cuaca_ann(data_lokasi)
    hasil['size'] = hasil['size'] * faktor * np.random.uniform(0.8, 1.2)
    hasil['status'] = hasil['size'].apply(lambda s: "Hujan Lebat" if s > 750 else "Mendung" if s > 450 else "Cerah")
    return hasil

def generate_detail_jalan(data_induk, jumlah):
    detail_data = []
    for _, row in data_induk.iterrows():
        for i in range(jumlah):
            detail_data.append({
                'city': f"Jalur {row['city']}", 
                'lat': row['lat'] + np.random.uniform(-0.03, 0.03),
                'lon': row['lon'] + np.random.uniform(-0.03, 0.03)
            })
    return pd.DataFrame(detail_data)

def prediksi_kemacetan_ann(data_lokasi, faktor_waktu):
    weights = np.array([0.6, -0.3]) 
    bias = 0.2
    hasil_prediksi = []
    for index, row in data_lokasi.iterrows():
        inputs = np.array([row['lat'], row['lon']])
        summation = np.dot(inputs, weights) + bias
        kepadatan = (abs(summation) % 1) * 100 * faktor_waktu
        kepadatan = min(100, kepadatan + random.randint(0, 20))
        if kepadatan > 80:
            status, warna = "Macet Total", [255, 0, 0, 200]
        elif kepadatan > 40:
            status, warna = "Padat Merayap", [255, 255, 0, 200]
        else:
            status, warna = "Lancar", [0, 255, 0, 200]
        hasil_prediksi.append({'density': kepadatan, 'status': status, 'color': warna})
    return pd.DataFrame(hasil_prediksi)
# --- 1. FUNGSI LOGIKA ANN ---

# ... fungsi yang sudah ada ...
# Menentukan status yang lebih variatif
def get_status(load):
    if load > 90: 
        return "🔴 Critical Overload"
    elif load > 70: 
        return "🟠 Heavy Traffic"
    elif load > 40: 
        return "🟡 Busy / Partly Cloudy" # Ini akan memunculkan status "Berawan"
    else: 
        return "🟢 Optimal / Clear Sky"

def hitung_faktor_kemacetan(jam_str):
    h = int(jam_str.split(":")[0])
    if 7 <= h <= 9: return 2.8   # Sibuk Pagi
    if 16 <= h <= 19: return 3.2  # Sibuk Sore
    if 22 <= h or h <= 4: return 0.5 # Tengah Malam (Lancar)
    return 1.5 # Jam Reguler

# --- 2. CSS & STATE ---
st.markdown("""
    <style>
    .block-container { padding: 0rem; }
    .floating-menu {
        position: absolute; top: 20px; left: 50px; z-index: 999;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

if "mode" not in st.session_state:
    st.session_state.mode = "none"

# --- 3. UI FLOATING MENU ---

with st.container():
    st.markdown('<br>', unsafe_allow_html=True)
    # Buat kolom. Saya sesuaikan rasionya agar lebih rapat
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 0.5, 1])
    
    with col1:
        if st.button("🌦️", key="btn_weather", help="Weather Forecasting"):
            st.session_state.mode = "weather"
            st.rerun()
            
    with col2:
        if st.button("🚀", key="btn_traffic", help="Traffic Analysis"):
            st.session_state.mode = "traffic"
            st.rerun()

    with col3:
        if st.button("📡", key="btn_signal", help="Signal Coverage"):
            st.session_state.mode = "signal"
            st.rerun()
            
    with col4:
        # Garis pemisah kustom
        st.markdown('<div class="menu-divider"></div>', unsafe_allow_html=True)

    with col5:
        if st.button("🧹", key="btn_reset", help="Clear Analysis"):
            st.session_state.mode = "none"
            st.rerun()

wilayah_utama = pd.DataFrame({
    'city': ['Jakarta Barat', 'Jakarta Selatan', 'Jakarta Pusat', 'Jakarta Timur'],
    'lat': [-6.1683, -6.2615, -6.1818, -6.2256],
    'lon': [106.7588, 106.8106, 106.8229, 106.9004]
})

# --- 4. MAPS LOGIC ---

# --- MODE CUACA ---
if st.session_state.mode == "weather":
    st.sidebar.write("### ⏳ Kontrol Cuaca")
    pilihan_waktu = st.sidebar.select_slider("Prediksi Waktu:", options=["Sekarang", "1 Jam", "3 Jam", "6 Jam", "Besok"])
    level_detail = st.sidebar.select_slider("Detail:", options=["Kota", "Kecamatan", "Kelurahan"])
    
    pengali = {"Sekarang": 1.0, "1 Jam": 1.1, "3 Jam": 1.2, "6 Jam": 1.3, "Besok": 1.5}
    
    if level_detail == "Kota": data_input = wilayah_utama; zoom = 10.5
    elif level_detail == "Kecamatan": data_input = generate_detail_wilayah(wilayah_utama, 12); zoom = 12
    else: data_input = generate_detail_wilayah(wilayah_utama, 40); zoom = 13.5

    hasil_ann = prediksi_cuaca_masa_depan(data_input, pengali[pilihan_waktu])
    df_weather = pd.concat([data_input, hasil_ann], axis=1)

    layer = pdk.Layer("HeatmapLayer", df_weather, get_position='[lon, lat]', get_weight='size', radiusPixels=40)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=-6.2088, longitude=106.8456, zoom=zoom)))
    
    st.write(f"### 📍 Prediksi Cuaca: **{pilihan_waktu}**")
    
    # Tabel Rekomendasi
    # Menghitung persentase keyakinan ANN (Simulasi dari nilai intensitas/size)
    # Kita asumsikan 1000 adalah batas maksimal intensitas (100%)
    df_weather['Persen Prediksi'] = df_weather['size'].apply(lambda x: f"{min(100, x/10):.1f}%")

    # Menambahkan Logika Rekomendasi
    df_weather['Rekomendasi'] = df_weather['status'].apply(
        lambda s: "⚠️ Tunda" if s == "Hujan Lebat" else "☁️ Siap Mantel" if s == "Mendung" else "✅ Aman"
    )

    # Menampilkan Tabel dengan kolom Persen Prediksi
    st.table(df_weather[['city', 'status', 'Persen Prediksi', 'Rekomendasi']].head(10))

# --- MODE TRAFFIC ---
elif st.session_state.mode == "traffic":
    # 1. Perbaikan freq="h" (huruf kecil) untuk Pandas versi terbaru
    daftar_jam = [t.strftime("%H:%M") for t in pd.date_range("00:00", "23:59", freq="h")]
    
    st.sidebar.write("### 🕒 Kontrol Traffic")
    
    # 2. Gunakan satu slider saja agar tidak bingung
    jam_pilihan = st.sidebar.select_slider("Pilih Jam:", options=daftar_jam, value="09:00")
    
    # 3. Hitung faktor menggunakan fungsi yang sudah dibuat
    faktor = hitung_faktor_kemacetan(jam_pilihan)
    
    # 4. Jalankan ANN Kemacetan
    data_jalan = generate_detail_jalan(wilayah_utama, 50)
    hasil_traffic = prediksi_kemacetan_ann(data_jalan, faktor)
    df_traffic = pd.concat([data_jalan, hasil_traffic], axis=1)

    # 5. Visualisasi
    layer_t = pdk.Layer(
        "ScatterplotLayer", 
        df_traffic, 
        get_position='[lon, lat]', 
        get_color='color', 
        get_radius=150,
        pickable=True
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer_t], 
        initial_view_state=pdk.ViewState(latitude=-6.2088, longitude=106.8456, zoom=11.5),
        map_style='mapbox://styles/mapbox/dark-v9',
        tooltip={"text": "Lokasi: {city}\nStatus: {status}\nKepadatan: {density:.1f}%"}
    ))
    
    # Format kolom density menjadi string dengan lambang persen
    df_traffic['Tingkat Kepadatan'] = df_traffic['density'].apply(lambda x: f"{x:.1f}%")

    st.write(f"### 🚦 Status Kemacetan Jam: **{jam_pilihan}**")
    
    # Menampilkan Tabel dengan kolom Tingkat Kepadatan
    st.table(df_traffic[['city', 'status', 'Tingkat Kepadatan']].head(10))

# --- MODE SINYAL JARINGAN (HEATMAP) ---
# --- MODE SINYAL JARINGAN (VISUALISASI TOWER & BEBAN) ---
# --- MODE SINYAL JARINGAN (CAKUPAN DKI JAKARTA) ---
# --- MODE SINYAL JARINGAN (CAKUPAN DKI JAKARTA) ---
# --- MODE SINYAL JARINGAN (SOLID AREA COVERAGE) ---
# --- MODE SINYAL JARINGAN (SMOOTH RADAR COVERAGE) ---
elif st.session_state.mode == "signal":
    st.sidebar.markdown("### 📡 Network Configuration")
    
    # 1. AMBIL INPUT USER TERLEBIH DAHULU
    net_type = st.sidebar.selectbox("Connection Type:", ["Physical SIM", "eSIM", "Satellite Link"])
    net_speed = st.sidebar.select_slider(
        "Network Strength:",
        options=["Emergency", "3G", "4G", "5G"],
        value="4G"
    )

    # Logika multiplier berdasarkan input
    multiplier = {"5G": 1.5, "4G": 1.0, "3G": 0.7, "Emergency": 0.3}[net_speed]

    # 2. PROSES DATA & ANN
    lat_range = np.linspace(-6.38, -6.07, 35) 
    lon_range = np.linspace(106.67, 106.99, 35)
    lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
    
    df_grid = pd.DataFrame({
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'city': [f"BTS-{i}" for i in range(len(lat_grid.flatten()))]
    })

    # Prediksi ANN
    hasil_ann = prediksi_cuaca_ann(df_grid)
    df_signal = pd.concat([df_grid, hasil_ann], axis=1)

    # HITUNG KOLOM BARU (Penting: Harus sebelum membuat Layer)
    df_signal['Network_Load'] = (df_signal['size'] * multiplier).apply(lambda x: min(100, x))
    df_signal['Signal_Type'] = net_speed
    df_signal['Sim_Type'] = net_type
    adj_multiplier = {"5G": 1.2, "4G": 0.8, "3G": 0.5, "Emergency": 0.2}[net_speed]
    
    df_signal['Network_Load'] = (df_signal['size'] * adj_multiplier)
    
    # Menambahkan fluktuasi sedikit agar ada area yang "Berawan/Optimal"
    noise = np.random.uniform(-5, 5, len(df_signal))
    df_signal['Network_Load'] = (df_signal['Network_Load'] + noise).clip(0, 100)

    # 3. VISUALISASI MAP
    layer_smooth = pdk.Layer(
        "HeatmapLayer",
        df_signal,
        get_position='[lon, lat]',
        get_weight='Network_Load', # Sekarang kolom ini sudah ada
        radiusPixels=140,
        intensity=0.9,
        threshold=0.01,
        colorRange=[
            [63, 0, 113, 100],   # Area Stabil
            [127, 0, 255, 150], 
            [255, 0, 255, 200], 
            [255, 0, 100, 255],  # Area Overload
        ]
    )

    
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer_smooth],
        initial_view_state=pdk.ViewState(
            latitude=-6.2100, 
            longitude=106.8300, 
            zoom=10.2, 
            pitch=0 
        ),
        map_style='mapbox://styles/mapbox/dark-v9'
    ))
    st.write(f"### 🌐 {net_speed} Coverage Map ({net_type})")
    # 4. TABEL DETAIL
    st.write("### 📊 Network Analytics Detail")
    
    def get_status(load):
        if load > 80: return "🔴 Critical Overload"
        if load > 50: return "🟡 Heavy Traffic"
        return "🟢 Optimal Condition"

    # Ambil 10 data terbesar untuk tabel
    df_detail = df_signal.nlargest(10, 'Network_Load').copy()
    df_detail['Status'] = df_detail['Network_Load'].apply(get_status)
    df_detail['Latency (ms)'] = df_detail['Network_Load'].apply(lambda x: int(x * 0.5 + 20))

    st.table(df_detail[[
        'city', 'Signal_Type', 'Sim_Type', 'Network_Load', 'Latency (ms)', 'Status'
    ]].rename(columns={
        'city': 'Tower ID', 
        'Network_Load': 'Load (%)',
        'Signal_Type': 'Band',
        'Sim_Type': 'Platform'
    }))

# --- TAMPILAN AWAL ---
else:
    st.map(pd.DataFrame({'lat': [-6.2088], 'lon': [106.8456]}), zoom=11)

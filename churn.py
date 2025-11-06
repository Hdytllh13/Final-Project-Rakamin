import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE
import shap
from io import BytesIO, StringIO

# =====================================================
# 🎨 STYLE & UI THEME
# =====================================================
st.set_page_config(page_title="Employee Churn Prediction", page_icon="💼", layout="wide")

st.markdown("""
<style>
h2, h3, h4 { color: #2C3E50; }
[data-testid="stMetricValue"] { color: #1E8449; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#E5E8E8",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "legend.fontsize": 8
})

# =====================================================
# 💾 MUAT MODEL
# =====================================================
class StrongHybridSampler:
    def __init__(self, random_state=42, target_factor=1.0, max_samples_per_class=None):
        self.random_state = random_state
        self.target_factor = target_factor
        self.max_samples_per_class = max_samples_per_class

    def _make_sampling_strategy(self, y):
        ctr = Counter(y)
        max_count = max(ctr.values())
        target = {}
        for cls, c in ctr.items():
            desired = int(max_count * self.target_factor)
            if desired <= c:
                continue
            if self.max_samples_per_class:
                desired = min(desired, self.max_samples_per_class)
            target[cls] = desired
        return target

    def fit_resample(self, X, y):
        sampling_strategy = self._make_sampling_strategy(y)
        if not sampling_strategy:
            return X, y
        try:
            sampler = BorderlineSMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)
            X_res, y_res = sampler.fit_resample(X, y)
        except Exception:
            sampler = SMOTEENN(random_state=self.random_state, sampling_strategy=sampling_strategy)
            X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res


@st.cache_resource
def load_models():
    churn_model = joblib.load("churn_model.pkl")
    churn_period_model = joblib.load("churn_period_model.pkl")
    churn_period_features = joblib.load("churn_period_features.pkl")
    scaler = joblib.load("scaler.pkl")
    return churn_model, churn_period_model, churn_period_features, scaler


churn_model, churn_period_model, churn_period_features, scaler = load_models()

# =====================================================
# 🧩 PREPROCESSING FUNGSI
# =====================================================
def _ensure_numeric_df(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def preprocess_batch_robust(df):
    df = df.copy()
    id_cols = [c for c in ['employee_id', 'employee_name'] if c in df.columns]
    id_data = df[id_cols].copy() if id_cols else pd.DataFrame()

    required_cols = [
        'target_achievement', 'company_tenure_years', 'distance_to_office_km',
        'job_satisfaction', 'manager_support_score', 'marital_status', 'working_hours_per_week'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"⚠️ Kolom hilang dalam file input: {missing}")

    df['achieve_status'] = df['target_achievement']
    df['promotion_potential'] = (
        (df['job_satisfaction'] + 2 * df['target_achievement']) * np.log(df['company_tenure_years'] + 1)
    )
    df['marital_status'] = df['marital_status'].apply(lambda x: 0 if str(x).lower() in ('married', 'm') else 1)

    selected = [
        'achieve_status', 'company_tenure_years', 'distance_to_office_km',
        'working_hours_per_week', 'manager_support_score', 'marital_status',
        'target_achievement', 'promotion_potential'
    ]
    X = _ensure_numeric_df(df[selected])

    if hasattr(scaler, 'feature_names_in_'):
        trained_feats = list(scaler.feature_names_in_)
        X = X.reindex(columns=trained_feats, fill_value=0)

    X_scaled = scaler.transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    if not id_data.empty:
        df_scaled = pd.concat([id_data.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
    return df_scaled


def align_for_model(X, model):
    if not hasattr(model, "feature_names_in_"):
        return X
    trained_feats = list(model.feature_names_in_)
    X_aligned = pd.DataFrame(0.0, index=X.index, columns=trained_feats)
    for col in trained_feats:
        if col in X.columns:
            X_aligned[col] = X[col].astype(float)
    return X_aligned


def predict_batch_safe(df_raw):
    X_scaled = preprocess_batch_robust(df_raw)
    id_cols = [c for c in ['employee_id', 'employee_name'] if c in X_scaled.columns]
    X_for_churn = align_for_model(X_scaled.drop(columns=id_cols, errors='ignore'), churn_model)

    churn_pred = churn_model.predict(X_for_churn).astype(int)
    churn_period_pred = []
    for i, c in enumerate(churn_pred):
        if int(c) == 0:
            churn_period_pred.append(0)
            continue
        Xi_cp = align_for_model(X_for_churn.iloc[[i]], churn_period_model)
        try:
            p = int(churn_period_model.predict(Xi_cp)[0]) + 1
        except Exception:
            p = 1
        churn_period_pred.append(p)

    df_result = df_raw.copy().reset_index(drop=True)
    df_result['churn_pred'] = churn_pred
    df_result['churn_period_pred'] = churn_period_pred
    mapping = {0: "Stayed", 1: "Onboarding", 2: "1 Month", 3: "3 Months"}
    df_result['churn_period_label'] = df_result['churn_period_pred'].map(mapping)
    df_result['churn_label_final'] = df_result['churn_pred'].map({0: "No Churn", 1: "Churn"})

    if id_cols:
        cols = id_cols + [c for c in df_result.columns if c not in id_cols]
        df_result = df_result[cols]
    return df_result

# =====================================================
# 🧍 TAB 1: Prediksi Individu
# =====================================================
tab1, tab2 = st.tabs(["🧍 Prediksi Individu", "📂 Prediksi Batch"])

with tab1:
    st.subheader("Prediksi Churn untuk 1 Karyawan")

    col1, col2 = st.columns(2)
    with col1:
        target_achievement = st.number_input("🎯 Target Achievement", 0.0, 2.0, 1.0, 0.01)
        company_tenure_years = st.number_input("🧭 Lama bekerja (tahun)", 0.0, 40.0, 3.0, 0.1)
    with col2:
        distance_to_office_km = st.number_input("📍 Jarak ke kantor (km)", 0.0, 60.0, 5.0, 0.1)
        working_hours_per_week = st.number_input("⌚ Jam kerja/minggu", 20, 80, 40)

    marital_status = st.selectbox("💍 Status Pernikahan", ["Married", "Single"])
    job_satisfaction = st.slider("😊 Kepuasan kerja (1-5)", 1, 5, 3)
    manager_support_score = st.slider("🤝 Dukungan manajer (1-5)", 1, 5, 3)

    if st.button("🔮 Jalankan Prediksi Individu"):
        df_input = pd.DataFrame([{
            "target_achievement": target_achievement,
            "company_tenure_years": company_tenure_years,
            "distance_to_office_km": distance_to_office_km,
            "job_satisfaction": job_satisfaction,
            "manager_support_score": manager_support_score,
            "marital_status": marital_status,
            "working_hours_per_week": working_hours_per_week
        }])
        X_scaled = preprocess_batch_robust(df_input)
        pred = churn_model.predict(X_scaled)[0]
        prob = churn_model.predict_proba(X_scaled)[0][1] * 100 if hasattr(churn_model, "predict_proba") else np.nan
        st.write(f"📈 Probabilitas churn: **{prob:.1f}%**")
        st.success("✅ Tidak akan churn" if pred == 0 else "🚨 Potensi churn tinggi")

# =====================================================
# 📂 TAB 2: Batch Upload & Analisis
# =====================================================
with tab2:
    st.subheader("📂 Prediksi Banyak Karyawan (Batch Upload)")
    uploaded = st.file_uploader("Upload file CSV / Excel", type=["csv", "xlsx"])

    if uploaded:
        df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        df_result = predict_batch_safe(df_raw)

        st.success(f"✅ File berhasil dimuat ({len(df_result)} baris)")

        churn_counts = df_result["churn_label_final"].value_counts().reindex(["Churn", "No Churn"]).fillna(0)
        period_counts = df_result["churn_period_label"].value_counts().reindex(["Onboarding", "1 Month", "3 Months", "Stayed"]).fillna(0)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%", startangle=90, colors=["#E74C3C", "#2ECC71"])
            ax.set_title("Distribusi Churn")
            st.pyplot(fig, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(period_counts.index, period_counts.values, color=["#F39C12", "#E67E22", "#D35400", "#3498DB"])
            ax.set_title("Distribusi Periode Churn")
            st.pyplot(fig, use_container_width=True)

        # Summary & Insight
        st.markdown("---")
        st.markdown("## 🧾 Ringkasan Hasil Prediksi Batch")

        total = len(df_result)
        churn_total = df_result["churn_pred"].sum()
        churn_pct = churn_total / total * 100
        period_top = period_counts.idxmax()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total Karyawan", total)
        col2.metric("🚪 Prediksi Churn", churn_total, f"{churn_pct:.1f}%")
        col3.metric("🧭 Rata-rata Tenure", f"{df_result['company_tenure_years'].mean():.1f} tahun")
        col4.metric("🔥 Periode Churn Tertinggi", period_top)

        if churn_pct > 50:
            st.error(f"Tingkat churn **sangat tinggi ({churn_pct:.1f}%)** — perhatikan kepuasan kerja dan dukungan manajer.")
        elif churn_pct > 30:
            st.warning(f"Tingkat churn **menengah ({churn_pct:.1f}%)** — fokus pada karyawan baru.")
        else:
            st.success(f"Tingkat churn **rendah ({churn_pct:.1f}%)** — strategi retensi efektif.")

        # Segmentasi Risiko & Faktor Penting
        st.markdown("---")
        st.markdown("### 🎯 Segmentasi Risiko & 💡 Analisis Faktor Penting")
        col_risk, col_faktor = st.columns(2)

        with col_risk:
            st.markdown("#### 🎯 Segmentasi Risiko Karyawan")
            if hasattr(churn_model, "predict_proba"):
                churn_probs = churn_model.predict_proba(align_for_model(preprocess_batch_robust(df_raw).drop(columns=['employee_id','employee_name'], errors='ignore'), churn_model))[:, 1]
                df_result["churn_prob"] = churn_probs
                bins = [0, 0.33, 0.66, 1.0]
                labels = ["Low", "Medium", "High"]
                df_result["risk_level"] = pd.cut(df_result["churn_prob"], bins=bins, labels=labels)
                risk_counts = df_result["risk_level"].value_counts().reindex(labels, fill_value=0)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.bar(risk_counts.index, risk_counts.values, color=["#2ECC71", "#F1C40F", "#E74C3C"])
                ax.set_title("Distribusi Risiko Churn")
                st.pyplot(fig, use_container_width=True)
                if risk_counts["High"] > 0:
                    st.warning(f"⚠️ {risk_counts['High']} karyawan berisiko tinggi churn.")
            else:
                st.info("Model tidak mendukung probabilitas churn (predict_proba).")

        with col_faktor:
            st.markdown("#### 💡 Analisis Faktor Penting terhadap Churn")
            faktor_mean_orig = df_result.groupby("churn_label_final")[["job_satisfaction", "manager_support_score", "target_achievement"]].mean().T
            rename_map = {"job_satisfaction": "Job Satisfaction", "manager_support_score": "Manager Support", "target_achievement": "Target\nAchievement"}
            faktor_mean = faktor_mean_orig.rename(index=rename_map)

            fig, ax = plt.subplots(figsize=(6, 4))
            faktor_mean.plot(kind="bar", ax=ax, width=0.65, color=["#3498DB", "#F39C12"])
            ax.set_title("Perbandingan Faktor Rata-rata: Churn vs No Churn")
            ax.set_ylabel("Rata-rata Skor")
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig, use_container_width=True)

        # Download hasil
        def to_excel(df):
            output = BytesIO()
            try:
                import openpyxl
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Prediksi")
            except Exception:
                import xlsxwriter
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Prediksi")
            return output.getvalue()

        excel_data = to_excel(df_result)
        st.download_button("📥 Download Hasil Prediksi (Excel)", data=excel_data,
                           file_name="hasil_prediksi_churn.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # SHAP Analisis
        st.markdown("---")
        st.subheader("🔍 Analisis Faktor Risiko (SHAP)")
        try:
            X_scaled = preprocess_batch_robust(df_raw)
            X_for_churn = align_for_model(X_scaled, churn_model)
            final_model = churn_model
            explainer = shap.Explainer(final_model, X_for_churn)
            shap_values = explainer(X_for_churn)
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.summary_plot(shap_values, X_for_churn, plot_type="bar", show=False)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Analisis SHAP gagal: {e}")

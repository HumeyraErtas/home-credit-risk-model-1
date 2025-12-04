import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------
# 1) MODELI VE FE DATAYI YÜKLE
# -------------------------

@st.cache_resource
def load_model_and_template():
    # Build absolute paths relative to this file so the app works
    # regardless of the current working directory.
    base_dir = Path(__file__).resolve().parent.parent

    # Final modeli yükle (mutlak path kullan)
    model_path = base_dir / "models" / "final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}.\nMake sure the file exists or update the path."
        )
    model = joblib.load(model_path)

    # FE'li dataset'i yükle (sadece medyan ve kolon bilgisi için)
    df_fe_path = base_dir / "data" / "processed" / "train_fe.csv"
    if not df_fe_path.exists():
        raise FileNotFoundError(f"FE dataset not found: {df_fe_path}")
    df_fe = pd.read_csv(df_fe_path)

    # TARGET'ı at, sadece feature'lar kalsın
    feature_cols = [c for c in df_fe.columns if c != "TARGET"]

    # Build a safe template row: numeric features -> median, non-numeric -> mode (most frequent)
    numeric_cols = df_fe[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in feature_cols if c not in numeric_cols]

    template_values = {}
    if numeric_cols:
        medians = df_fe[numeric_cols].median()
        template_values.update(medians.to_dict())

    for c in non_numeric_cols:
        # use mode (most frequent). If mode is empty (all NaN), fall back to first value or None
        modes = df_fe[c].mode(dropna=True)
        if not modes.empty:
            template_values[c] = modes.iloc[0]
        else:
            # fallback: take first non-null value if exists, else None
            non_nulls = df_fe[c].dropna()
            template_values[c] = non_nulls.iloc[0] if len(non_nulls) > 0 else None

    # Ensure ordering matches feature_cols and create a single-row DataFrame
    template_row = pd.DataFrame([template_values])[feature_cols]

    return model, feature_cols, template_row

model, feature_cols, template_row = load_model_and_template()


# -------------------------
# 2) YARDIMCI FONKSİYONLAR
# -------------------------

def build_single_input_row(
    feature_cols,
    template_row,
    amt_income,
    amt_credit,
    amt_annuity,
    cnt_fam_members,
    age_years,
    emp_years
):
    """
    Kullanıcının girdiği birkaç temel feature'a göre,
    train_fe yapısına uygun tek satırlık bir DataFrame oluşturur.
    Diğer kolonlar, medyan (template_row) değerleri ile doldurulur.
    """

    row = template_row.copy()  # tüm feature'lar dolu (median)

    # Ham feature'ları override et
    if "AMT_INCOME_TOTAL" in feature_cols:
        row["AMT_INCOME_TOTAL"] = amt_income

    if "AMT_CREDIT" in feature_cols:
        row["AMT_CREDIT"] = amt_credit

    if "AMT_ANNUITY" in feature_cols:
        row["AMT_ANNUITY"] = amt_annuity

    if "CNT_FAM_MEMBERS" in feature_cols:
        row["CNT_FAM_MEMBERS"] = cnt_fam_members

    # DAYS_BIRTH / DAYS_EMPLOYED FE kodunda pozitifti (abs aldık),
    # o yüzden burada da pozitif gün cinsinden hesaplayalım.
    days_birth = age_years * 365
    if "DAYS_BIRTH" in feature_cols:
        row["DAYS_BIRTH"] = days_birth

    if emp_years is not None:
        days_emp = emp_years * 365
        if "DAYS_EMPLOYED" in feature_cols:
            row["DAYS_EMPLOYED"] = days_emp

    # FE sırasında ürettiğimiz kolonları da güncelleyelim:
    # AGE
    if "AGE" in feature_cols:
        row["AGE"] = age_years

    # Log transform'lar
    if "AMT_INCOME_TOTAL_LOG" in feature_cols:
        row["AMT_INCOME_TOTAL_LOG"] = np.log1p(amt_income)

    if "AMT_CREDIT_LOG" in feature_cols:
        row["AMT_CREDIT_LOG"] = np.log1p(amt_credit)

    if "AMT_ANNUITY_LOG" in feature_cols:
        row["AMT_ANNUITY_LOG"] = np.log1p(amt_annuity)

    # Oranlar
    if "DEBT_INCOME_RATIO" in feature_cols:
        row["DEBT_INCOME_RATIO"] = amt_credit / (amt_income + 1)

    if "CREDIT_ANNUITY_RATIO" in feature_cols:
        row["CREDIT_ANNUITY_RATIO"] = amt_credit / (amt_annuity + 1)

    if "INCOME_PER_PERSON" in feature_cols:
        row["INCOME_PER_PERSON"] = amt_income / (cnt_fam_members + 1)

    if "PAYMENT_RATE" in feature_cols:
        row["PAYMENT_RATE"] = amt_annuity / (amt_credit + 1)

    # Geri kalan tüm feature'lar median değer olarak kalıyor.
    # Model, bu tek satırı XFeature yapısında bekliyor.
    return row[feature_cols]


def predict_proba_single(row_df):
    proba = model.predict_proba(row_df)[:, 1][0]  # riskli sınıf olasılığı
    return proba


# -------------------------
# 3) STREAMLIT ARAYÜZÜ
# -------------------------

st.set_page_config(page_title="Home Credit Risk Model", page_icon="💳", layout="centered")

st.title("💳 Home Credit Default Risk – Tahmin Uygulaması")
st.write(
    """
    Bu arayüz, Zero2End ML Bootcamp final projesi kapsamında geliştirdiğin 
    **kredi geri ödememe riski** modelini kullanır. 
    
    Aşağıdan tek bir müşteri için tahmin alabilir veya FE'li bir CSV dosyası 
    yükleyerek toplu tahmin yapabilirsin.
    """
)

mode = st.sidebar.radio(
    "Mod Seç:",
    ("Tekil Tahmin (Form ile)", "Toplu Tahmin (FE'li CSV yükle)")
)

# -------------------------
# MOD 1: TEKİL TAHMİN
# -------------------------
if mode == "Tekil Tahmin (Form ile)":
    st.subheader("🔹 Tek Müşteri İçin Risk Tahmini")

    col1, col2 = st.columns(2)

    with col1:
        amt_income = st.number_input(
            "Aylık Gelir (AMT_INCOME_TOTAL)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=150_000.0,
            step=1_000.0
        )
        amt_credit = st.number_input(
            "Kredi Tutarı (AMT_CREDIT)",
            min_value=0.0,
            max_value=2_000_000.0,
            value=500_000.0,
            step=5_000.0
        )
        amt_annuity = st.number_input(
            "Aylık Taksit (AMT_ANNUITY)",
            min_value=0.0,
            max_value=200_000.0,
            value=25_000.0,
            step=500.0
        )

    with col2:
        cnt_fam_members = st.number_input(
            "Aile Üye Sayısı (CNT_FAM_MEMBERS)",
            min_value=0,
            max_value=20,
            value=3,
            step=1
        )
        age_years = st.number_input(
            "Yaş (Yıl)",
            min_value=18,
            max_value=90,
            value=35,
            step=1
        )
        emp_years = st.number_input(
            "Toplam Çalışma Süresi (Yıl)",
            min_value=0,
            max_value=60,
            value=5,
            step=1
        )

    if st.button("Tahmin Et"):
        # Tek satırlık input DF'ini oluştur
        input_row = build_single_input_row(
            feature_cols,
            template_row,
            amt_income,
            amt_credit,
            amt_annuity,
            cnt_fam_members,
            age_years,
            emp_years
        )

        proba = predict_proba_single(input_row)

        st.markdown("---")
        st.write(f"**Modelin Tahmin Ettiği Geri Ödememe Riski (TARGET=1) Olasılığı:**")
        st.markdown(f"### 🎯 %{proba * 100:.2f}")

        # Basit yorum
        if proba < 0.2:
            st.success("Bu müşteri düşük risk segmentinde görünüyor.")
        elif proba < 0.5:
            st.warning("Bu müşteri orta risk segmentinde. Ek kontrol gerekebilir.")
        else:
            st.error("Bu müşteri yüksek risk segmentinde. Daha dikkatli değerlendirilmelidir.")


# -------------------------
# MOD 2: TOPLU TAHMİN (CSV)
# -------------------------
else:
    st.subheader("📂 Toplu Tahmin – FE'li Dataset ile")

    st.write(
        """
        Burada, **03_feature_engineering notebook'unun ürettiği yapıya uygun** 
        FE'li bir CSV dosyasını (ör: `train_fe.csv`'e benzer) yükleyip, 
        çok sayıda müşteri için toplu risk tahmini alabilirsin.
        
        - Dosyanın `TARGET` kolonu **olmayabilir** (veya varsa yok sayılır).
        - Kolon isimlerinin, eğitimde kullanılan feature isimleriyle 
          (train_fe'deki `TARGET` hariç kolonlar) uyumlu olması gerekir.
        """
    )

    uploaded_file = st.file_uploader("FE'li CSV dosyasını yükle", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Yüklenen veri boyutu:", df_input.shape)

        # Varsa TARGET'ı at
        if "TARGET" in df_input.columns:
            df_input = df_input.drop(columns=["TARGET"])

        # Eğitimde kullanılan feature kolonlarına göre hizala
        missing_cols = [c for c in feature_cols if c not in df_input.columns]
        extra_cols = [c for c in df_input.columns if c not in feature_cols]

        if missing_cols:
            st.error(
                f"Aşağıdaki kolonlar eksik, model bu kolonları bekliyor:\n{missing_cols[:20]}"
            )
        else:
            # Sadece modelin beklediği kolonları kullan
            df_input = df_input[feature_cols]

            proba_batch = model.predict_proba(df_input)[:, 1]
            df_result = df_input.copy()
            df_result["RISK_PROBA"] = proba_batch

            st.success("Tahminler tamamlandı!")
            st.write(df_result.head())

            # İndirme linki
            csv_out = df_result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Sonuçları CSV olarak indir",
                data=csv_out,
                file_name="predictions_with_risk.csv",
                mime="text/csv"
            )

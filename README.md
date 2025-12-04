# 🏦 Home Credit Default Risk – Uçtan Uca Makine Öğrenmesi Projesi  
Zero2End Machine Learning Bootcamp Final Projesi  
---

## 📌 1. Problem Tanımı
Kredi veren finans kuruluşları, başvuru yapan bireylerin gelecekte kredi geri ödemelerinde sorun yaşayıp yaşamayacağını doğru tahmin etmek zorundadır. Yanlış olumlu kararlar (riskli müşteriye kredi vermek) finansal kayıplara yol açarken, yanlış olumsuz kararlar (risksiz müşteriyi reddetmek) müşteri memnuniyetini düşürür.

Bu proje, **Home Credit** veri setini kullanarak her bir başvuru için **geri ödememe (default) riskini tahmin eden** uçtan uca bir makine öğrenmesi çözümü geliştirmeyi amaçlar.

Bu kapsamda:
- Kapsamlı EDA yapılmış
- Feature engineering uygulanmış
- Optuna/RSCV ile model optimize edilmiş
- ROC-AUC metriği ile değerlendirilmiş
- Streamlit ile arayüz geliştirilmiştir.

---

## 📌 2. Veri Seti  
Kullanılan veri seti Kaggle'ın **Home Credit Default Risk** yarışmasına aittir.  

- **307.511 satır**, **122+ kolon**  
- Gerçek müşteri kredi başvuru verisi  
- Tabular format (CSV)  
- IMBALANCED TARGET (1 sadece %8)

### Hedef Değişken:
- `TARGET = 1`: Ödeme güçlüğü riski yüksek  
- `TARGET = 0`: Normal müşteri  

---

## 📌 3. Validasyon Şeması (Zorunlu Soru)
Veri ciddi derecede dengesiz olduğu için **Stratified Train-Test Split** kullanılmıştır.

- `%20 validation`
- `stratify=TARGET`  
- Sabit `random_state=42`

---

## 📌 4. Baseline Model (Zorunlu Soru)
Minimal ön işleme + LightGBM kullanılarak elde edilen ilk skor:

| Model | ROC-AUC |
|-------|---------|
| Baseline LightGBM | **≈ 0.75** |

Bu skor feature engineering ve optimizasyonun başlangıç referansıdır.

---

## 📌 5. Feature Engineering (Zorunlu Soru – Detaylı)
Feature engineering adımlarımız 4 ana grupta yapılmıştır:

### **A) Core Feature Transformations**
- `DAYS_*` kolonları pozitif değerlere dönüştürüldü  
- `AGE` (yıl cinsinden) üretildi  
- `LOG(AMT_INCOME)`, `LOG(AMT_CREDIT)`, `LOG(AMT_ANNUITY)` uygulandı  

### **B) Financial Ratios**
- `DEBT_INCOME_RATIO = CREDIT / INCOME`  
- `CREDIT_ANNUITY_RATIO = CREDIT / ANNUITY`  
- `INCOME_PER_PERSON`  
- `PAYMENT_RATE = ANNUITY / CREDIT`  

### **C) External Scores Aggregation**
- `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_MAX`

### **D) Other Tables Aggregation (bureau, installments, previous…)**
Projeye dahil edilirse aşağıdaki özet bilgiler üretildi:
- Toplam gecikme günleri
- Ortalama borç
- Limit kullanım oranı
- Taksit ödeme davranışı  
(Not: Bu dosyalar bulunamazsa proje FE'si core FE üzerinden devam eder.)

### 📌 FE Sonucu:
- Başlangıç kolon sayısı: **122**
- FE sonrası kolon sayısı: **134+**

---

## 📌 6. Model Optimizasyonu (Zorunlu Soru)
RandomizedSearchCV ile LightGBM hiperparametre taraması yapılmıştır.

Optimizasyon sonrası validation skorları:

| Model | ROC-AUC |
|-------|---------|
| Baseline | ~0.75 |
| Final LightGBM (Optimized) | **~0.80–0.82** |

---

## 📌 7. Final vs Baseline Farkı (Zorunlu Soru)
Feature engineering ve optimizasyon adımları model performansını anlamlı biçimde artırmıştır:

- **+0.05 – 0.07 ROC-AUC iyileşmesi**
- EXT_SOURCE feature’ları ve finansal oranlar en çok katkı yapan feature’lar olmuştur.

---

## 📌 8. Business Uyumu (Zorunlu Soru)
Model çıktısı **risk skoru** olduğundan bankanın kredi politikalarını doğrudan destekler:

- Yüksek risk → kredi reddi / ek güvence talebi  
- Orta risk → manuel inceleme  
- Düşük risk → hızlı onay  

Modelin açıklanabilirliği (feature importance + SHAP) iş tarafına güven verir.

---

## 📌 9. Monitoring (Zorunlu Soru)
Model canlıya alındığında izlenecek metrikler:

- **Input drift:** AGE, PAYMENT_RATE, DEBT_INCOME_RATIO dağılım değişimleri  
- **Model drift:** Periyodik ROC-AUC kontrolü  
- **Output drift:** Ortalama tahmin değerindeki değişimler  

Drift tespit edilirse model yeniden eğitilir.

---

## 📌 10. Deployment – Streamlit Arayüzü
`streamlit_app.py` kullanıcıların:

- Tek bir müşteri girişi ile risk skoru görmesine  
- FE’li CSV yükleyerek toplu tahmin almasına  

imkan tanır.

Komut:
```bash
streamlit run app/streamlit_app.py

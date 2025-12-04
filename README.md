# Home Credit Risk Model

Kısa açıklama
- Bu repo, Zero2End ML Bootcamp kapsamında hazırlanmış "Home Credit" kredi geri ödememe risk modelinin kaynak kodlarını, ön işleme ve değerlendirme notebook'larını ve basit bir Streamlit uygulamasını içerir.

İçerik (kök dizin)
- `app/streamlit_app.py` — Tekil ve toplu tahmin için Streamlit arayüzü.
- `data/` — Ham ve işlenmiş veri örnekleri (`data/processed/train_fe.csv` dahil).
- `models/` — Eğitilmiş modeller (ör: `final_model.pkl`, `baseline_model.pkl`).
- `notebooks/` — EDA, feature engineering, modelleme ve değerlendirme notebook'ları.
- `src/` — Ön işleme, feature engineering ve eğitim script'leri.

Hızlı başlangıç

1) Ortam kurulumu

Windows PowerShell örneği (conda/venv kullanıyorsanız aktif edin):

```powershell
pip install -r requirements.txt
# Eğer requirements.txt yoksa temel paketleri yükleyin:
pip install streamlit pandas numpy scikit-learn joblib shap matplotlib seaborn
```

2) Streamlit uygulamasını çalıştırma

Projeyi root dizininde çalıştırın:

```powershell
streamlit run app\streamlit_app.py
```

Notlar:
- `app/streamlit_app.py` model ve FE template dosyalarını `models/final_model.pkl` ve `data/processed/train_fe.csv` yollarından yükler. Eğer farklı bir isim veya yol kullanıyorsanız dosyaları uygun şekilde yeniden adlandırın ya da `app/streamlit_app.py` içinde `model_path` ve `df_fe_path` değişkenlerini güncelleyin.
- Uygulama, model dosyası eksikse veya `train_fe.csv` bulunamazsa hatayı açık bir mesajla gösterir.

Notebook'lar
- `notebooks/01_eda.ipynb` — Veri keşfi.
- `notebooks/02_baseline_model.ipynb` — Baseline model.
- `notebooks/03_feature_engineering.ipynb` — Feature engineering adımları.
- `notebooks/04_model_optimization.ipynb` — Model seçimi ve optimizasyon.
- `notebooks/05_final_model_evaluation.ipynb` — Final model değerlendirme, SHAP açıklamaları.

SHAP ile ilgili not
- `05_final_model_evaluation.ipynb` içinde SHAP summary plot oluştururken `explainer.shap_values` çıktısı bazen liste, bazen ndarray olabilir. Notebook'ta bu duruma dayanıklı kod bulunduğu için "Summary plots need a matrix" gibi bir hata alırsanız ilgili hücreyi güncelleyin veya hücreyi tekrar çalıştırın.

Model tekrar eğitme
- Modeli yeniden eğitmek isterseniz `src/train.py` ve ilgili notebook'ları kullanabilirsiniz. Eğitilmiş model `joblib.dump(model, 'models/final_model.pkl')` ile `models/` altında saklanmalıdır.

Geliştirici notları
- Bu repo eğitim amaçlı hazırlanmıştır. Üretim ortamına taşımadan önce veri gizliliği, model doğrulama ve performans testleri yapılmalıdır.

Katkıda bulunma
- PR'ler welcome. Küçük düzeltmeler, README geliştirmeleri ve notebook temizliği için katkı bekleniyor.

Lisans
- Bu repo içinde bir `LICENSE` dosyası bulunuyor. Lisans koşullarına uygun kullanın.

---
Eğer README'da başka bir başlık veya örnek komut isterseniz, belirtin; ben eklerim.

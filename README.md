# Proje 3: Akıllı Veri Analitiği ve Makine Öğrenmesi Uygulaması

Bu proje, AWS SageMaker ve Python (Scikit-Learn) kullanılarak bulut üzerinde bir makine öğrenmesi modelinin uçtan uca (end-to-end) geliştirilmesini, eğitilmesini ve dağıtılmasını göstermektedir.

## Proje Yapısı

Projeye ait dosya ve klasör hiyerarşisi aşağıdaki gibidir:

```text
cloudfinal2/
│
├── main.py                # 1. Pipeline'ı yöneten ana dosya (S3 Yükleme, Eğitim Başlatma, Deploy ve Çıkarım)
├── requirements.txt       # Proje için gerekli olan Python kütüphaneleri (pip install -r requirements.txt)
├── README.md              # Proje açıklamaları ve kullanım talimatları
│
├── data/                  # Veri setlerinin bulunduğu klasör (Eğer yoksa main.py tarafından sentetik olarak üretilir)
│   └── dataset.csv        # AWS S3'e yüklenecek olan veri seti (Gereksinim 1)
│
└── src/                   # Eğitim scriptinin bulunduğu klasör (SageMaker'a gönderilir)
    └── train.py           # 2. SageMaker Eğitim makinesinde çalışacak olan Scikit-Learn model eğitim kodu
```

## Teknik Gereksinimlerin Karşılanması

1. **Veri Kaynağı:** `main.py` dosyası içerisindeki veri yükleme bölümü, belirtilen `dataset.csv` dosyasını `boto3` ve `sagemaker` SDK kullanarak AWS S3 bucket'ına yükler.
2. **Model Geliştirme:** `src/train.py` dosyası, Scikit-learn kütüphanesini kullanarak `RandomForestClassifier` algoritması ile sınıflandırma modeli geliştirir.
3. **Bulut Entegrasyonu:** `main.py` içerisindeki `SKLearn` estimator (tahminleyici) sınıfı, SageMaker üzerinde bir makine kiralayarak (`ml.m5.large`) model eğitimini (training) bulutta gerçekleştirir. Eğitimi biten model `.deploy()` metoduyla bir Endpoint olarak (API) dağıtılır.
4. **Çıktı (Inference):** `main.py` içerisindeki son bölümde, oluşturulan Endpoint'e örnek veriler gönderilerek makine öğrenmesi tahminleri alınır ve bu çıktılar ekrana raporlanarak gösterilir.
5. **Raporlama:** Hoca talebi doğrultusunda, hem `main.py` hem de `src/train.py` içerisine her adımın amacını detaylı bir şekilde açıklayan, profesyonel ve eğitim amaçlı Türkçe yorum satırları eklenmiştir.

## Nasıl Çalıştırılır?

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. AWS CLI üzerinde kimlik bilgilerinizin (AWS Access Key ID, AWS Secret Access Key, Default Region) yapılandırıldığından emin olun:
   ```bash
   aws configure
   ```
3. Projeyi çalıştırın:
   ```bash
   python main.py
   ```
> **Not:** Script çalıştırıldığında S3'e veri yükleyecek, SageMaker üzerinden eğitim makinesi ayağa kaldıracak, modeli eğitecek, API endpointi oluşturacak ve tahmin sonuçlarını yazdıracaktır. İşlem tamamlandıktan sonra AWS faturası oluşmaması için endpoint'i silme onayı isteyecektir.

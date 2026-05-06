"""
PROJE 3: Akıllı Veri Analitiği ve Makine Öğrenmesi Uygulaması
-------------------------------------------------------------------
Bu betik, AWS SageMaker kullanılarak uçtan uca bir makine öğrenmesi
sürecini yönetir. İşlem adımları:
1. Veri Kaynağı: S3 bucket'a veri yükleme.
2. Model Eğitimi: SageMaker üzerinde Scikit-learn modeli eğitme.
3. Bulut Entegrasyonu & Dağıtım: Eğitilen modeli bir SageMaker Endpoint olarak dağıtma.
4. Çıktı & Çıkarım (Inference): Dağıtılan model üzerinden tahminler alma.
"""

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3
import pandas as pd
import os

def main():
    # =====================================================================
    # 1. ORTAM DEĞİŞKENLERİ VE YAPILANDIRMA (CONFIGURATION)
    # =====================================================================
    # AWS oturumunu (session) ve SageMaker rolünü başlatıyoruz.
    print("AWS SageMaker oturumu başlatılıyor...")
    try:
        sagemaker_session = sagemaker.Session()
        # AWS ortamında çalıştırıldığında rolü otomatik alır. 
        # Lokal ortamda çalıştırdığımız için IAM Rolü ARN'sini string olarak tanımladık:
        role = "arn:aws:iam::126458880359:role/SageMakerExecutionRole"
        bucket = sagemaker_session.default_bucket() # Varsayılan S3 bucket'ını alıyoruz.
    except Exception as e:
        print("HATA: AWS kimlik bilgileri bulunamadı veya SageMaker rolü alınamadı.")
        print("Lütfen 'aws configure' ile kimlik bilgilerinizi tanımladığınızdan emin olun.")
        print(f"Hata Detayı: {e}")
        return

    prefix = 'sagemaker/proje3-ml-app' # S3 üzerinde verilerin kaydedileceği klasör yolu.

    print(f"Kullanılan IAM Rolü: {role}")
    print(f"Kullanılan S3 Bucket: {bucket}")

    # =====================================================================
    # 2. VERİ KAYNAĞI: S3'E VERİ YÜKLEME
    # =====================================================================
    # Yerel ortamda bulunan veri setinin S3'e yüklenmesi.
    local_data_path = 'data/dataset.csv'

    # Eğer data klasörü ve dosyası yoksa örnek bir sentetik veri seti oluşturalım:
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(local_data_path):
        print("Yerel veri seti bulunamadı. Örnek bir veri seti (Classification) oluşturuluyor...")
        from sklearn.datasets import make_classification
        # 1000 satırlı, 5 özellikli (feature) sentetik bir veri kümesi oluşturuyoruz
        X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        df.to_csv(local_data_path, index=False)

    print(f"Veri S3'e yükleniyor... Yüklenen S3 Yolu: s3://{bucket}/{prefix}/data")
    
    # 'upload_data' fonksiyonu yerel dosyayı alır ve belirttiğimiz S3 bucket'ına yükler.
    train_input = sagemaker_session.upload_data(local_data_path, bucket=bucket, key_prefix=f'{prefix}/data')
    print(f"Veri başarıyla S3'e yüklendi: {train_input}")

    # =====================================================================
    # 3. BULUT ENTEGRASYONU VE MODEL GELİŞTİRME (TRAINING)
    # =====================================================================
    # Scikit-learn Estimator tanımı. 
    # 'src/train.py' dosyası SageMaker'ın başlatacağı eğitim sunucusunda (instance) çalıştırılacak kodumuzdur.
    print("SageMaker Scikit-learn Estimator (Eğitim Modeli) yapılandırılıyor...")
    sklearn_estimator = SKLearn(
        entry_point='train.py',       # Eğitim işlemlerini yapan asıl script
        source_dir='src',             # Scriptin bulunduğu kaynak klasör
        role=role,                    # Eğitimi yapacak olan IAM rolü
        instance_count=1,             # Eğitim için kullanılacak makine sayısı
        instance_type='ml.m5.large',  # Eğitim makinesi tipi (İhtiyaca göre değiştirilebilir)
        framework_version='1.2-1',    # Kullanılacak Scikit-learn versiyonu
        py_version='py3',             # Kullanılacak Python versiyonu
        hyperparameters={             # 'train.py' içine gönderilecek hiperparametreler
            'n_estimators': 100,      # Random Forest ağaç sayısı
            'random_state': 42
        }
    )

    # Eğitimi başlatma ('fit' fonksiyonu ile S3'teki veri yolunu eğitim sunucusuna gönderiyoruz)
    print("Model eğitimi (training) başlatılıyor... Bu işlem AWS bulutunda gerçekleştiği için birkaç dakika sürebilir.")
    sklearn_estimator.fit({'train': train_input})
    print("Model eğitimi başarıyla tamamlandı!")

    # =====================================================================
    # 4. MODEL DAĞITIMI (DEPLOYMENT / ENDPOINT OLUŞTURMA)
    # =====================================================================
    # Eğitilen model, API üzerinden gerçek zamanlı (real-time) tahminler yapabilmek için 
    # bir SageMaker Endpoint'i olarak dağıtılır (deploy edilir).
    print("Model SageMaker Endpoint olarak dağıtılıyor (deployment süreci başlatıldı)...")
    predictor = sklearn_estimator.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large'
    )
    print(f"Model başarıyla buluta dağıtıldı! Endpoint Adı: {predictor.endpoint_name}")

    # =====================================================================
    # 5. ÇIKTI VE TAHMİN (INFERENCE)
    # =====================================================================
    # Dağıtımını yaptığımız Endpoint'e veri gönderip tahmin sonuçlarını alıyoruz.
    print("Endpoint üzerinden örnek verilerle tahmin (inference) yapılarak çıktılar alınıyor...")

    # Test için yereldeki veri setinden hedef değişken hariç ilk 5 satırı alalım
    test_data = pd.read_csv(local_data_path)
    X_test = test_data.drop('target', axis=1).iloc[:5] # Modelin tahmin edeceği girdiler (Features)
    y_true = test_data['target'].iloc[:5].values       # Karşılaştırma için gerçek hedefler (Labels)

    # AWS üzerindeki modele tahmin isteği gönderme (Predict)
    predictions = predictor.predict(X_test.values)

    # Çıktıların Raporlanması (Konsola yazdırma)
    print("\n" + "="*40)
    print("          TAHMİN SONUÇLARI (INFERENCE)")
    print("="*40)
    for i, (pred, true) in enumerate(zip(predictions, y_true)):
        print(f"Örnek {i+1} | Tahmin Edilen Sınıf: {pred} <---> Gerçek Sınıf: {true}")
    print("="*40 + "\n")

    # =====================================================================
    # 6. KAYNAKLARI TEMİZLEME (CLEANUP)
    # =====================================================================

    cleanup_choice = input("Maliyeti önlemek için SageMaker Endpoint'i silinsin mi? (E/H): ")
    if cleanup_choice.lower() == 'e':
        print("Kullanılan kaynaklar (Endpoint) temizleniyor...")
        predictor.delete_endpoint()
        print("Kaynaklar başarıyla silindi. Proje sonlandırıldı.")
    else:
        print(f"Endpoint ({predictor.endpoint_name}) çalışmaya devam ediyor. DİKKAT: Ücretlendirme devam edecektir.")

if __name__ == '__main__':
    main()

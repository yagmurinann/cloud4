"""
PROJE 3 - EĞİTİM BETİĞİ (Training Script)
-------------------------------------------------------------------
Bu betik SageMaker tarafından başlatılan eğitim makinesinde (instance) çalıştırılır.
Görevi: 
1. S3'ten indirilen veriyi okumak.
2. Belirlenen Algoritma (Scikit-Learn Random Forest) ile modeli eğitmek.
3. Eğitilen modeli SageMaker'ın belirlediği dizine kaydederek AWS'de saklanmasını sağlamaktır.
"""

import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # ---------------------------------------------------------------------
    # 1. ARGÜMAN PARSER VE ORTAM DEĞİŞKENLERİ
    # ---------------------------------------------------------------------
    # SageMaker ortamından gelen argümanları (hiperparametreler vb.) parser ile alıyoruz.
    parser = argparse.ArgumentParser()
    
    # Hiperparametreleri alıyoruz (main.py dosyasında 'hyperparameters' sözlüğünde belirlediğimiz değerler)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=42)
    
    # SageMaker'ın AWS ortamındaki standart değişkenleri (verilerin ve modelin nereye kaydedileceği)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN')) # Eğitim verisinin yolu
    
    args = parser.parse_args()
    
    # ---------------------------------------------------------------------
    # 2. VERİ KAYNAĞINI OKUMA (DATA LOADING)
    # ---------------------------------------------------------------------
    # SageMaker, eğitimi başlattığımızda S3'teki veriyi otomatik olarak eğitim sunucusuna (args.train dizinine) indirir.
    print(f"Eğitim verisi okunuyor. Dizin: {args.train}")
    
    # İlgili dizindeki tüm .csv dosyalarını buluyoruz
    train_files = [os.path.join(args.train, file) for file in os.listdir(args.train) if file.endswith('.csv')]
    
    if len(train_files) == 0:
        raise ValueError("HATA: Eğitim dizininde .csv dosyası bulunamadı!")
        
    # Pandas kullanarak .csv dosyasını bir DataFrame'e yüklüyoruz
    df = pd.read_csv(train_files[0]) # Projede tek bir csv dosyası olduğunu varsayıyoruz
    
    # Veri setini Özellikler (X) ve Hedef Sınıf (y) olarak ayırıyoruz
    # Hedef sütunun adının 'target' olduğunu varsayıyoruz. Kendi verinizde bu ismi değiştirmelisiniz.
    X_train = df.drop('target', axis=1)
    y_train = df['target']
    
    # ---------------------------------------------------------------------
    # 3. MODEL GELİŞTİRME VE EĞİTİM (MODEL DEVELOPMENT)
    # ---------------------------------------------------------------------
    # Scikit-Learn kullanarak Random Forest (Rastgele Orman) sınıflandırma modelini tanımlıyoruz.
    print(f"Model başlatılıyor: RandomForestClassifier (n_estimators={args.n_estimators})")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    
    # Modeli verilerle eğitiyoruz (Training)
    print("Model eğitimi yapılıyor...")
    model.fit(X_train, y_train)
    
    # Modelin eğitim verisi üzerindeki başarısını (Accuracy) hesaplıyoruz
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    print(f"Başarı Değerlendirmesi | Model Eğitim Doğruluğu (Accuracy): % {train_accuracy * 100:.2f}")
    
    # ---------------------------------------------------------------------
    # 4. MODELİ KAYDETME (SAVING MODEL)
    # ---------------------------------------------------------------------
    # Eğitilen modeli SageMaker'ın sistem değişkeni olan SM_MODEL_DIR dizinine (args.model_dir) kaydediyoruz.
    # SageMaker eğitim bittiğinde bu dizindeki 'model.joblib' dosyasını alıp S3'e tar.gz formatında yükleyecektir.
    model_path = os.path.join(args.model_dir, 'model.joblib')
    print(f"Eğitilen model kaydediliyor... Dosya Yolu: {model_path}")
    joblib.dump(model, model_path)
    
    print("Eğitim ve model kaydetme işlemi başarıyla tamamlandı!")

# -------------------------------------------------------------------------
# SAGE MAKER İÇİN ÖZEL FONKSİYONLAR (OPSİYONEL AMA GEREKLİ OLABİLİR)
# -------------------------------------------------------------------------
def model_fn(model_dir):
    """
    SageMaker modelin dağıtımı (deployment) yapıldığında endpoint'i ayağa kaldırırken 
    modeli belleğe yüklemek için bu fonksiyonu arar.
    
    Parametreler:
        model_dir (str): Modelin bulunduğu dizin.
        
    Döndürür:
        model: Belleğe yüklenmiş makine öğrenmesi modeli nesnesi.
    """
    print(f"Model yükleniyor (Endpoint için). Dizin: {model_dir}")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

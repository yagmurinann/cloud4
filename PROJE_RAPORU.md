# Proje 3: Akıllı Veri Analitiği ve Makine Öğrenmesi Uygulaması

Bu proje, AWS (Amazon Web Services) platformu üzerinde **SageMaker** kullanılarak uçtan uca (end-to-end) bir makine öğrenmesi modelinin geliştirilmesi, eğitilmesi ve canlı ortama alınması (deployment) süreçlerini kapsamaktadır.

---

## 🏗️ Proje Mimarisi ve Kullanılan Teknolojiler

*   **Veri Kaynağı (Depolama):** Amazon S3 (Simple Storage Service)
*   **Makine Öğrenmesi Servisi:** Amazon SageMaker
*   **Model Algoritması:** Scikit-Learn - Random Forest Classifier
*   **Programlama Dili:** Python 3
*   **Kütüphaneler:** `sagemaker`, `boto3`, `pandas`, `scikit-learn`

---

## 🛠️ Adım Adım Uygulama Aşamaları

### Adım 1: AWS IAM (Kimlik ve Erişim Yönetimi) Yetkilendirmeleri
Projenin AWS üzerinde kaynakları (S3, SageMaker makineleri vb.) yönetebilmesi için lokal makineden AWS'ye güvenli bağlantı sağlandı. 
*   **Yapılan İşlem:** `yagmur` IAM kullanıcısına makine öğrenmesi modellerini eğitebilmesi ve S3'e dosya yükleyebilmesi için `AmazonSageMakerFullAccess` yetkisi ile özel bir `iam:PassRole` politikası tanımlandı.
*   **Kod İçerisindeki Yeri:** `main.py` içerisindeki `role = "arn:aws:iam::..."` satırında yetkilendirme sağlandı.



### Adım 2: Veri Setinin Hazırlanması ve Amazon S3'e Yüklenmesi
Makine öğrenmesi modelinin eğitilebilmesi için öncelikle bir veri setine ihtiyaç vardır.
*   **Yapılan İşlem:** Lokal ortamda (`data/dataset.csv` yoksa) 1000 satırlık sentetik bir veri seti üretildi. Bu veri seti AWS SDK'sı (`boto3`) kullanılarak Amazon S3 bucket'ına başarıyla yüklendi.
*   **Kazanım:** Bulut ortamında veriye güvenli ve hızlı erişim sağlandı.



### Adım 3: Amazon SageMaker Üzerinde Model Eğitimi (Training)
Veri S3'e yüklendikten sonra, AWS bulutunda sanal bir eğitim makinesi ayağa kaldırılarak asıl eğitim işlemi başlatıldı.
*   **Yapılan İşlem:** `main.py` dosyası üzerinden bir `SKLearn Estimator` başlatıldı. Eğitim için `ml.m5.large` tipi bir bulut sunucusu kiralandı. Eğitim esnasında `src/train.py` dosyası çalıştırılarak Random Forest (Rastgele Orman) algoritmasıyla veri seti eğitildi ve başarı oranı hesaplandı.
*   **Kayıt:** Eğitilen model bir `model.joblib` dosyası olarak tekrar AWS sistemine kaydedildi.

-

### Adım 4: Modelin Dağıtımı (Deployment ve Endpoint Oluşturma)
Eğitilen modelin gerçek dünyada kullanılabilmesi ve veri gönderildiğinde anlık tahminler üretebilmesi için canlıya alınması gereklidir.
*   **Yapılan İşlem:** SageMaker'ın `.deploy()` fonksiyonu kullanılarak `ml.m5.large` tipinde bir Endpoint (API arayüzü) oluşturuldu.


### Adım 5: Veriden Bilgi Çıkarma ve Tahmin (Inference)
Canlıya alınan modelin doğruluğunu test etmek amacıyla model API'sine örnek veriler gönderildi.
*   **Yapılan İşlem:** Hedef sınıfı (target) gizlenmiş örnek veriler SageMaker Endpoint'ine gönderildi. Modelin tahmin ettiği sınıflar ile verinin gerçek sınıfları karşılaştırmalı olarak terminal ekranına raporlandı.
*   **Sonuç:** Modelin veriyi başarılı bir şekilde sınıflandırdığı doğrulandı.



### Adım 6: Maliyet Optimizasyonu (Kaynakların Temizlenmesi)
Bulut bilişim mimarisindeki en önemli adımlardan biri olan maliyet yönetimi projenin son adımında uygulandı.
*   **Yapılan İşlem:** Kullanım dışı kaldığında saatlik maliyet oluşturmaması için oluşturulan SageMaker Endpoint'i `predictor.delete_endpoint()` komutuyla sistemden silindi.


---

## 📂 Kod Yapısı

Proje modüler bir mimariyle iki ana parçaya bölünmüştür:

1. **`main.py`:** Pipeline yönetim dosyasıdır. AWS ile bağlantı kurar, S3'e veri atar, `train.py` dosyasını tetikleyerek eğitimi bulutta başlatır ve tahmin sonuçlarını terminale yazdırır.
2. **`src/train.py`:** Sadece AWS SageMaker sunucusu içerisinde çalışan eğitim kodudur. Veriyi parçalar, modeli kurar, eğitir ve model dosyasını paketler. 

*(Kod içerisindeki yorum satırlarında tüm satırların işlevleri detaylı olarak anlatılmıştır.)*

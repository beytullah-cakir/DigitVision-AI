# 🔢 DigitVision AI: MNIST Recognition System

**DigitVision AI**, el yazısı rakamları yüksek doğrulukla tanıyan, modern bir kullanıcı arayüzüne sahip uçtan uca bir derin öğrenme projesidir. Proje, TensorFlow kullanarak eğitilmiş bir Evrişimli Sinir Ağı (CNN) ve Flask tabanlı bir web arayüzünden oluşur.

## 🚀 Öne Çıkan Özellikler

- **Gelişmiş CNN Mimarisi:** Batch Normalization, Dropout ve çok katmanlı Conv2D ile %99+ test doğruluğu.
- **Data Augmentation:** Model, rakamların farklı açılardan ve kaydırılmış hallerinden eğitilerek gerçek dünya el yazısına karşı dayanıklı hale getirilmiştir.
- **Akıllı Ön İşleme:** Çizilen rakamı otomatik olarak bulur, kırpar ve merkezleyerek modelin en iyi sonucu vermesini sağlar.
- **Modern Web Arayüzü:** Karanlık mod (Dark Mode) destekli, tepkisel ve kullanıcı dostu çizim paneli.
- **Performans Analizi:** Eğitim sonrası otomatik oluşan Karışıklık Matrisi (Confusion Matrix) ile hata analizi.

## 🛠️ Teknoloji Yığını

- **Derin Öğrenme:** TensorFlow, Keras
- **Veri İşleme:** NumPy, OpenCV/PIL
- **Görselleştirme:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Frontend:** Vanilla JS, CSS3, HTML5

## 📋 Kurulum

1.  Gerekli kütüphaneleri yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

2.  Modeli eğitin (Opsiyonel - `mnist_model.h5` zaten mevcutsa atlanabilir):

    ```bash
    python train_model.py
    ```

3.  Web uygulamasını başlatın:

    ```bash
    python app.py
    ```

4.  Tarayıcınızda açın: `http://127.0.0.1:5000`

## 📁 Proje Yapısı

- `app.py`: Web sunucusu ve görüntü işleme mantığı.
- `train_model.py`: Yapay sinir ağı eğitimi ve model kaydı.
- `mnist_model.h5`: Eğitilmiş yapay zeka modeli.
- `static/`: Dashboard için JavaScript ve stil dosyaları.
- `templates/`: HTML arayüzü.
- `confusion_matrix.png`: Modelin başarı analiz grafiği.

## 🧠 Model Detayı

Model, MNIST veri seti üzerinde 10 dönem (epoch) boyunca eğitilmiştir. Eğitim sırasında rastgele döndürme ve yakınlaştırma teknikleri kullanılarak, kullanıcıların web arayüzünde yapabileceği farklı çizim tarzlarına uyum sağlanmıştır.

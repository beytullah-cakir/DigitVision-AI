import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

"""
MNIST El Yazısı Rakam Tanıma - Gelişmiş Eğitim Betiği
Data Augmentation (Veri Artırma) ve Batch Normalization (Yığın Normalizasyonu) 
eklenerek modelin gerçek dünya çizimlerine karşı dayanıklılığı artırılmıştır.
"""

def train_mnist_model():
    """MNIST veri setini kullanarak evrişimli sinir ağı (CNN) modelini eğitir."""
    
    print("Veri seti yükleniyor...")
    # TensorFlow/Keras içindeki hazır MNIST verisini çekiyoruz
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizasyon: 0-255 arasındaki piksel değerlerini 0-1 arasına ölçekliyoruz
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape: Kanal bilgisini (gri tonlama için 1) dizilere ekliyoruz
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    print("Gelişmiş model mimarisi oluşturuluyor...")
    # Evrişimli Sinir Ağı (CNN) model yapısı
    model = models.Sequential([
        # İlk Evrişim Bloğu
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(), # Katmanlar arası değerleri stabilize eder
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), # Görüntüyü yarı yarıya küçültür
        layers.Dropout(0.2), # Ezberlemeyi (overfitting) önlemek için rastgele %20 nöron kapatır

        # İkinci Evrişim Bloğu
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Tam Bağlantılı (Fully Connected) Katmanlar
        layers.Flatten(), # 2D görüntüyü 1D vektöre çevirir
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax') # 10 sınıf (0-9) için olasılık dağılımı
    ])

    # Modeli derliyoruz
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # DATA AUGMENTATION: Modeli farklı açılara ve kaymalara karşı eğitir.
    # Bu özellik, kullanıcıların farklı yazım tarzlarını tanımasını sağlar.
    datagen = ImageDataGenerator(
        rotation_range=10,      # 10 dereceye kadar döndürme
        zoom_range=0.1,         # %10 yakınlaştırma/uzaklaştırma
        width_shift_range=0.1,  # %10 yatay kaydırma
        height_shift_range=0.1  # %10 dikey kaydırma
    )
    
    # Gereksiz beklemeyi önlemek için EarlyStopping (Erken Durdurma)
    # Eğer doğrulama kaybı (val_loss) 3 epoch boyunca iyileşmezse eğitimi durdurur.
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Model eğitiliyor (Data Augmentation aktif)...")
    batch_size = 64
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              epochs=10, 
              validation_data=(x_test, y_test),
              callbacks=[early_stop])

    print("Model değerlendiriliyor...")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f'\nTest Doğruluğu: {test_acc}')

    # Modeli diske kaydediyoruz
    model.save('mnist_model.h5')
    print("Model 'mnist_model.h5' olarak güncellendi.")

    # Karışıklık Matrisi (Confusion Matrix) oluşturma
    # Modelin hangi rakamı hangi rakamla karıştırdığını gösterir.
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.title('Gelişmiş Model Karışıklık Matrisi')
    plt.savefig('confusion_matrix.png')
    print("Karışıklık matrisi 'confusion_matrix.png' olarak kaydedildi.")

if __name__ == "__main__":
    train_mnist_model()


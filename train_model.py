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
Data Augmentation ve Batch Normalization eklenerek model dayanıklılığı artırılmıştır.
"""

def train_mnist_model():
    print("Veri seti yükleniyor...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizasyon
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    print("Gelişmiş model mimarisi oluşturuluyor...")
    model = models.Sequential([
        # İlk blok
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # İkinci blok
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Tam bağlantılı katmanlar
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # DATA AUGMENTATION: Modeli farklı açılara ve kaymalara karşı eğitir.
    # Bu özellik 9 gibi sayıların farklı yazım tarzlarını tanımasını sağlar.
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    # Gereksiz beklemeyi önlemek için EarlyStopping
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

    model.save('mnist_model.h5')
    print("Model 'mnist_model.h5' olarak güncellendi.")

    # Karışıklık Matrisi
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.title('Gelişmiş Model Karışıklık Matrisi')
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    train_mnist_model()

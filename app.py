from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

# Flask uygulamasını başlatıyoruz
app = Flask(__name__)

# Eğitilmiş modelin yolu
MODEL_PATH = 'mnist_model.h5'
model = None

def get_model():
    """Modeli yükleyen veya yüklü olanı döndüren yardımcı fonksiyon."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            print("Model bulunamadı! Lütfen önce train_model.py dosyasını çalıştırın.")
    return model

@app.route('/')
def index():
    """Ana sayfayı render eder."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Gelen görüntüyü işleyip tahmin yapan API endpoint'i."""
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Görüntü verisi bulunamadı'}), 400

    # Base64 formatındaki görüntüyü çözüyoruz
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    # Görüntüyü gri tonlamaya (L mode) çeviriyoruz
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # --- GELİŞMİŞ ÖN İŞLEME (MNIST Formatına Uygun) ---
    # 1. Rakamın bulunduğu alanı (bounding box) buluyoruz
    # Görüntü siyah arka plan üzerine beyaz çizim olduğu için:
    img_np = np.array(image)
    rows = np.any(img_np > 30, axis=1) # 30 eşik değeri (hassasiyet)
    cols = np.any(img_np > 30, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return jsonify({'error': 'Çizim bulunamadı'}), 400

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 2. Rakamı kırpıyoruz
    cropped = image.crop((cmin, rmin, cmax, rmax))
    
    # 3. Kenar oranını koruyarak kareleştiriyoruz ve merkeze alıyoruz
    width, height = cropped.size
    max_dim = max(width, height)
    new_image = Image.new('L', (max_dim, max_dim), (0))
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    new_image.paste(cropped, offset)

    # 4. 20x20 boyuta getiriyoruz (MNIST standardı: Rakam 20x20 alana sığmalı)
    new_image = new_image.resize((20, 20), Image.Resampling.LANCZOS)
    
    # 5. 28x28'e tamamlamak için 4 piksel padding (kenar boşluğu) ekliyoruz
    final_image = Image.new('L', (28, 28), (0))
    final_image.paste(new_image, (4, 4))

    # Hazır veriyi diziye çeviriyoruz ve normalize ediyoruz (0-1 arasına çekiyoruz)
    img_array = np.array(final_image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Tahmin aşaması
    m = get_model()
    if m is None:
        return jsonify({'error': 'Model henüz hazır değil'}), 500
        
    prediction = m.predict(img_array)
    digit = np.argmax(prediction[0]) # En yüksek olasılıklı rakamı alıyoruz
    confidence = float(np.max(prediction[0])) # Olasılık değerini alıyoruz

    return jsonify({
        'digit': int(digit),
        'confidence': confidence
    })

if __name__ == '__main__':
    # Flask sunucusunu başlatıyoruz
    app.run(debug=True, port=5000)


// HTML elementlerini seçiyoruz
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictBtn = document.getElementById('predict-btn');
const digitRes = document.getElementById('digit-res');
const confidenceRes = document.getElementById('confidence-res');
const placeholderText = document.getElementById('placeholder-text');
const loader = document.getElementById('loader');

// Çizim ayarlarını yapılandırıyoruz
let isDrawing = false;
ctx.lineWidth = 25; // Çizgi kalınlığı (MNIST fırça yapısına yakın)
ctx.lineCap = 'round'; // Çizgi uçlarının yuvarlak olması
ctx.strokeStyle = 'white'; // Çizim rengi beyaz (arka plan siyah olacak)

/**
 * Çizimi başlatan fonksiyon.
 */
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

/**
 * Çizimi durduran fonksiyon.
 */
function stopDrawing() {
    isDrawing = false;
    ctx.beginPath(); // Yeni çizgi için yolu sıfırla
}

/**
 * Tuval üzerine çizim yapan ana fonksiyon.
 */
function draw(e) {
    if (!isDrawing) return;

    // Fare veya dokunma konumunu hesapla
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Fare olaylarını dinle
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Dokunmatik ekran desteği (Mobil için)
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}, false);

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}, false);

canvas.addEventListener('touchend', (e) => {
    const mouseEvent = new MouseEvent("mouseup", {});
    canvas.dispatchEvent(mouseEvent);
}, false);

// Temizle butonu işleyicisi
clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Tuvali temizle
    digitRes.style.display = 'none';
    confidenceRes.style.display = 'none';
    placeholderText.textContent = 'Çizim yapın ve Tahmin Et\'e basın.';
    placeholderText.style.display = 'block';
});

// Tahmin et butonu işleyicisi
predictBtn.addEventListener('click', async () => {
    // Çizilen görüntüyü Base64 formatında al
    const dataURL = canvas.toDataURL('image/png');

    // UI durumlarını güncelle (Yükleniyor göstergesi vb.)
    placeholderText.style.display = 'none';
    digitRes.style.display = 'none';
    confidenceRes.style.display = 'none';
    loader.style.display = 'block';

    try {
        // Sunucuya POST isteği gönder
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataURL }),
        });

        const data = await response.json();

        loader.style.display = 'none';
        
        if (data.error) {
            placeholderText.textContent = data.error;
            placeholderText.style.display = 'block';
        } else {
            // Tahmin sonucunu ekranda göster
            digitRes.textContent = data.digit;
            confidenceRes.textContent = `Doğruluk: ${(data.confidence * 100).toFixed(2)}%`;
            
            digitRes.style.display = 'block';
            confidenceRes.style.display = 'block';
        }
    } catch (error) {
        // Hata durumunda kullanıcıyı bilgilendir
        loader.style.display = 'none';
        placeholderText.textContent = 'Sunucuyla bağlantı kurulamadı.';
        placeholderText.style.display = 'block';
    }
});


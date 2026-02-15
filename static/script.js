const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictBtn = document.getElementById('predict-btn');
const digitRes = document.getElementById('digit-res');
const confidenceRes = document.getElementById('confidence-res');
const placeholderText = document.getElementById('placeholder-text');
const loader = document.getElementById('loader');

// Set up drawing
let isDrawing = false;
ctx.lineWidth = 25;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch support
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

// Clear canvas
clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    digitRes.style.display = 'none';
    confidenceRes.style.display = 'none';
    placeholderText.style.display = 'block';
});

// Predict digit
predictBtn.addEventListener('click', async () => {
    const dataURL = canvas.toDataURL('image/png');

    placeholderText.style.display = 'none';
    digitRes.style.display = 'none';
    confidenceRes.style.display = 'none';
    loader.style.display = 'block';

    try {
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
            digitRes.textContent = data.digit;
            confidenceRes.textContent = `Doğruluk: ${(data.confidence * 100).toFixed(2)}%`;
            
            digitRes.style.display = 'block';
            confidenceRes.style.display = 'block';
        }
    } catch (error) {
        loader.style.display = 'none';
        placeholderText.textContent = 'Sunucuyla bağlantı kurulamadı.';
        placeholderText.style.display = 'block';
    }
});

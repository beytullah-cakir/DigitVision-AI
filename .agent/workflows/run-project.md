---
description: MNIST Rakam Tanıma Projesini Çalıştırma
---

Bu workflow, CNN tabanlı rakam tanıma uygulamasını başlatmanızı sağlar.

// turbo

1. Gerekli kütüphaneleri yükle:

   ```bash
   pip install -r requirements.txt
   ```

2. Eğer model eğitilmemişse (veya tekrar eğitmek isterseniz):

   ```bash
   python train_model.py
   ```

3. Web uygulamasını başlatın:

   ```bash
   python app.py
   ```

4. Tarayıcınızda şu adresi açın: `http://127.0.0.1:5000`

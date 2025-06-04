# 🤖 Bird vs Drone Binary Classification Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📝 Proje Açıklaması

Bu proje, derin öğrenme kullanarak **kuş** ve **drone** görüntülerini sınıflandıran bir binary classification (ikili sınıflandırma) sistemidir. Projede çeşitli CNN mimarileri test edilmiş ve karşılaştırılmıştır.

## 🎯 Amaç

Birçok otonom ve güvenlik uygulamaları için kritik olan kuş-drone ayrımını otomatik olarak yapabilen bir model geliştirmek.

## 📊 Veri Seti [link](https://www.sciencedirect.com/science/article/pii/S2352340923004742)

- **Toplam Görüntü Sayısı**: 20,952 görüntü
- **Eğitim Seti**: 18,323 görüntü
- **Doğrulama Seti**: 1,740 görüntü  
- **Test Seti**: 889 görüntü
- **Sınıflar**: 
  - 🐦 Bird (Kuş)
  - 🚁 Drone (İHA)

## 🏗️ Model Mimarileri

Projede aşağıdaki CNN mimarileri test edilmiştir:

### 1. Custom CNN
- Özel tasarlanmış CNN mimarisi
- Farklı batch size'lar ile test edildi (32, 64, 128)

### 2. Transfer Learning Modelleri
- **ResNet18 & ResNet50**: Microsoft tarafından geliştirilen residual network
- **VGG16**: Oxford VGG grubu tarafından geliştirilen klasik mimari
- **MobileNet V2**: Mobil cihazlar için optimize edilmiş hafif mimari
- **EfficientNet B0**: Google tarafından geliştirilen verimli mimari

## 🚀 Kurulum

### Gereksinimler

```bash
git clone https://github.com/kemaltml/ClassificationWithDLProject.git
cd ClassificationWithDLProject
```

### Sanal Ortam Oluşturma

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### Ana Bağımlılıklar
- PyTorch 2.7.0
- NumPy 1.26.4
- Pandas 2.2.3
- Matplotlib 3.10.3
- Seaborn 0.13.2
- Scikit-learn 1.6.1
- CUDA 12.6 (GPU desteği için)

## 💻 Kullanım

### 1. Veri Hazırlama

```python
# labeling.py dosyası ile etiketleri hazırlayın
python labeling.py
```

### 2. Model Eğitimi

Ana notebook dosyalarını kullanarak modelleri eğitebilirsiniz:

- model_train/ : Tüm ağırlıklarının yeniden hesaplanarak eğitim yapıldığı modeller.
- model_TL/ : Transfer learning ile eğitimin yapıldığı modeller

### 3. Model Değerlendirmesi

Eğitilmiş modeller `models/` klasörleründe saklanır.

## 📈 Sonuçlar

Proje sonuçları `figures/` klasöründe görselleştirilmiştir:

- **Accuracy Grafikleri**: Model performans karşılaştırmaları
- **Loss Grafikleri**: Eğitim ve doğrulama kayıpları
- **Confusion Matrix**: Sınıflandırma matrisleri
- **Transfer Learning Sonuçları**: TL modellerinin performansı

### Model Performansları

| Model | Accuracy | Loss | Özellikler |
|-------|----------|------|------------|
| Custom CNN | - | - | Özel tasarım, farklı batch size'lar |
| ResNet18 | - | - | Residual connections |
| ResNet50 | - | - | Daha derin residual network |
| VGG16 | - | - | Klasik CNN mimarisi |
| MobileNet V2 | - | - | Hafif ve hızlı |
| EfficientNet B0 | - | - | Verimli scaling |

*Not: Detaylı metrikler notebook dosyalarında bulunabilir.*

## 📁 Proje Yapısı

```
datamine/
├── Dataset/
│   ├── train/          # Eğitim veri seti
│   ├── valid/          # Doğrulama veri seti
│   ├── test/           # Test veri seti
│   └── data.yaml       # Veri konfigürasyonu
├── figures/            # Sonuç grafikleri ve görseller
├── models/             # Eğitilmiş model dosyaları
├── model_train/        # Eğitim modelleri
├── model_test/         # Test modelleri  
├── model_TL/           # Transfer learning modelleri
├── results/            # Sonuç dosyaları
├── labeling.py         # Veri etiketleme scripti
├── labels_train.csv    # Eğitim etiketleri
├── labels_valid.csv    # Doğrulama etiketleri
├── requirements.txt    # Python bağımlılıkları
└── README.md          # Bu dosya
```

## 🛠️ Teknik Detaylar

### GPU Desteği
Proje CUDA 12.6 ile GPU desteği içerir. NVIDIA GPU'lar için optimize edilmiştir.

### Veri Artırma (Data Augmentation)
Modelların genelleme yeteneğini artırmak için çeşitli veri artırma teknikleri kullanılmıştır.

### Transfer Learning
ImageNet üzerinde önceden eğitilmiş modeller kullanılarak transfer learning uygulanmıştır.

## 🤝 Katkıda Bulunma

1. Bu repo'yu fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👤 Yazar

**Kemal**
- GitHub: [@kemaltml](https://github.com/kemaltml)

## 🙏 Teşekkürler

- Veri seti sağlayıcıları
- PyTorch geliştirici topluluğu
- Açık kaynak kütüphane geliştiricileri



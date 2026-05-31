# TestEnvironment – Nihai TSM Kamera Başına Autoencoder Çıkarım (Inference) Kılavuzu

Bu klasör, eğitimi tamamlanmış nihai autoencoder (otomatik kodlayıcı) modelleriyle çıkarım (inference) yapmak üzere tasarlanmış hafif bir test ortamı içerir. Modelleri yeniden eğitmeye gerek kalmadan, halihazırda çıkarılmış olan TSM + 3D el pozu öznitelik (feature) dosyalarını test etmek için kullanılır.

## 1. Amaç

Ana proje kapsamında, Assembly101 montaj videoları için çeşitli anomali tespit stratejileri test edilmiştir. Ortak (pooled) modeller, kamera başına modeller, SlowFast öznitelikleri, TSM öznitelikleri, farklı autoencoder mimarileri, eşik (threshold) yüzdelikleri ve füzyon (birleştirme) yöntemleri karşılaştırıldıktan sonra seçilen nihai sistem şu şekildedir:

| Bileşen                        | Nihai Seçim                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| Öznitelik Türü (Feature type)  | TSM görsel öznitelikleri + 3D el pozu öznitelikleri                                  |
| Girdi Boyutu (Input dimension) | 4064                                                                                 |
| Model Stratejisi               | Kamera başına bir autoencoder                                                        |
| Autoencoder Mimarisi           | 4064 → 256 → 64 → 16 → 64 → 256 → 4064                                               |
| Dropout                        | Nihai seçilen modelde kullanılmadı                                                   |
| Eşik Stratejisi                | Doğrulama (validation) seti normal rekonstrüksiyon hatası, 75. yüzdelik (percentile) |
| Füzyon Yöntemi                 | Mevcut kamera skorları üzerinde Maksimum Füzyon (Max fusion)                         |

Bu test ortamı, eğitilmiş nihai modelleri kullanır ve bunları test öznitelik dosyalarına uygular. Ham videolardan öznitelik çıkarımı **yapmaz** ve yeni autoencoder'lar **eğitmez**.

## 2. Klasör Yapısı

Beklenen klasör yapısı şu şekildedir:

```text
TestEnvironment/
│
├── TestSingleVideoWithFinalAE.py
├── README.md
├── requirements.txt
│
├── FinalTSMPerCameraAE_Results/
│   ├── final_tuning_summary.csv
│   └── small_no_dropout/
│       ├── C10095/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       ├── C10118/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       ├── C10119/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       ├── C10390/
│       │   ├── autoencoder_best.pth
│       │   └── scaler.pkl
│       └── C10404/
│           ├── autoencoder_best.pth
│           └── scaler.pkl
│
├── per_camera_tsm/
│   ├── test_features_tsm_C10095.npy
│   ├── test_features_tsm_C10118.npy
│   ├── test_features_tsm_C10119.npy
│   ├── test_features_tsm_C10390.npy
│   ├── test_features_tsm_C10404.npy
│   ├── test_features_tsm_window_labels_C10095.npy
│   ├── test_features_tsm_window_labels_C10118.npy
│   ├── test_features_tsm_window_labels_C10119.npy
│   ├── test_features_tsm_window_labels_C10390.npy
│   └── test_features_tsm_window_labels_C10404.npy
│
└── single_video_inference_outputs/
    └── betik çalıştırıldıktan sonra otomatik olarak oluşturulur

```

## 3. Gerekli Girdi Dosyaları

### 3.1 Nihai Model Dosyaları

Betik, her kamera için şu dosyaları bekler:

```text
FinalTSMPerCameraAE_Results/small_no_dropout/<KAMERA_ID>/autoencoder_best.pth
FinalTSMPerCameraAE_Results/small_no_dropout/<KAMERA_ID>/scaler.pkl

```

Her kameranın kendine ait eğitilmiş bir autoencoder'ı ve ölçeklendiricisi (scaler) olduğu için bu dosyalar zorunludur. `C10095` kamerasından çıkarılan bir öznitelik, mutlaka `C10095` ölçeklendiricisi ve autoencoder'ı ile test edilmelidir. Kamera modelleri birbirine karıştırılmamalıdır.

### 3.2 Eşik (Threshold) Dosyası

Betik ayrıca şu dosyaya ihtiyaç duyar:

```text
FinalTSMPerCameraAE_Results/final_tuning_summary.csv

```

Bu dosya, her kamera için nihai eşik değerlerini okumak amacıyla kullanılır. Seçilen nihai eşik ayarı `p=75`'tir.

### 3.3 Test Öznitelik Dosyaları

Test öznitelikleri, aşağıdaki boyuta (shape) sahip TSM + 3D el pozu öznitelikleri olmalıdır:

```text
(pencere_sayisi, 4064)

```

Beklenen dosya adları:

```text
per_camera_tsm/test_features_tsm_C10095.npy
per_camera_tsm/test_features_tsm_C10118.npy
per_camera_tsm/test_features_tsm_C10119.npy
per_camera_tsm/test_features_tsm_C10390.npy
per_camera_tsm/test_features_tsm_C10404.npy

```

Betik tek bir kamerayla, birkaç kamerayla veya 5 kameranın tümüyle çalışabilir:

| Mevcut Öznitelik Dosyaları | Çıkarım (Inference) Modu       |
| -------------------------- | ------------------------------ |
| 1 Kamera                   | Tek kameralı çıkarım           |
| 2–4 Kamera                 | Kısmi kameralı maksimum füzyon |
| 5 Kamera                   | Tam nihai maksimum füzyon      |

Eksiksiz nihai sistem performansını görmek için 5 kameranın da mevcut olması gerekir.

### 3.4 Etiket (Label) Dosyaları

Eğer etiket dosyaları mevcutsa; betik Doğruluk (Accuracy), Keskinlik (Precision), Duyarlılık (Recall), F1-skoru, AUROC ve hata matrisini (confusion matrix) hesaplar.

Beklenen dosya adları:

```text
per_camera_tsm/test_features_tsm_window_labels_C10095.npy
per_camera_tsm/test_features_tsm_window_labels_C10118.npy
per_camera_tsm/test_features_tsm_window_labels_C10119.npy
per_camera_tsm/test_features_tsm_window_labels_C10390.npy
per_camera_tsm/test_features_tsm_window_labels_C10404.npy

```

Etiketlerin anlamı:

```text
0 = Normal / Doğru montaj penceresi
1 = Anomali / Hata veya düzeltme penceresi

```

Etiketler eksik olsa bile betik anomali tahminlerini ve zaman çizelgesi çıktılarını üretmeye devam eder, ancak değerlendirme metriklerini hesaplayamaz.

## 4. Betiğin Çalışma Prensibi

Betik, aşağıdaki çıkarım iş akışını (pipeline) takip eder:

```text
TSM + 3D el pozu öznitelik dosyaları
        ↓
Mevcut kamera dosyalarının tespiti
        ↓
Her kamera için eşleşen scaler.pkl dosyasının yüklenmesi
        ↓
Her kamera için eşleşen autoencoder_best.pth dosyasının yüklenmesi
        ↓
Her pencere için rekonstrüksiyon hatasının hesaplanması
        ↓
Her kamera skorunun kendi eşik değerine bölünerek normalize edilmesi
        ↓
Mevcut kameralar arasında maksimum füzyon (max fusion) uygulanması
        ↓
Pencerelerin Normal veya Anomali olarak sınıflandırılması
        ↓
CSV, JSON ve PNG çıktı dosyalarının kaydedilmesi

```

## 5. Betiği Çalıştırma

`TestEnvironment` klasöründe PowerShell veya terminalinizi açın.

Gerekirse sanal ortamınızı (virtual environment) aktifleştirin:

```powershell
.\venv\Scripts\activate

```

Ardından betiği çalıştırın:

```powershell
python TestSingleVideoWithFinalAE.py

```

Beklenen terminal çıktısı şuna benzer olacaktır:

```text
DEVICE: cuda
Camera: C10095
Feature shape: (839, 4064)
Detected anomaly windows: ...
Available cameras: ['C10095', 'C10118', 'C10119', 'C10390', 'C10404']
Inference mode: full final max fusion

```

Eğer sisteminizde CUDA mevcutsa betik GPU üzerinde çalışacaktır. Aksi takdirde otomatik olarak CPU'ya döner.

## 6. Doğrulanmış Mevcut Test Örneği

Gerçekleştirilen doğrulanmış test çalışmasında, öznitelik dosyaları şu boyutlardaydı:

```text
test_features_tsm_C10095.npy (839, 4064)
test_features_tsm_C10118.npy (839, 4064)
test_features_tsm_C10119.npy (839, 4064)
test_features_tsm_C10390.npy (839, 4064)
test_features_tsm_C10404.npy (839, 4064)

```

Etiket dosyaları da bu boyutlarla eşleşmekteydi:

```text
test_features_tsm_window_labels_C10095.npy (839,)
test_features_tsm_window_labels_C10118.npy (839,)
test_features_tsm_window_labels_C10119.npy (839,)
test_features_tsm_window_labels_C10390.npy (839,)
test_features_tsm_window_labels_C10404.npy (839,)

```

Doğrulanmış tam füzyon (full-fusion) performansı şu şekildedir:

| Metrik                | Değer  |
| --------------------- | ------ |
| Doğruluk (Accuracy)   | %91.06 |
| Keskinlik (Precision) | %76.95 |
| Duyarlılık (Recall)   | %92.49 |
| F1-Skoru              | %84.01 |
| AUROC                 | %95.73 |

Hata Matrisi (Confusion Matrix) Değerleri:

|                    | Tahmin Edilen Normal | Tahmin Edilen Anomali |
| ------------------ | -------------------- | --------------------- |
| **Gerçek Normal**  | 567                  | 59                    |
| **Gerçek Anomali** | 16                   | 197                   |

**Sonuçların Yorumu:**

- Model, 197 anomali penceresini doğru tespit etmiştir (True Positive).
- 16 anomali penceresini gözden kaçırmıştır (False Negative).
- 567 normal pencereyi doğru şekilde sınıflandırmıştır (True Negative).
- 59 defa hatalı alarm vermiştir (False Positive).
- Yüksek duyarlılık (Recall) değeri, sistemin anomalileri yakalama konusunda oldukça güçlü olduğunu göstermektedir.

## 7. Oluşturulan Çıktılar

Betik çalıştıktan sonra çıktılar şu klasöre kaydedilir:

```text
single_video_inference_outputs/

```

### 7.1 CSV ve JSON Çıktıları

| Çıktı Dosyası               | Açıklama                                                                |
| --------------------------- | ----------------------------------------------------------------------- |
| `C10095_camera_results.csv` | C10095 için pencere başına rekonstrüksiyon sonuçları                    |
| `C10118_camera_results.csv` | C10118 için pencere başına rekonstrüksiyon sonuçları                    |
| `C10119_camera_results.csv` | C10119 için pencere başına rekonstrüksiyon sonuçları                    |
| `C10390_camera_results.csv` | C10390 için pencere başına rekonstrüksiyon sonuçları                    |
| `C10404_camera_results.csv` | C10404 için pencere başına rekonstrüksiyon sonuçları                    |
| `fusion_timeline.csv`       | Nihai birleştirilmiş pencere başına tahmin zaman çizelgesi              |
| `fusion_test_results.csv`   | Analiz/demo kullanımı için aynı nihai birleştirilmiş tahmin tablosu     |
| `detected_intervals.csv`    | Peş peşe gelen anomali pencerelerinin aralıklar halinde gruplanmış hali |
| `performance_metrics.csv`   | Accuracy, Precision, Recall, F1, AUROC ve TP/FP/FN/TN değerleri         |
| `performance_metrics.json`  | Aynı metriklerin JSON formatındaki hali                                 |
| `inference_mode.json`       | Çıkarımın tekli, kısmi veya tam füzyon modunda yapıldığının kaydı       |

### 7.2 PNG Görsel Çıktıları

| Çıktı Grafiği                       | İçerik                                                     | Önerilen Kullanım Alanı       |
| ----------------------------------- | ---------------------------------------------------------- | ----------------------------- |
| `fusion_score_distribution.png`     | Eşik değeriyle birlikte normal ve anomali skor dağılımları | Raporlar ve Sunumlar          |
| `confusion_matrix.png`              | TP, FP, FN, TN sayımları (Hata Matrisi)                    | Raporlar ve Sunumlar          |
| `roc_curve.png`                     | ROC eğrisi ve AUROC değeri                                 | Raporlar ve Sunumlar          |
| `performance_metrics_table.png`     | Ana metrik değerlerinin tablo hali                         | Sunumlar ve Posterler         |
| `fusion_timeline_plot.png`          | Test pencereleri boyunca nihai füzyon skoru değişimi       | Raporlar ve Demo anlatımları  |
| `fusion_timeline_detailed.png`      | Test pencereleri üzerindeki TP, FP ve FN noktaları         | Detaylı Analizler / Ekler     |
| `per_camera_detected_anomalies.png` | Kamera başına tespit edilen anomali pencere sayıları       | Çoklu kamera davranış analizi |

> **Not:** Aralık (interval) sütun grafikleri, görsel olarak çok kalabalık olduğu ve zaman çizelgesi grafiklerine kıyasla daha az kullanışlı bilgi sunduğu için nihai sürümden bilinçli olarak kaldırılmıştır.

## 8. Önemli Notlar

### 8.1 Bu betik model eğitmez

Bu betik sadece çıkarım (inference) ve test içindir. Eğitim verisi kullanmaz ve model ağırlıklarını güncellemez. Yeniden eğitmek veya deneyi tekrarlamak için `FinalTSMPerCameraTuning.py` dosyasını kullanmalısınız.

### 8.2 Bu betik videodan öznitelik çıkarmaz

Ham `.mp4` video dosyaları doğrudan bu betiğe verilemez. Öznitelik çıkarım adımının önceden tamamlanmış olması gerekir. Doğru iş akışı şu şekildedir:

```text
Ham Video → Öznitelik Çıkarım Betiği → TSM + El Pozu (.npy) Dosyaları → TestSingleVideoWithFinalAE.py → Çıktılar

```

### 8.3 Öznitelik boyutu mutlaka 4064 olmalıdır

Eğer öznitelik dosyasının ikinci boyutu 4064 değilse, betik hata vererek duracaktır. Bu durum kararlılık için önemlidir, çünkü nihai autoencoder modeli 4064 boyutlu vektörlerle eğitilmiştir.

### 8.4 Kamera ID'si dosya adında yer almalıdır

Öznitelik dosyasının adında kameranın ID'si mutlaka geçmelidir (Örn: `test_features_tsm_C10095.npy`). Betik, hangi scaler ve autoencoder'ı yükleyeceğini bu ID'ye bakarak seçer.

### 8.5 Tam füzyon sonucu için 5 kamera da gereklidir

Eğer 5 kamera dosyası da mevcutsa, betik nihai "max-fusion" sistemini çalıştırır. Daha az kamera varsa betik yine de çalışır ancak bu durum "tek kameralı" veya "kısmi füzyonlu" çıkarım olarak yorumlanmalıdır.

## 9. Hızlı Sorun Giderme (Troubleshooting)

- **Hata:** `Feature klasörü bulunamadı`
- **Çözüm:** `per_camera_tsm/` klasörünün doğru yerde yaratıldığından ve adının doğruluğundan emin olun.

- **Hata:** `model dosyası yok` veya `scaler dosyası yok`
- **Çözüm:** İlgili kamera klasörünün içinde `autoencoder_best.pth` ve `scaler.pkl` dosyalarının yer aldığını kontrol edin.

- **Hata:** Öznitelik boyutu (dimension) hatası
- **Çözüm:** Beklenen boyut `(pencere_sayısı, 4064)` şeklindedir. İkinci boyut 4064 değilse, öznitelik çıkarma adımınız nihai modelle uyumsuz demektir.

- **Hata:** "Lengths are different" (Uzunluklar farklı olduğu için füzyon yapılamıyor)
- **Çözüm:** Füzyona giren tüm kamera öznitelik dizilerinin aynı pencere sayısına sahip olması gerekir (Örn: Hepsi 839 pencere olmalıdır). Kameralardan birinin pencere sayısı farklıysa, öznitelik çıkarma/pencerleme adımında bir hizalama problemi var demektir.

- **Sorun:** Metrikler üretilmiyor
- **Çözüm:** Metriklerin hesaplanabilmesi için etiket (label) dosyalarının bulunması şarttır. Etiketler yoksa betik yalnızca tahminleri ve zaman çizelgelerini oluşturur.

## 10. Sunumlar İçin Kısa Özet

Bu test ortamı, eğitilmiş TSM + 3D el pozu tabanlı kamera başına autoencoder modellerini kullanır. Her bir kameraya ait öznitelik dosyası kendi ölçeklendiricisiyle (scaler) normalize edildikten sonra ilgili autoencoder modelinden geçirilir. Elde edilen rekonstrüksiyon hatası, o kameranın kendi eşik değerine bölünerek standart bir "anomali skoru" elde edilir. Elde edilen tüm kamera skorları "maksimum füzyon (max fusion)" yöntemiyle birleştirilir. Eğer nihai füzyon skoru 1.0 değerini aşarsa, ilgili zaman dilimi "anomali" olarak sınıflandırılır. Betik; ROC eğrisi, hata matrisi, skor dağılımı ve zaman çizelgesi gibi görsel kanıtları ve sayısal raporları otomatik olarak çıktı olarak sunar.

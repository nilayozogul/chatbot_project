# 🤖 Üretken Yapay Zeka ile Chatbot Geliştirme Temelleri
## 📘 Proje Sahibi
**Ayça Nilay Özoğul - 170421012**

## 🧠 Proje Tanıtımı
Bu proje kapsamında, müşteri destek süreçlerini daha verimli ve akıllı hale getirecek bir **RAG (Retrieval Augmented Generation) + Intent Classification** tabanlı chatbot geliştirilmiştir.
Kullanıcıdan gelen doğal dil soruları analiz edilerek, önce doğru intent sınıfı belirlenmekte, ardından ilgili bilgiye dayalı bir yanıt üretilmektedir.
Proje, **Meta LLaMA 3 8B Instruct** ve **Gemini 1.5 Flash** gibi güçlü LLM altyapılarını kullanmaktadır.
Ek olarak, veri seti üzerinde eğitilen **Logistic Regression** tabanlı intent sınıflandırma modeli ile chatbot'un anlama yeteneği artırılmıştır.

## 📊 Veri Seti Özellikleri
- Kaynak: Kaggle
- Satır Sayısı: 1035
- Sütunlar: `user_input`, `intent`, `answer`
- Intent Sınıfları: `cancel_subscription`, `greeting`, `goodbye`, `order_status`, `payment_update`, `product_availability`, `rejection`, `shipping_info`, `thanks`
- Her intent sınıfı için dengeli örnek (115 adet)

### 🔧 Veri Temizleme
- Eksik veriler temizlendi
- Sınıflar dengelendi
- Metinler küçük harfe çevrildi ve noktalama temizliği yapıldı

![Veri Görseli 1](images_from_docx/image_1.png)

## 🎯 Intent ve Sınıflandırma Yöntemi
Intent, kullanıcının diyalogdaki amacını temsil eder. Örneğin: `Siparişimin durumunu öğrenmek istiyorum` → `order_status`

**Kullanılan Yöntemler:**
- Embedding: `SentenceTransformer (all-MiniLM-L6-v2)`
- Sınıflayıcı: `Logistic Regression` (sklearn)

![Intent Görseli](images_from_docx/image_2.png)

## 🤖 Kullanılan LLM Modelleri
**🔹 Meta LLaMA 3 8B Instruct**
- Açık kaynaklı, OpenRouter API ile erişim
- Yüksek doğruluk ve dil anlama yeteneği

**🔹 Gemini 1.5 Flash**
- Google AI tabanlı hızlı model
- Gerçek zamanlı yanıtlar için uygun

Her iki model de RAG sürecine entegre edilmiştir.

![LLM Görseli](images_from_docx/image_3.png)

## 🛠️ Model Eğitim Süreci
- Embedding: `all-MiniLM-L6-v2`
- Sınıflayıcı: `Logistic Regression (max_iter=1000)`
- Eğitim/Test: %80 - %20 ayrımı
- 5 Fold Cross Validation uygulandı

**Başarı Metrikleri:**
- Precision: `1.00`
- Recall: `1.00`
- F1-Score: `1.00`
- Confusion Matrix: Her sınıf doğru sınıflandırıldı

![Model Performansı](images_from_docx/image_4.png)

## 💻 Streamlit Arayüzü
- Kullanıcı girişi, tahmin edilen intent ve confidence değeri gösterilir
- %70 altı confidence için fallback intent gösterimi yapılır
- Toggle ile Classifier tabanlı ve RAG tabanlı cevap arasında seçim yapılabilir
- LLaMA ve Gemini model seçimi yapılabilir

![Arayüz Görseli](images_from_docx/image_5.png)

## 📈 Model Karşılaştırması
| Metric            | LLaMA | Gemini |
|-------------------|-------|--------|
| Accuracy          | 10/10 | 10/10  |
| Fluency           | 10/10 | 9/10   |
| Conciseness       | 8/10  | 10/10  |
| Match to Ground   | 10/10 | 10/10  |

![Karşılaştırma Tablosu](images_from_docx/image_6.png)

## ✅ Sonuç ve Değerlendirme
- Basit bir intent sınıflayıcı + RAG destekli LLM kombinasyonu ile yüksek doğruluk elde edilmiştir.
- Farklı LLM’ler (LLaMA ve Gemini) farklı kullanıcı deneyimlerine hitap etmektedir.
- Sistem kolayca genişletilebilir ve yeniden kullanılabilir bir yapıya sahiptir.

![Genel Değerlendirme](images_from_docx/image_7.png)
# RAG + Intent Classifier Chatbot

Bu proje, RAG (Retrieval Augmented Generation) ve Intent Classification teknolojilerini kullanarak gelişmiş bir chatbot uygulamasıdır. Hem Google Gemini hem de Meta Llama modellerini destekler.

## 🏗️ Proje Yapısı

```
GitHub/
├── data/
│   └── data_set.xlsx           # Eğitim veri seti
├── models/
│   ├── llama_model.py          # Llama model sınıfı
│   ├── gemini_model.py         # Gemini model sınıfı
│   ├── intent_classifier.pkl   # Eğitilmiş intent classifier
│   ├── vector_store.pkl        # Vector store
│   └── faiss.index            # FAISS index dosyası
├── app/
│   └── streamlit_app.py        # Streamlit web uygulaması
├── rag_utils.py               # RAG yardımcı fonksiyonları
├── README.md                  # Bu dosya
└── requirements.txt           # Python bağımlılıkları
```

## 🚀 Kurulum

1. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key'leri ayarlayın:**
   - `Gemini/API_KEY.txt` dosyasına Google Gemini API key'inizi ekleyin
   - `Llama/API_KEY.txt` dosyasına OpenRouter API key'inizi ekleyin

3. **Model dosyalarını kopyalayın:**
   ```bash
   # Intent classifier ve vector store dosyalarını models klasörüne kopyalayın
   cp Gemini/intent_classifier.pkl GitHub/models/
   cp Gemini/vector_store.pkl GitHub/models/
   cp Gemini/faiss.index GitHub/models/
   ```

## 🎯 Kullanım

1. **Uygulamayı başlatın:**
   ```bash
   cd GitHub/app
   streamlit run streamlit_app.py
   ```

2. **Web arayüzünde:**
   - Sol panelden model seçin (Gemini veya Llama)
   - Sorunuzu yazın
   - İsteğe bağlı olarak sadece classifier kullanın
   - Gönder butonuna tıklayın

## 🔧 Özellikler

### Intent Classification
- Kullanıcı sorularının amacını tahmin eder
- Güven skoruna göre fallback mekanizması
- Önceden eğitilmiş sınıflandırıcı kullanır

### RAG (Retrieval Augmented Generation)
- Benzer soruları vector store'dan bulur
- AI modeli ile gelişmiş cevaplar üretir
- Knowledge base'den bilgi alır

### Model Desteği
- **Google Gemini**: Gemini 1.5 Flash modeli
- **Meta Llama**: Llama 3 8B Instruct modeli (OpenRouter üzerinden)

## 📊 Veri Seti

`data_set.xlsx` dosyası şu kolonları içermelidir:
- `user_input`: Kullanıcı soruları
- `intent`: Soru kategorileri
- `answer`: Beklenen cevaplar

## 🛠️ Geliştirme

### Yeni Model Ekleme
1. `models/` klasöründe yeni model sınıfı oluşturun
2. `streamlit_app.py`'de model seçeneklerine ekleyin
3. API key dosyasını ekleyin

### Intent Classifier Eğitimi
```bash
cd Gemini  # veya Llama
python train_intent_classifier.py
```

## 📝 Notlar

- API key'lerinizi güvenli tutun
- Vector store büyük dosyalar içerebilir
- FAISS index'i CPU optimizasyonu için kullanılır
- Fallback threshold'u 70% olarak ayarlanmıştır

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 
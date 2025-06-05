# tax-gst-ai-assistant
---
title: Tax & GST AI Assistant
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🏛️ Tax & GST AI Assistant

An intelligent AI assistant for Indian Income Tax and GST queries, powered by Google's FLAN-T5 and LangChain RAG (Retrieval Augmented Generation).

## 🚀 Features

- **Real-time Tax Assistance**: Get instant answers to Income Tax and GST questions
- **RAG-powered Accuracy**: Uses retrieval augmented generation for factual responses
- **Interactive Chat Interface**: User-friendly Gradio interface with sample questions
- **Source Citations**: Shows relevant tax law sources for each answer
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

## 🛠️ Technology Stack

- **Model**: Google FLAN-T5-base (Text-to-Text Transfer Transformer)
- **Framework**: LangChain for RAG implementation
- **Vector Store**: FAISS for efficient document retrieval
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Frontend**: Gradio for interactive web interface
- **Deployment**: Hugging Face Spaces

## 🎯 Use Cases

- **Income Tax Queries**: Deductions, TDS rates, tax calculations
- **GST Information**: Registration requirements, tax rates, compliance
- **Tax Planning**: Investment options, presumptive taxation
- **Quick Reference**: Section numbers, thresholds, deadlines

## 💡 Sample Questions

- "What is the rate of TDS on salary?"
- "What are the different GST rates?"
- "How much can I invest under Section 80C?"
- "When is GST registration mandatory?"
- "What is presumptive taxation under 44AD?"

## ⚠️ Disclaimer

This AI assistant is designed for educational and informational purposes only. Tax laws are complex and subject to change. Always consult with qualified tax professionals or official government resources for definitive tax advice and compliance requirements.

## 🔧 Technical Details

### Model Architecture
- **Base Model**: google/flan-t5-base (250M parameters)
- **Task**: Text-to-text generation optimized for Q&A
- **Context Length**: Up to 1024 tokens
- **Temperature**: 0.3 for balanced creativity and accuracy

### RAG Implementation
- **Document Chunking**: 500 characters with 50-character overlap
- **Retrieval**: Top-3 most relevant document chunks
- **Embedding Model**: All-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: FAISS for fast similarity search

### Performance Optimizations
- **Mixed Precision**: FP16 when GPU available
- **Model Caching**: Gradio cache for faster subsequent loads
- **Efficient Tokenization**: Optimized for deployment

## 📊 Model Performance

- **Response Time**: ~3-5 seconds per query
- **Context Accuracy**: Leverages retrieved tax documents
- **Coverage**: Indian Income Tax Act 1961 & GST Act 2017
- **Language**: English (Indian tax terminology)

## 🤝 Contributing

This project demonstrates the application of modern NLP techniques to domain-specific knowledge retrieval. Feel free to:

- Report issues or suggest improvements
- Contribute additional tax knowledge sources
- Enhance the user interface
- Optimize model performance

## 📝 License

MIT License - Feel free to use this code for educational and commercial purposes.

## 👨‍💻 Developer

Created with ❤️ for the AI and tax community. 


---

*Built using Hugging Face Transformers, LangChain, and Gradio*

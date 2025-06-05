import gradio as gr
import os
import functools
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    except Exception as e:
        print(f"Failed to install {package}: {e}")
        return False

# Install required packages if missing
required_packages = {
    "PyPDF2": "PyPDF2",
    "transformers": "transformers", 
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "numpy": "numpy"
}

for module_name, package_name in required_packages.items():
    try:
        __import__(module_name.replace("-", "_"))
    except ImportError:
        print(f"Installing {package_name}...")
        install_package(package_name)

# Now import everything
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
    print("✅ PyPDF2 loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import PyPDF2: {e}")
    print("Installing PyPDF2...")
    install_package("PyPDF2")
    try:
        from PyPDF2 import PdfReader
        PDF_AVAILABLE = True
        print("✅ PyPDF2 installed and loaded")
    except ImportError:
        print("❌ PyPDF2 still not available")
        PDF_AVAILABLE = False
        PdfReader = None

# Try to import torch, make it optional
try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch loaded successfully")
except ImportError:
    print("⚠️ PyTorch not available, using CPU-only mode")
    TORCH_AVAILABLE = False

# Import LangChain with fallbacks
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("✅ LangChain Community loaded successfully")
except ImportError:
    print("Installing langchain-community...")
    install_package("langchain-community")
    try:
        from langchain_community.llms import HuggingFacePipeline
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ LangChain Community installed and loaded")
    except ImportError:
        print("❌ Using fallback langchain imports")
        from langchain.llms import HuggingFacePipeline
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✅ All AI packages loaded successfully")
except ImportError as e:
    print(f"Installing missing AI packages: {e}")
    for pkg in ["transformers", "langchain", "sentence-transformers"]:
        install_package(pkg)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(path):
    """Extract text from PDF file with robust error handling"""
    if not PDF_AVAILABLE:
        print("❌ PyPDF2 not available, cannot read PDFs")
        return ""
    
    try:
        print(f"📄 Attempting to read PDF: {path}")
        reader = PdfReader(path)
        text_content = []
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                text_content.append(page_text)
                print(f"✅ Successfully read page {i+1}")
            except Exception as e:
                print(f"⚠️ Error reading page {i+1}: {e}")
                continue
        
        full_text = "\n".join(text_content)
        print(f"✅ Successfully extracted {len(full_text)} characters from {path}")
        return full_text
        
    except Exception as e:
        print(f"❌ Error reading PDF {path}: {str(e)}")
        return ""

def load_tax_documents():
    """Load and process tax documents from PDFs"""
    documents = []
    
    # Define your PDF file paths for different environments
    local_pdf_path = r"C:\Users\Laptop22\OneDrive - 4X4 Advisory Services Private Limited\Desktop\Pandas\Tax assistant\Data(Tax)"
    
    pdf_files = [
        "income-tax-bill-2025.pdf",
        "CGST-Act-Updated-30092020.pdf"
    ]
    
    print(f"🔍 Looking for PDF files...")
    print(f"PDF processing available: {PDF_AVAILABLE}")
    
    for pdf_file in pdf_files:
        # Try deployment path first (same directory as app.py)
        deployment_path = pdf_file
        # Try local development path
        local_path = os.path.join(local_pdf_path, pdf_file)
        
        pdf_path = None
        if os.path.exists(deployment_path):
            pdf_path = deployment_path
            print(f"📂 Found PDF in deployment directory: {deployment_path}")
        elif os.path.exists(local_path):
            pdf_path = local_path
            print(f"📂 Found PDF in local directory: {local_path}")
        
        if pdf_path and PDF_AVAILABLE:
            print(f"📖 Loading {pdf_path}...")
            text = extract_text_from_pdf(pdf_path)
            if text and len(text) > 100:  # Ensure we got meaningful content
                documents.append(text)
                print(f"✅ Successfully loaded {len(text)} characters from {pdf_file}")
            else:
                print(f"⚠️ Failed to extract meaningful text from {pdf_path}")
        else:
            if not PDF_AVAILABLE:
                print(f"❌ Cannot read {pdf_file}: PyPDF2 not available")
            else:
                print(f"❌ PDF file not found: {pdf_file}")
    
    # Enhanced fallback data with actual tax content
    if not documents:
        print("📚 No PDF files found or loaded, using comprehensive sample tax data...")
        documents = [
            """Income Tax Act 1961 - Key Sections:
            
Section 80C: Deduction up to Rs 1.5 lakh for investments in ELSS, PPF, NSC, Life Insurance Premium, Tuition fees for children, Home loan principal repayment, ULIP, NPS contributions.

Section 80D: Deduction for health insurance premiums - Up to Rs 25,000 for self and family, additional Rs 25,000 for parents (Rs 50,000 if parents are senior citizens).

Section 44AD: Presumptive taxation scheme for small businesses - 8% of turnover for digital payments, 6% for cash payments. Applicable for businesses with turnover up to Rs 2 crores.

TDS Rates: Salary (as per slab), Property rent 10% (if rent > Rs 50,000/month), Professional services 10% (payment > Rs 30,000), Contractor payments 1-2%, Commission 5%.

Capital Gains: Short term (≤1 year) - 15% for equity, as per slab for others. Long term (>1 year) - 10% above Rs 1 lakh for equity, 20% with indexation for real estate.""",

            """GST Act - Central Goods and Services Tax:
            
GST Rates Structure:
- 5% on essential items (food grains, medicines, books, milk)
- 12% on standard items (computers, processed food, mobile phones)  
- 18% on most goods and services (restaurants, telecom, textiles, soaps)
- 28% on luxury items (cars, tobacco, aerated drinks, AC, dishwasher)

Registration Requirements:
- Mandatory if turnover exceeds Rs 20 lakh (Rs 10 lakh for special category states)
- Interstate supply requires registration regardless of turnover
- E-commerce operators need compulsory registration
- Voluntary registration allowed below threshold

Input Tax Credit (ITC): Credit of GST paid on inputs can be used to pay GST on outputs. Proper invoices and compliance required.

Composition Scheme: Available for small traders with turnover up to Rs 1.5 crores. Tax rates: 1% for traders, 2% for manufacturers, 6% for restaurants.

Returns: GSTR-1 (monthly/quarterly), GSTR-3B (monthly), GSTR-9 (annual). Due dates vary based on turnover and registration type."""
        ]
    
    print(f"📊 Total documents loaded: {len(documents)}")
    return documents

@functools.lru_cache()
def initialize_model():
    """Initialize the model and retrieval system"""
    try:
        print("🚀 Initializing FLAN-T5-Large model...")
        
        model_name = "google/flan-t5-large"
        print(f"📦 Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": True}
        if TORCH_AVAILABLE:
            try:
                model_kwargs["torch_dtype"] = torch.float16
                print("💾 Using float16")
            except Exception:
                model_kwargs["torch_dtype"] = torch.float32
                print("💾 Using float32")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        print("✅ Model loaded")
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            device=-1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ Pipeline created")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        tax_documents = load_tax_documents()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        texts = text_splitter.create_documents(tax_documents)
        print(f"📚 Created {len(texts)} chunks")
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        tax_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert in Indian tax laws. Answer based on the context.

Context: {context}
Question: {question}

Answer:"""
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": tax_prompt},
            return_source_documents=True
        )
        
        print("🎉 Model ready!")
        return qa_chain
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def get_fallback_answer(question):
    """Rule-based fallback answers"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['80c', 'deduction']):
        return "**Section 80C:** Deduction up to Rs 1.5 lakh for ELSS, PPF, NSC, life insurance, etc."
    elif any(word in question_lower for word in ['gst', 'rate']):
        return "**GST Rates:** 5% (essential), 12% (standard), 18% (most items), 28% (luxury)"
    elif any(word in question_lower for word in ['tds']):
        return "**TDS:** Salary (as per slab), Rent 10%, Professional services 10%"
    else:
        return "Ask about Income Tax, GST rates, TDS, or Section 80C deductions."

def answer_tax_question(question, history):
    """Process tax questions"""
    if not question.strip():
        return "Please ask a tax question."
    
    try:
        qa_chain = initialize_model()
        if qa_chain is None:
            return get_fallback_answer(question)
        
        result = qa_chain({"query": question})
        answer = result["result"]
        
        if answer and len(answer.strip()) > 10:
            return f"**Answer:** {answer.strip()}"
        else:
            return get_fallback_answer(question)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return get_fallback_answer(question)

# Sample questions
sample_questions = [
    "What is Section 80C deduction limit?",
    "What are GST rates?",
    "TDS rate on salary?",
    "Section 44AD details?",
    "Capital gains tax rates?"
]

# Create Gradio interface
with gr.Blocks(title="Tax Assistant", theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1>🏛️ Tax & GST AI Assistant</h1>
        <p>Get answers to Income Tax and GST questions!</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=400,
                type="messages"
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Ask your tax question...",
                    label="Question",
                    lines=2
                )
                submit_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Sample Questions:")
            
            for sample_q in sample_questions:
                btn = gr.Button(sample_q, size="sm")
                btn.click(lambda q=sample_q: q, outputs=question_input)
    
    def submit_question(question, history):
        if question.strip():
            history = history or []
            history.append({"role": "user", "content": question})
            
            response = answer_tax_question(question, history)
            history.append({"role": "assistant", "content": response})
            
            return history, ""
        return history, question
    
    submit_btn.click(
        submit_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )
    
    question_input.submit(
        submit_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
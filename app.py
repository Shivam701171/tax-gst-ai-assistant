import gradio as gr
import os
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import functools
from PyPDF2 import PdfReader

def extract_text_from_pdf(path):
    """Extract text from PDF file"""
    try:
        reader = PdfReader(path)
        return "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF {path}: {str(e)}")
        return ""

def load_tax_documents():
    """Load and process tax documents from PDFs"""
    documents = []
    
    # Define your PDF file paths
    pdf_files = [
        r"C:\Users\Laptop22\OneDrive - 4X4 Advisory Services Private Limited\Desktop\Pandas\Tax assistant\Data(Tax)\income-tax-bill-2025.pdf",  # Update with your actual file path
        "C:\Users\Laptop22\OneDrive - 4X4 Advisory Services Private Limited\Desktop\Pandas\Tax assistant\Data(Tax)\CGST-Act-Updated-30092020.pdf"  # Update with your actual file path
    ]
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"Loading {pdf_file}...")
            text = extract_text_from_pdf(pdf_file)
            if text:
                documents.append(text)
            else:
                print(f"Failed to extract text from {pdf_file}")
        else:
            print(f"PDF file not found: {pdf_file}")
    
    # Fallback to sample data if PDFs not found
    if not documents:
        print("No PDF files found, using sample tax data...")
        documents = [
            "Income Tax Act 1961 Section 80C allows deduction up to Rs 1.5 lakh for investments in ELSS, PPF, NSC, etc.",
            "GST rates: 5% on essential items, 12% on standard items, 18% on most goods and services, 28% on luxury items.",
            "Section 44AD presumptive taxation: 8% of turnover for digital payments, 6% for cash payments.",
            "TDS rates: 10% on salary, 1% on property rent, 2% on contractor payments above Rs 30,000.",
            "GST registration mandatory if turnover exceeds Rs 20 lakh (Rs 10 lakh for special states).",
            "Capital gains tax: Short term 15%, Long term 10% above Rs 1 lakh for equity.",
        ]
    
    return documents

@functools.lru_cache()
def initialize_model():
    """Initialize the model and retrieval system"""
    try:
        # Load model with optimizations for deployment
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Create optimized pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=1024,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Create LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Initialize embeddings for retrieval
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load your actual tax documents
        tax_documents = load_tax_documents()
        
        # Create vector store with proper chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Same as your original code
            chunk_overlap=50
        )
        
        texts = text_splitter.create_documents(tax_documents)
        print(f"Created {len(texts)} document chunks for retrieval")
        
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create custom prompt
        tax_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert in Indian Income Tax and GST laws. 
Based on the provided context, answer the question accurately with:
1. Relevant section numbers or rules
2. Applicable rates, limits, or thresholds
3. Practical examples when helpful

Context: {context}

Question: {question}

Provide a clear, detailed answer:"""
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": tax_prompt,
                "verbose": False
            },
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return None

def answer_tax_question(question, history):
    """Process tax questions and return answers"""
    if not question.strip():
        return "Please ask a specific tax or GST related question."
    
    try:
        # Get the QA chain
        qa_chain = initialize_model()
        if qa_chain is None:
            return "Sorry, there was an issue loading the model. Please try again."
        
        # Get answer
        result = qa_chain({"query": question})
        answer = result["result"]
        
        # Format response with sources
        response = f"**Answer:** {answer}\n\n"
        
        if "source_documents" in result:
            response += "**Sources:**\n"
            for i, doc in enumerate(result["source_documents"][:2]):
                response += f"• {doc.page_content[:100]}...\n"
        
        return response
        
    except Exception as e:
        return f"I apologize, but I encountered an error processing your question: {str(e)}. Please try rephrasing your question."

# Sample questions for easy testing
sample_questions = [
    "What is the rate of TDS on salary?",
    "What are the different GST rates?",
    "How much can I invest under Section 80C?",
    "When is GST registration mandatory?",
    "What is presumptive taxation under 44AD?",
]

# Create Gradio interface
with gr.Blocks(
    title="Tax & GST AI Assistant",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    """
) as demo:
    
    gr.HTML("""
    <div class="header">
        <h1>🏛️ Tax & GST AI Assistant</h1>
        <p>Get instant answers to your Income Tax and GST questions!</p>
        <p><em>Powered by FLAN-T5 and LangChain RAG</em></p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=400,
                placeholder="Ask me anything about Indian Income Tax or GST laws!",
                type="messages"  # Fixed the warning
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Type your tax question here...",
                    label="Your Question",
                    lines=2
                )
                submit_btn = gr.Button("Ask", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Try these sample questions:")
            
            sample_buttons = []
            for sample_q in sample_questions:
                btn = gr.Button(sample_q, size="sm")
                sample_buttons.append(btn)
    
    # Event handlers
    def submit_question(question, history):
        if question.strip():
            # Add user question to chat
            history = history or []
            history.append({"role": "user", "content": question})
            
            # Get AI response
            response = answer_tax_question(question, history)
            history.append({"role": "assistant", "content": response})
            
            return history, ""
        return history, question
    
    def use_sample_question(sample_q):
        return sample_q
    
    # Connect events
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
    
    # Connect sample question buttons
    for i, btn in enumerate(sample_buttons):
        btn.click(
            lambda q=sample_questions[i]: q,
            outputs=question_input
        )
    
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; color: #666;">
        <p>⚠️ <strong>Disclaimer:</strong> This is an AI assistant for educational purposes. 
        Always consult with a qualified tax professional for official advice.</p>
        <p>Built with ❤️ using Hugging Face Transformers & LangChain</p>
    </div>
    """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
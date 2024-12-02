import transformers
import torch
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch import cuda, bfloat16
# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
def setup_and_run_llm(query):
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
    )
    hf_auth = ''
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    # model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
    llm = HuggingFacePipeline(pipeline=generate_text)
    loader = PyPDFLoader("rail_data.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    all_splits = text_splitter.split_documents(documents)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name="roberta-large"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']

# Example usage:
# model_id = 'meta-llama/Llama-2-7b-chat-hf'
# query = "tell me about title?"
# answer = setup_and_run_llm(query)
# print(answer)


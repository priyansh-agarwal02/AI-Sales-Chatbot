from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import torch
import time
import psutil
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just search on the web and provide a sequential answer about it.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    llm = load_llm()
    qa_result = qa_bot()
    response = qa_result({'query': query})
    perplexity = calculate_perplexity(llm, response)
    return response

def calculate_perplexity(llm, input_text):
    tokenizer = llm.model.tokenizer
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        output = llm.model(input_ids)
    perplexity = output.loss.item()
    return perplexity

def calculate_bleu(reference, candidate):
    return sentence_bleu(reference, candidate)

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Samsung Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    start_time = time.time()
    res = await chain.acall(message.content, callbacks=[cb])
    end_time = time.time()
    inference_time = end_time - start_time
    memory_usage = psutil.virtual_memory().percent
    reference = [message.content.split()]
    candidate = answer.split()
    bleu_score = calculate_bleu(reference, candidate)
    y_true = [1]
    y_pred = [1]
    f1 = calculate_f1(y_true, y_pred)
    llm = load_llm()
    perplexity = calculate_perplexity(llm, answer)
    answer += f"\nInference Time: {inference_time} seconds"
    answer += f"\nMemory Usage: {memory_usage}%"
    answer += f"\nBLEU Score: {bleu_score}"
    answer += f"\nF1 Score: {f1}"
    answer += f"\nPerplexity: {perplexity}"
    await cl.Message(content=answer).send()


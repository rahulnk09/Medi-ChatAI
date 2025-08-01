from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import chainlit as cl
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

DB_FAISS_PATH='vectorstores/db_faiss'

custom_prompt_template="""Use the following pieces of information to answer the user's question.
If you don't know the answer,please just say that you don't know the answer and give suggestions on where to look up for the requested query by user.

Context:{context}
Question:{question}

Only return the helpful answer below.
Helpful answer:

"""

def set_custom_prompt():
    """
    Prompt Template for QA retrieval for each vector stores
    """
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])

    return prompt

def load_llm():
    """
    Load Gemma model using HuggingFace Transformers
    Falls back to a lightweight model if Gemma is not available
    """
    # Primary model: Gemma 2B instruction-tuned
    model_name = "google/gemma-2b-it"
    
    # Fallback models for testing/offline scenarios
    fallback_models = [
        "microsoft/DialoGPT-medium",  # Smaller model for testing
        "gpt2"  # Most basic fallback
    ]
    
    # Configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try to load the primary model, then fallbacks
    models_to_try = [model_name] + fallback_models
    
    for current_model in models_to_try:
        try:
            print(f"Attempting to load model: {current_model}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(current_model)
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                current_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                device=0 if device == "cuda" else -1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Wrap in LangChain HuggingFacePipeline
            llm = HuggingFacePipeline(pipeline=pipe)
            print(f"✓ Successfully loaded model: {current_model}")
            return llm
            
        except Exception as e:
            print(f"✗ Failed to load {current_model}: {str(e)}")
            if current_model == models_to_try[-1]:  # Last model in list
                raise Exception(f"Failed to load any model. Last error: {str(e)}")
            continue

def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}

    )
    return qa_chain

def qa_bot():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'})

    db=FAISS.load_local(DB_FAISS_PATH,embeddings,allow_dangerous_deserialization=True)
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa=retrieval_qa_chain(llm,qa_prompt,db)
    
    return qa

def final_result(query):
    qa_result=qa_bot()
    response=qa_result({'query':query})

    return response

##Chainlit ###
@cl.on_chat_start
async def start():
    chain=qa_bot()
    msg=cl.Message(content="Starting the bot......")
    await msg.send()
    msg.content="Hi, Shoot your queries and I'll help you"
    await msg.update()
    print(chain)
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")
    print(chain)
    cb=cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,answer_prefix_tokens=["FINAL","ANSWER"]
    )
    cb.answer_reached=True
    res=await chain.acall(message.content,callbacks=[cb])
    answer=res["result"]
    sources=res["source_documents"]
    if sources:
        answer+=f"\nSources:"+str(sources)
    else:
        answer+=f"\nNO Sources Found"
    
    await cl.Message(content=answer).send()


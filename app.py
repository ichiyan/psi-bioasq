import chainlit as cl 
import argparse
import os
import torch 

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableConfig
from langchain_ollama import OllamaLLM
from langchain.callbacks.base import BaseCallbackHandler


from qdrant_client import QdrantClient

from utils import QdrantMRLRetriever


#Configuration
QDRANT_URL = os.environ["QDRANT_HOST"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = "bioasq-snowflake-multi"
MODEL_NAME = "potsu-potsu/snowflake-embed-mrl-train40k"
MRL_DIMS = [256, 512, 768]
LIMITS = [300,200,5]
OLLAMA_LLM_NAME = "mistral:instruct"



@cl.on_chat_start
async def on_chat_start():

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        template = """Answer the question based only on the following context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        if "snowflake" in MODEL_NAME:
            query_prefix = "Represent this sentence for searching relevant passages: "
        else:
            query_prefix = ""


        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY
        )

        
        retriever = QdrantMRLRetriever(
            client=client, 
            collection_name=COLLECTION_NAME, 
            model_name=MODEL_NAME,
            device=device, 
            mrl_dims=MRL_DIMS,
            limits=LIMITS,
            query_prefix=query_prefix
        )

        llm = OllamaLLM(
            model=OLLAMA_LLM_NAME, 
            base_url="http://localhost:11434",
            device=device,
        )


        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser() 
        )

        cl.user_session.set("runnable", runnable)

        await cl.Message(
                content="🚀 System initialized successfully! Ready to answer your questions."
            ).send()
    
    except Exception as e:
        await cl.Message(
            content=f"❌ Error initializing the system: {str(e)}"
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg 
            self.elements = []
        

        def create_element(self, document: Document) -> cl.Text:
            doc_id = document.metadata["id"]
            score = document.metadata["_score"]
            top_keywords = document.metadata["keywords"].split(", ")
            keywords_preview = top_keywords[:5] if len(top_keywords) > 5 else top_keywords
            content_preview = document.page_content[:500] + "..." if len(document.page_content) > 500 else document.page_content
            content = f"keywords preview: {keywords_preview} \n\n {content_preview}"


            if score >= 0.7:
                emoji = "🟢" 
            elif score >= 0.5:
                emoji = "🟡" 
            else:
                emoji = "🔴"


            return cl.Text(
                name=f"{emoji} Document # {doc_id}: {score:.2f}",
                content=content,
                display="inline", 
                size="small"
            )
        

        def on_retriever_end(self, documents, *, run_id, parent_run_id = None, **kwargs):
            for doc in documents:
                print(doc.metadata["id"], doc.metadata["_score"])
                self.elements.append(self.create_element(document=doc))
            
            if self.elements:
                self.msg.elements = self.elements 
        

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
            PostMessageHandler(msg)
        ]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
        



    







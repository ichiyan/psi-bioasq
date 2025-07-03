import os
from dotenv import load_dotenv, find_dotenv
import torch
from tqdm.auto import tqdm
import argparse


from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents.base import Document
from sentence_transformers import SentenceTransformer


from utils import get_corpus_documents_huggingface

class Color():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'


def create_vector_database(
        url: str, 
        api_key: str, 
        model_name: str, 
        use_mrl_embeddings: bool, 
        mrl_dims: list[int], 
        collection_name: str, 
        documents: list[Document], 
        device: str, 
        batch_size: int=32) -> None:

    try: 
        client = QdrantClient(
            url=url, 
            api_key=api_key, 
        )
    except Exception as e:
        print(f"An exception occurred when trying to connect to Qdrant client: {str(e)}")


    if use_mrl_embeddings and mrl_dims:
        sm_dim, md_dim, lg_dim = mrl_dims

        sm_model = SentenceTransformer(model_name, truncate_dim=sm_dim, device=device)
        md_model = SentenceTransformer(model_name, truncate_dim=md_dim, device=device)
        lg_model = SentenceTransformer(model_name, truncate_dim=lg_dim, device=device)

        try:
            client.create_collection(
                collection_name=collection_name, 
                vectors_config={
                    "small-embeddings": models.VectorParams(size=sm_dim, distance=models.Distance.COSINE),
                    "medium-embeddings": models.VectorParams(size=md_dim, distance=models.Distance.COSINE),
                    "large-embeddings": models.VectorParams(size=lg_dim, distance=models.Distance.COSINE),
                }
            )

            print("Ingesting data...")

            for doc in tqdm(documents):
                client.upsert(
                    collection_name=collection_name, 
                    points = [
                        models.PointStruct(
                            id=doc.metadata["id"],
                            vector={
                                "small-embeddings": sm_model.encode(doc.page_content), 
                                "medium-embeddings": md_model.encode(doc.page_content),
                                "large-embeddings": lg_model.encode(doc.page_content),
                            }, 
                            payload={
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                            }
                        )
                    ]
                )
        except Exception as e:
            print(f"An exception occurred when trying to ingest data: {str(e)}")

    else:
        embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = {'device': device},
            encode_kwargs = {'normalize_embeddings': True}
        )

        print("Ingesting data...")

        try:
            vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                url=url,
                api_key=api_key,
                collection_name=collection_name,
                https=True,
                timeout=300,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"An exception occurred when trying to ingest data: {str(e)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Ingestion to Qdrant")
    parser.add_argument(
        "--qdrant_url", type=str, default=os.environ["QDRANT_HOST"],
        help="Qdrant cloud cluster URL",
    )
    parser.add_argument(
        "--qdrant_api_key", type=str, default=os.environ["QDRANT_API_KEY"],
        help="Qdrant cloud cluster API key",
    )
    parser.add_argument(
        "--collection_name", type=str, default="bioasq-snowflake-multi",
        help="Qdrant database collection name",
    )
    parser.add_argument(
        "--model_name", type=str, default="potsu-potsu/snowflake-embed-mrl-train40k",
        help="embedding model",
    )
    parser.add_argument(
        "--use_mrl_embeddings", action="store_false",    # defaults true
        help="use matryoshka representation learning embeddings",
    )
    parser.add_argument(
        "--mrl_dims", type=int, nargs=3, default=[256, 512, 768],
        help="list of matryoshka embeddings dimensions from smallest to largest",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="batch size for uploading documents (batch upload not implemented for MRL)",
    )
   

    args = parser.parse_args()

    # create global variables without the args prefix
    # for attribute_name in vars(args).keys():
    #     globals()[attribute_name] = getattr(args, attribute_name)

    print(f"\n{Color.CYAN}ARGS:\n")
    for arg_name, arg_value in vars(args).items():
        print(f"{Color.CYAN}{arg_name}: {Color.WHITE}{arg_value}")

    is_ok = input(f"{Color.YELLOW}\nInput 1 to proceed with data ingestion: ")

    if is_ok != "1":
        exit()


    load_dotenv(find_dotenv())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus_documents = get_corpus_documents_huggingface(repo_id="potsu-potsu/mini-bioasq-with-metadata", filename="text-corpus/test-00000-of-00001.parquet")

    create_vector_database(
        url=args.qdrant_url, 
        api_key=args.qdrant_api_key, 
        model_name=args.model_name, 
        use_mrl_embeddings=args.use_mrl_embeddings, 
        mrl_dims=args.mrl_dims, 
        collection_name=args.collection_name,
        documents=corpus_documents,
        batch_size=args.batch_size,
        device=device,
    )


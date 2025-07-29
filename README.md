# Implementation of Information Retrieval Methods on Biomedical Data 

You can read the paper/write-up [here](https://drive.google.com/file/d/1ccvrY4Ber6EdItu_nlBqrGI9VQlfvG_z/view?usp=sharing). 

## Setup:

1. Install dependendencies and activate environment

```
pipenv install
pipenv shell
```

2. Create a copy of .example.env file
```
cp .example.env .env
```
3. Put your environment variables on the `.env` file
4. Run Docker

 ```
 cd psi-docker
 docker-compose up --build
 ```

 This will run Ollama via Docker and automatically download the 4-bit quantized Mistral 7B Instruct model (4.1 GB).
 This is needed to run the "Keyword Search via LLM", "Query Enhancement Using Keywords", "Query Enhancement Using Graph Retrieval" sections in `retrieval.ipynb` and the Chainlit `app.py`.

## Notes:

- The code for fine-tuning the embedding model is located in the `finetune-embedding.ipynb` notebook. Currently, the code is set up to run on [Modal Labs](https://modal.com/).
- The implementation of the different retrieval methods are found in the `retrieval.ipynb` notebook. After running the retrieval across all queries, you can evaluate the results using the `evaluate.ipynb` notebook.
- To run the "Query Enhancement Using Graph Retrieval" section in `retrieval.ipynb`, first ingest the documents to Neo4j using the `ingest-graph.ipynb` notebook. The data ingestion is also set up to run on [Modal Labs](https://modal.com/). 
- The `retrieval.ipynb` notebook already has the code for data ingestion to Qdrant but alternatively, you can use `ingest-qdrant.py` to use either dense retrieval or three-stage retrieval using a model trained with Matryoshka Representation Learning.
```
pipenv shell
python ingest-qdrant.py --help
```
- The Chainlit demo application currently uses the three-stage retrieval approach. It shows some information on the different steps in the chain to highlight the process. After the LLM response to the query, the top 5 retrieved documents and their similarity scores are also shown
  for quick inspection, but more details can be found in the LangSmith trace. There is currently a bug with side display in Chainlit, so the documents are just appended in the main chat area. To run:
```
pipenv shell
chainlit run app.py
```
![image](https://github.com/user-attachments/assets/f200666a-b731-445d-aa30-3b85a9d9d5b1)
![image](https://github.com/user-attachments/assets/1fd89885-875e-4dc0-9fa4-38f32bca79f8)
![image](https://github.com/user-attachments/assets/686ae835-cd21-4339-9f0a-06b91685cb88)
![image](https://github.com/user-attachments/assets/6bd440da-dd6c-4b07-974d-6da0d663208a)



 
 

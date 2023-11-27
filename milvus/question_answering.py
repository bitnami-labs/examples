import os
import pandas as pd
import numpy as np
import csv
import gradio
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import pipe, ops
from towhee.datacollection import DataCollection

# Milvus parameters
host = os.environ["MILVUS_HOST"]
port = os.environ["MILVUS_PORT"]

# Function to create Milvus collection
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', max_length=500, is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)
    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

# Function to perform chat
def chat(message, history):
    history = history or []
    ans_pipe = (
        pipe.input('question')
            .map('question', 'vec', ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
            .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
            .map('vec', 'res', ops.ann_search.milvus_client(host=host, port=port, collection_name='question_answer', limit=1))
            .map('res', 'answer', lambda x: [id_answer[int(i[0])] for i in x])
            .output('question', 'answer')
    )
    response = ans_pipe(message).get()[1][0]
    history.append((message, response))
    return history, history

# Create the 'question_answer' collection in Milvus
df = pd.read_csv('question_answer.csv')
id_answer = df.set_index('id')['answer'].to_dict()
print('-> Creating collection in Milvus')
connections.connect(host=host, port=port)
collection = create_milvus_collection('question_answer', 768)

# Load questions/answers embedding into Milvus
print('-> Loading question embedding into Milvus')
insert_pipe = (
    pipe.input('id', 'question', 'answer')
        .map('question', 'vec', ops.text_embedding.dpr(model_name='facebook/dpr-ctx_encoder-single-nq-base'))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map(('id', 'vec'), 'insert_status', ops.ann_insert.milvus_client(host=host, port=port, collection_name='question_answer'))
        .output()
)
with open('question_answer.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        insert_pipe(*row)

# Ask a question with Milvus and Towhee
print('-> Question query')
collection.load()
ans_pipe = (
    pipe.input('question')
        .map('question', 'vec', ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map('vec', 'res', ops.ann_search.milvus_client(host=host, port=port, collection_name='question_answer', limit=1))
        .map('res', 'answer', lambda x: [id_answer[int(i[0])] for i in x])
        .output('question', 'answer')
)
ans = ans_pipe("If I don't own the car, can I get insurance?")
ans = DataCollection(ans)
ans.show()

# Build a showcase with interface
print('-> Showcase interface')
collection.load()
chatbot = gradio.Chatbot(color_map=("green", "gray"))
interface = gradio.Interface(
    chat,
    ["text", "state"],
    [chatbot, "state"],
    allow_screenshot=False,
    allow_flagging="never",
)
interface.launch(server_name="0.0.0.0", inline=True, share=True)

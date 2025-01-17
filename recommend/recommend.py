from search.embed import Embed
from search.quantize import quantize
from search.retrieve import FaissRetreiver, CosineRetreiver
import numpy as np
from search.rerank import BiEncoderReranker
from search.keywordsearch import KeywordSearch
import warnings
from search.utils import keyword_check, parse_html, Embeddings, Documents
from whoosh.index import open_dir
from aiocache import cached
from aiocache.serializers import JsonSerializer
from typing import List
import asyncio
from search.utils import calculate_time_decay_weights, calculate_weighted_user_embedding, remove_duplicates_keep_last
import os
import streamlit as st
warnings.filterwarnings('ignore')

api_key = os.environ["API_KEY"] if "API_KEY" in os.environ else st.secrets["API_KEY"]

def load_data():
    binary_embd = asyncio.run(Embeddings.load(file_path=r'data\wikipedia-dataset-embeddings-binary.npy', indices='all', extension='npy'))
    int8_embd = asyncio.run(Embeddings.load(r'data\wikipedia-dataset-embeddings-int8.npy', indices='all', extension='npy'))

    return binary_embd, int8_embd

binary_embd, int8_embd = load_data()

def load_retrievers():
    faiss_retreiver = FaissRetreiver(topk=250, doc_embeddings=binary_embd, precision='binary', params='auto')
    emb = Embed(model='gemini', out_dims=256, api_key=api_key)
    cosine = CosineRetreiver(topk=100, threshold=0.75)
    reranker = BiEncoderReranker()
    ks = KeywordSearch()
    index = ks.open_index('data\indexdir')

    return faiss_retreiver, emb, cosine, reranker, ks, index

faiss_retreiver, emb, cosine, reranker, ks, index = load_retrievers()

@cached(ttl=600, serializer=JsonSerializer())
async def embed(query : str):
    result = await emb.generate_embeddings(query)
    return result['embedding']

async def generate_userpref_embedding(documents : List, decay_rate : float) -> np.ndarray:
    result_embed = await embed(documents)
    result_embed = np.array(result_embed['embedding'])
    weights = calculate_time_decay_weights(len(result_embed), decay_rate)
    
    user_pref = calculate_weighted_user_embedding(result_embed, weights.astype('float32'))
    return user_pref
    
if __name__ == "__main__":

    #query = "Who is the PM of India?"

    file_paths = {'titles' : r'data\wikipedia-dataset-titles.h5',
              'summary' : r'data\wikipedia-dataset-summary.h5',
              'description' : r'data\wikipedia-dataset-description.h5'}
    

    user_docs =  asyncio.run(Documents.load(r'data/users/U001.h5', indices='all'))
    user_docs = remove_duplicates_keep_last(user_docs)
    user_docs = user_docs
    user_docs = [doc.decode('utf-8') if isinstance(doc, bytes) else doc for doc in user_docs]

    user_pref = asyncio.run(generate_userpref_embedding(user_docs, -0.01))

    result = user_pref
    binary_query = quantize(result, precision = 'binary', calibration=None)

    faiss_results = faiss_retreiver.retrieve(query_embedding=binary_query, document_indices=list(range(len(binary_embd))), doc_embeddings=binary_embd)
    int8_embd_retreived = np.array([int8_embd[idx] for idx in faiss_results.keys()])
    cosine = CosineRetreiver(topk=100, threshold=0.75)
    cosineresults = cosine.retrieve(query_embedding=result, doc_embeddings=int8_embd_retreived, document_indices=list(faiss_results.values()))

    docs = [asyncio.run(Documents.load(file_path, indices = sorted(list(cosineresults.values()))) for file_path in file_paths.values())]
    doc_indexes = {i : idx for i, idx in enumerate(sorted(list(cosineresults.values())))}

    docs = [parse_html(doc) for doc in docs]
    docs = [doc for doc in zip(*docs)]
    rerank_docs = ['\n'.join(doc) for doc in docs]
    rerank_docs, similarities, documents = reranker.rerank('\n'.join(user_docs[-3:][::-1]), rerank_docs)
    

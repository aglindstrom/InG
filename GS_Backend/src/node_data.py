from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from mediocreatbest import auto
from pydantic import BaseModel

import pandas as pd
import numpy as np
import functools
import logging
import asyncio
import os

app = FastAPI()

logger = logging.getLogger('uvicorn.error')

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@functools.cache
def read_node_data_file():
    node_data = pd.read_csv('./data/node_data.csv.gz', compression='gzip')
    return node_data

@app.get("/node/description")
async def get_node_description(node:str):
    node_data = read_node_data_file()
    return {'node': node, 'description': f"GET: description for {node}"}
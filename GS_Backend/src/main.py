from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

import pandas as pd
import functools
import asyncio
import json
import re

app = FastAPI()

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
def load_nodes():
    nodes = pd.read_csv("data/Medical-Codes.node.csv") 
    nodes = nodes.drop(columns=["x", "y"])
    nodes = nodes.reset_index()
    nodes = nodes.rename(columns={"index":"id"})

    return nodes

@functools.cache
def load_edges():
    edges = pd.read_csv("data/Medical-Codes.edge.csv")
    edges = edges.rename(columns={"src":"source", "tgt":"target"})
    
    return edges

@functools.cache
def load_keys():
    graph = load_nodes()
    graph_types = graph.dtypes
    graph_keys = graph_types.keys().to_list()
    keys = {}

    for i in graph_keys:
        keys[i] = {"type": str(graph_types[i]), "min": int(graph[i].min()), "max": int(graph[i].max()), "default-op": "="}

    return keys

def load_graph():
    graph_nodes = load_nodes()
    graph_edges = load_edges()

    return [graph_nodes, graph_edges]


@app.get("/api/graph/keys/")
async def get_graph_keys():
    return load_keys()

@app.get("/api/graph")
async def get_graph_by_nodes(codes:str="A", depth:int = 0, directions:str = "0"):
    code_list = codes.split(",")
    directions_list = directions.split(",")

    graph_nodes = load_nodes()
    anchors = None
    nodes = None
    edges = None

    for code in code_list:
        code = code.strip()
        node = graph_nodes.loc[graph_nodes['dx10'] == code]
        anchors = pd.concat([anchors, node])

    if type(anchors) == type(None):
        return {"nodes": [], "edges": []}

    if depth == 0:
        return {"nodes": anchors.to_json(orient='records'), "edges": "[]"}
    
    idx = 0

    for anchor in anchors.iterrows():
        if str(directions_list[idx]) == "1" or str(directions_list[idx]) == "0":
            edges = pd.concat([edges, get_edges(anchor[1]['id'], 'source', 'target', depth-1)])
        if str(directions_list[idx]) == "2" or str(directions_list[idx]) == "0":
            edges = pd.concat([edges, get_edges(anchor[1]['id'], 'target', 'source', depth-1)])
        idx += 1

    if edges.empty:
        return {"nodes":anchors.to_json(orient='records'), "edges": "[]"}

    for edge in edges.iterrows():
        nodes = pd.concat([nodes, graph_nodes.loc[graph_nodes['id'] == edge[1]['source']]])
        nodes = pd.concat([nodes, graph_nodes.loc[graph_nodes['id'] == edge[1]['target']]])

    edges = edges.drop_duplicates()            
    edges = edges.sort_values('source')
    nodes = nodes.drop_duplicates()
    nodes = nodes.sort_values('id')


    return {"nodes": nodes.to_json(orient='records'), "edges": edges.to_json(orient='records')}



def get_edges(node, source, target, depth):
    
    graph_edges = load_edges()
    edges = graph_edges.loc[graph_edges[source] == node]

    if depth == 0:
        return edges
    
    e = None

    for edge in edges.iterrows():
        e = pd.concat([e, get_edges(edge[1][target], source, target, depth-1)])
    
    return pd.concat([edges, e])

        

@app.get("/api/arrays")
async def get_arrays(array):
    array = array.split(",")
    return len(array)



'''
@app.get("/api/graph/{filter_string}")
async def get_graph(filter_string: str, depth: int = 0):
    filters = filter_string.split("&")
    split_filter = []
    
    for filter in filters:
        pair = re.split("([=<>])", filter)
        split_filter.append(pair)
    
    graph_nodes = load_nodes()
    graph_edges = load_edges()
    nodes = graph_nodes;
    edges = None

    for filter in split_filter:
        match filter[1]:
            case '=':
                nodes = nodes.loc[nodes[filter[0]] == int(filter[-1])]
            case '>':
                nodes = nodes.loc[nodes[filter[0]] > int(filter[-1])]
            case '<':
                nodes = nodes.loc[nodes[filter[0]] < int(filter[-1])]
 
    if depth == 0:
        for node in nodes.iterrows():
            edges = pd.concat([edges, graph_edges.loc[graph_edges['source'] == node[1]['id']]])
    else: 
        for i in range(0, depth):    
            edges = None
            for node in nodes.iterrows():
                edges = pd.concat([edges, graph_edges.loc[graph_edges['source'] == node[1]['id']]])
            
            if type(edges) == type(None):
                return {"nodes":nodes.to_json(orient="records"), "edges": "{\"error\":\"no data\"}"}

            for edge in edges.iterrows():
                nodes = pd.concat([nodes, graph_nodes.loc[graph_nodes['id'] == edge[1]['target']]])
            
    edges = edges.drop_duplicates()            
    nodes = nodes.drop_duplicates()
    edges = edges.sort_values('source')
    nodes = nodes.sort_values('id')

    return { "nodes":nodes.to_json(orient="records"), "edges":edges.to_json(orient="records") }
'''

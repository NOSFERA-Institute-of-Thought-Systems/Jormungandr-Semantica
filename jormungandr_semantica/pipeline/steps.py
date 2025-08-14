# Imports...
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import networkx as nx
import jormungandr_semantica as js
from umap import UMAP
from sklearn.cluster import KMeans
from bertopic import BERTopic
from hdbscan import HDBSCAN
import pygsp.graphs as graphs
from scipy.sparse.csgraph import connected_components

# PipelineData and abstract classes...
class PipelineData:
    def __init__(self, docs, embeddings, labels_true):
        self.docs, self.embeddings, self.labels_true = docs, embeddings, labels_true
        self.graph, self.representation, self.reduced_representation, self.labels_pred = None, None, None, None
class PipelineStep(ABC):
    def __init__(self, config): self.config = config
    @abstractmethod
    def run(self, data): pass
class GraphConstructor(PipelineStep): pass
class RepresentationBuilder(PipelineStep): pass
class Reducer(PipelineStep): pass
class Clusterer(PipelineStep): pass

# FaissGraphConstructor with robust connectivity check...
class FaissGraphConstructor(GraphConstructor):
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building k-NN graph with Faiss (sparse)...")
        n_points, k = data.embeddings.shape[0], self.config['k']
        neighbors, distances = js.build_faiss_knn_graph(data.embeddings.astype('float32'), k=k)
        
        rows, cols = np.repeat(np.arange(n_points), k), neighbors.flatten()
        sigma = np.mean(distances[:, -1])
        weights = np.exp(-distances.flatten() / (sigma + 1e-8))
        
        adj = sp.coo_matrix((weights, (rows, cols)), shape=(n_points, n_points)).tocsr()
        adj = adj.maximum(adj.T)
        
        epsilon = 1e-5
        adj.data[adj.data < epsilon] = 0
        adj.eliminate_zeros()
        
        n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        
        if n_components == 1:
            adj.setdiag(0)
            data.graph = graphs.Graph(W=adj)
        else:
            print(f"[WARNING] Graph not connected. Found {n_components}. Using largest component.")
            unique, counts = np.unique(labels, return_counts=True)
            largest_idx = np.where(labels == unique[np.argmax(counts)])[0]
            
            sub_adj = adj[largest_idx, :][:, largest_idx]
            data.docs = [data.docs[i] for i in largest_idx]
            data.embeddings = data.embeddings[largest_idx]
            data.labels_true = data.labels_true[largest_idx]
            sub_adj.setdiag(0)
            data.graph = graphs.Graph(W=sub_adj)

        return data

# Simplified Representation Builders...
class DirectRepresentationBuilder(RepresentationBuilder):
    def run(self, data: PipelineData) -> PipelineData:
        data.representation = data.embeddings
        return data

class WaveletRepresentationBuilder(RepresentationBuilder):
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with SGWT...")
        coeffs = js.compute_heat_wavelets(data.graph, data.embeddings, self.config['wavelet_scales'], self.config['n_eigenvectors'])
        n_nodes, n_features, n_scales = coeffs.shape
        data.representation = coeffs.reshape((n_nodes, n_features * n_scales))
        return data

class ACMWRepresentationBuilder(RepresentationBuilder):
    def run(self, data: PipelineData) -> PipelineData:
        print("Step: Building representation with ACMW...")
        coeffs = js.compute_acmw_wavelets(data.graph, data.embeddings, self.config['wavelet_scales'], self.config['n_eigenvectors'])
        n_nodes, n_features, n_scales = coeffs.shape
        data.representation = coeffs.reshape((n_nodes, n_features * n_scales))
        return data

# Reducer and Clusterer...
class UMAPReducer(Reducer):
    def run(self, data: PipelineData) -> PipelineData:
        reducer = UMAP(n_components=self.config['umap_dims'], random_state=self.config['seed'])
        data.reduced_representation = normalize(reducer.fit_transform(data.representation))
        return data

class KMeansClusterer(Clusterer):
    def run(self, data: PipelineData) -> PipelineData:
        num_clusters = len(np.unique(data.labels_true))
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.config['seed'], n_init='auto')
        data.labels_pred = kmeans.fit_predict(data.reduced_representation)
        return data
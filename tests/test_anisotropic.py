# File: tests/test_anisotropic.py

import numpy as np
import pygsp.graphs as graphs
import pytest
from aglt.pipeline.steps import RankSGWTRepresentationBuilder, PipelineData


def test_rank_sgwt_builder_runs_without_error():
    """
    Tests that the RankSGWTRepresentationBuilder, which uses Forman-Ricci
    curvature, runs to completion without error and produces an output of the
    correct shape. This is a more stable test for the anisotropic pipeline.
    """
    # 1. Define parameters and configuration
    n_nodes, n_features, n_scales = 30, 2, 2
    config = {
        "seed": 42,
        "rank_quantile": 0.1,
        "rank_enhancement": 1.5,
        "rank_dampening": 0.5,
        "wavelet_scales": [5, 10],
        "n_eigenvectors": 20,  # Use a small number for a fast test
    }

    # 2. Create a toy graph and a PipelineData object
    G = graphs.Path(N=n_nodes)
    embeddings = np.random.rand(n_nodes, n_features)
    # The builder expects a PipelineData object as input
    data = PipelineData(docs=[], embeddings=embeddings, labels_true=[])
    data.graph = G

    # 3. Instantiate and run the builder
    builder = RankSGWTRepresentationBuilder(config)

    try:
        result_data = builder.run(data)
    except Exception as e:
        pytest.fail(
            f"RankSGWTRepresentationBuilder raised an unexpected exception: {e}"
        )

    # 4. Assert correctness
    expected_shape = (n_nodes, n_features * n_scales)
    representation = result_data.representation

    assert representation is not None, "Representation was not created"
    assert representation.shape == expected_shape, "Output shape is incorrect"
    assert np.all(np.isfinite(representation)), "Output contains non-finite values"

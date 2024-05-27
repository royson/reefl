import torch
import numpy as np 
from io import BytesIO

from functools import reduce
from typing import List, Tuple, Any, Dict, cast

import numpy as np

from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
import numpy.typing as npt

import pdb

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

def softmax(x, T=1.0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x/T) / np.sum(np.exp(x/T), axis=0)

def get_layer(model, layer_name):
    assert layer_name.endswith('.weight') or layer_name.endswith('.bias'), 'layer name must be learnable (end with .weight or .bias'
    layer = model
    for attrib in layer_name.split('.'):
        layer = getattr(layer, attrib)
    return layer

def bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)  # type: ignore
    return cast(NDArray, ndarray_deserialized)

def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]

def aggregate_inplace(results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average. Assumes updated parameters are of the same size"""
    # Count total examples
    num_examples_total = sum([fit_res.num_examples for _, fit_res in results])

    # Compute scaling factors for each result
    scaling_factors = [
        fit_res.num_examples / num_examples_total for _, fit_res in results
    ]

    # Let's do in-place aggregation
    # get first result, then add up each other
    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params

def aggregate_inplace_early_exit(global_sd: Dict[str, NDArrays], clients_local_sd_keys: Dict[str, List[str]], results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average with results of varying sizes."""
    aggregated_sd = {k:np.zeros(v.shape) for k, v in global_sd.items()}

    # Count total examples per sd key
    aggregated_sd_num_examples_total = {k: 0 for k in global_sd.keys()} 
    for client, fit_res in results:
        local_sd_keys = clients_local_sd_keys[client.cid]
        for k in local_sd_keys: 
            aggregated_sd_num_examples_total[k] += fit_res.num_examples
    
    # in-place aggregation!
    for client, fit_res in results:
        local_sd_keys = clients_local_sd_keys[client.cid]
        for k, x in zip(local_sd_keys, parameters_to_ndarrays(fit_res.parameters)):
            weight = fit_res.num_examples / aggregated_sd_num_examples_total[k]
            aggregated_sd[k] += weight * x

    # for keys without updates, use global parameters
    keys_without_update = [k for k, count in aggregated_sd_num_examples_total.items() if count == 0]

    for k in keys_without_update:
        aggregated_sd[k] = global_sd[k]
    
    return list(aggregated_sd.values())

def aggregate_scalefl(global_sd: Dict[str, NDArrays], 
                clients_local_sd_keys: Dict[str, List[str]], 
                results: List[Tuple[ClientProxy, FitRes]],
                is_weight: Dict[str, bool]) -> NDArrays:
    """Compute in-place weighted average with results of varying depth and width."""
    """Adopted & modified from: https://github.com/adap/flower/blob/main/baselines/depthfl/depthfl/strategy_hetero.py"""
    aggregated_sd = {k:np.zeros(v.shape) for k, v in global_sd.items()}

    client_weights = []
    for client, fit_res in results:
        client_weight = dict(zip(clients_local_sd_keys[client.cid], parameters_to_ndarrays(fit_res.parameters)))
        client_weights.append(client_weight) 

    del results

    for k, params in aggregated_sd.items(): # all parameters in global model
        count = np.zeros(params.shape)

        for client_weight in client_weights:
            if k not in client_weight:
                continue
            
            if is_weight[k]:
                grid_indices = torch.meshgrid([torch.arange(size) for size in client_weight[k].shape], indexing='ij')
                aggregated_sd[k][grid_indices] += client_weight[k]
                count[grid_indices] += 1
            else:
                aggregated_sd[k] += client_weight[k]
                count += 1

        aggregated_sd[k][count > 0] = np.divide(aggregated_sd[k][count > 0], count[count > 0])
        aggregated_sd[k][count == 0] = global_sd[k][count == 0] # for parameters without updates, use global params

    return list(aggregated_sd.values())

def aggregate_inplace_early_exit_feddyn(global_sd: Dict[str, NDArrays], 
        clients_local_sd_keys: Dict[str, List[str]], 
        results: List[Tuple[ClientProxy, FitRes]],
        h_dict: Dict[str, NDArrays], # weights and biases only
        num_clients: int,
        alpha: float,
        ) -> NDArrays:
    """Compute in-place FedDyn with results of varying sizes.
        https://arxiv.org/pdf/2111.04263.pdf
        Modified from: https://github.com/adap/flower/blob/main/baselines/depthfl/depthfl/strategy.py
    """
    aggregated_sd = {k:np.zeros(v.shape) for k, v in global_sd.items()}

    # Count total examples per sd key
    aggregated_sd_count = {k: 0 for k in global_sd.keys()} 
    # for client, fit_res in results:
    #     local_sd_keys = clients_local_sd_keys[client.cid]
    #     for k in local_sd_keys: 
    #         aggregated_sd_count[k] += 1
    
    # in-place aggregation!
    for client, fit_res in results:
        local_sd_keys = clients_local_sd_keys[client.cid]
        assert len(local_sd_keys) == len(parameters_to_ndarrays(fit_res.parameters))
        for k, x in zip(local_sd_keys, parameters_to_ndarrays(fit_res.parameters)):
            aggregated_sd[k] += x # take sum without weighting
            aggregated_sd_count[k] += 1

    # update h variable and apply it
    for k, v in aggregated_sd.items():
        if aggregated_sd_count[k] > 0:
            aggregated_sd[k] = v / aggregated_sd_count[k]

            h_dict[k] = (
                    h_dict[k] 
                    - alpha 
                    * aggregated_sd_count[k]
                    * (aggregated_sd[k] - global_sd[k])
                    / num_clients
            )
    
            aggregated_sd[k] = aggregated_sd[k] - h_dict[k] / alpha

    # for keys without updates, use global parameters
    keys_without_update = [k for k, count in aggregated_sd_count.items() if count == 0]

    for k in keys_without_update:
        aggregated_sd[k] = global_sd[k]
    
    return list(aggregated_sd.values())


def aggregate_inplace_early_exit_fedsparseadam(global_sd: Dict[str, NDArrays], 
                                clients_local_sd_keys: Dict[str, List[str]], 
                                results: List[Tuple[ClientProxy, FitRes]],
                                m_t: Dict[str, NDArrays],
                                v_t: Dict[str, NDArrays],
                                beta_1: float,
                                beta_2: float,
                                tau: float,
                                eta: float) -> NDArrays:
    """Compute in-place weighted average with results of varying sizes."""
    aggregated_sd = {k:np.zeros(v.shape) for k, v in global_sd.items()}

    # Count total examples per sd key
    aggregated_sd_num_examples_total = {k: 0 for k in global_sd.keys()} 
    for client, fit_res in results:
        local_sd_keys = clients_local_sd_keys[client.cid]
        for k in local_sd_keys: 
            aggregated_sd_num_examples_total[k] += fit_res.num_examples
    
    # in-place aggregation!
    for client, fit_res in results:
        local_sd_keys = clients_local_sd_keys[client.cid]
        for k, x in zip(local_sd_keys, parameters_to_ndarrays(fit_res.parameters)):
            weight = fit_res.num_examples / aggregated_sd_num_examples_total[k]
            aggregated_sd[k] += weight * x

    # sparse fedadam
    for k, count in aggregated_sd_num_examples_total.items():
        if count > 0:
            # updated model = initial model - grad
            g = global_sd[k] - aggregated_sd[k]
            m_t[k] = np.multiply(beta_1, m_t[k]) + (1 - beta_1) * g            
            v_t[k] = np.multiply(beta_2, v_t[k]) + (1 - beta_2) * np.multiply(g, g)

            aggregated_sd[k] = global_sd[k] - eta * m_t[k] / (np.sqrt(v_t[k]) + tau)

    # for keys without updates, use global parameters
    keys_without_update = [k for k, count in aggregated_sd_num_examples_total.items() if count == 0]

    for k in keys_without_update:
        aggregated_sd[k] = global_sd[k]
    
    return list(aggregated_sd.values())
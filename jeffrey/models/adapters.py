from operator import attrgetter
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.modeling_outputs import BaseModelOutputWithPooling



class Adapter(nn.Module):
    pass


class Normalization(nn.Module):
    def __init__(self, p: float = 2.0) -> None:
        super().__init__()
        self.p = p
        
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=self.p, dim=1)


class FCLayer(Adapter):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        activation_func: Optional[nn.Module] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        order: str = "LBADN",
    ) -> None:
        super().__init__()
        
        order = order.upper()
        
        layers = {
            "L": ("linear", nn.Linear(in_features, out_features, bias=bias))
        }
        
        if activation_func is not None:
            layers["A"] = ("activation_func", activation_func)
        if batch_norm:
            layers["B"] = ("batch_norm", nn.BatchNorm1d(out_features if order.index["B"] > order.index["L"] else in_features))
        if dropout > 0.0:
            layers["D"] = ("dropout", nn.Dropout(dropout))
        if "N" in order:
            layers["N"] = ("normalization", Normalization())
            
        self.layers = nn.Sequential()
        for layer_code in order:
            if layer_code in layers:
                name, layer = layers[layer_code]
                self.layers.add_module(name=name, module=layer)
                
    def forward(self, x: Tensor) -> Tensor:
        output = self.layers(x)
        return output
    

class MLPLayer(Adapter):
    def __init__(
        self,
        output_feature_sizes: List[int],
        biases: Optional[List[bool]] = None,
        activation_funcs: Optional[List[str]] = None,
        dropout_rates: Optional[List[float]] = None,
        batch_norms: Optional[List[bool]] = None,
        order: str = "LBADN",
        standardize_input: bool = True
    ) -> None:
        super().__init__()
        
        self.output_feature_sizes = output_feature_sizes
        self.output_embedding_size = output_feature_sizes[-1]
        
        num_layers = len(self.output_feature_sizes) - 1  # ex: [1024, 512, 256] -> 2
        biases = [False] * num_layers if biases is None else biases
        activation_funcs = [None] * num_layers if activation_funcs is None else activation_funcs
        dropout_rates = [0.0] * num_layers if dropout_rates is None else dropout_rates
        batch_norms = [False] * num_layers if batch_norms is None else batch_norms
        
        assert num_layers == len(biases) == len(activation_funcs) == len(dropout_rates) == len(batch_norms)
        
        self.adapter = nn.Sequential()
        
        if standardize_input:  # Using Layer Normalization
            self.adapter.add_module(
                name="LayerNorm", 
                module=nn.LayerNorm(normalized_shape=output_feature_sizes[0], elementwise_affine=False)
            )
        
        for i in range(num_layers):
            activation_func = activation_funcs[i]
            self.adapter.add_module(
                name=f"fc_layer_{i}",
                module=FCLayer(
                    in_features=output_feature_sizes[i],
                    out_features=output_feature_sizes[i+1],
                    bias=biases[i],
                    activation_func=getattr(nn, activation_func)() if activation_func is not None else None,
                    dropout=dropout_rates[i],
                    batch_norm=batch_norms[i],
                    order=order
                )
            )
            
    def forward(self, backbone_output: Tensor) -> Tensor:
        output = self.adapter(backbone_output)
        return output
    
    
class MLPWithPooling(Adapter):
    def __init__(
        self,
        output_feature_sizes: List[int],
        biases: Optional[List[bool]] = None,
        activation_funcs: Optional[List[str]] = None,
        dropout_rates: Optional[List[float]] = None,
        batch_norms: Optional[List[bool]] = None,
        order: str = "LBADN",
        standardize_input: bool = True,
        pooling_method: Optional[str] = None,
        output_attribute_to_use: Optional[Literal["pooler_output", "last_hidden_state"]] = None
    ) -> None:
        super().__init__()
        
        self.output_feature_sizes = output_feature_sizes
        self.output_embedding_size = output_feature_sizes[-1]
        num_layers = len(output_feature_sizes) - 1
        
        if num_layers > 0:
            self.projection = MLPLayer(
                output_feature_sizes=output_feature_sizes,
                biases=biases,
                activation_funcs=activation_funcs,
                dropout_rates=dropout_rates,
                batch_norms=batch_norms,
                order=order,
                standardize_input=standardize_input
            )
        else:
            self.projection = nn.Identity()
            
        if pooling_method == "mean_pooler":
            self.pooler = mean_pool_tokens
        elif pooling_method == "cls_pooler":
            self.pooler = cls_pool_tokens
        else:
            self.pooler = nn.Identity()
            
        if output_attribute_to_use is not None:
            self.get_output_tensor = attrgetter(output_attribute_to_use)
        else:
            self.get_output_tensor = nn.Identity()
            
    def forward(self, backbone_output: BaseModelOutputWithPooling) -> Tensor:
        output = self.get_output_tensor(backbone_output)
        output = self.pooler(output)
        output = self.projection(output)
        return output
            
        
def mean_pool_tokens(tensor: Tensor) -> Tensor:
    # tensor: (batch_size, num_tokens, embed_size)
    dims = len(tensor.shape)
    if dims != 3:
        raise ValueError(f"Tokens pooling expects exactly 3 dimensional tensor, got: {dims}")
    return torch.mean(tensor, dim=1)

def cls_pool_tokens(tensor: Tensor) -> Tensor:
    # tensor: (batch_size, num_tokens, embed_size)
    dims = len(tensor.shape)
    if dims != 3:
        raise ValueError(f"Tokens pooling expects exactly 3 dimensional tensor, got: {dims}")
    return tensor[:, 0, :]




from torch import Tensor
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        """
        Paper: https://arxiv.org/abs/1910.07467
        """
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.scale


def _activation_quant(x: Tensor) -> Tensor:
    scale: Tensor = 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-5)
    # round to int8
    y: Tensor = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()


def _weight_quant(w: Tensor) -> tuple[Tensor, Tensor]:
    scale: Tensor = 1.0 / w.abs().mean().clamp(min=1e-5)
    # round to 1.58Bit
    quant: Tensor = (w * scale).round().clamp(-1, 1) / scale
    w_quant: Tensor = w + (quant - w).detach()
    scale = abs(w_quant).max().detach()
    w_quant = w_quant / scale
    return w_quant, scale

class BitLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        assert not kwargs.get('bias',False), 'Bias is not supported in BitLinear layer.'
        super(BitLinear, self).__init__(*args, **kwargs)
        self.rms_norm = RMSNorm(self.in_features)

    @torch.compile 
    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.rms_norm(x)
        x_quant = _activation_quant(x_norm)
        w_quant, scale = _weight_quant(self.weight)
        
        output = nn.functional.linear(x_quant, w_quant)
        return output * scale      
        
def _get_bitlinear(linear: nn.Linear):
    
    weight = linear.weight
    
    bitlinear = BitLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
    )
    bitlinear.weight = weight
    
    return bitlinear


def apply_bitlinear(
    model: nn.Module,
    target_layers: list[str] | None = None,
):
    if isinstance(model, nn.Linear):
        return _get_bitlinear(model)

    if isinstance(model, (nn.Module, nn.ModuleDict)):
        for key, value in model._modules.items():
            if isinstance(value, nn.Linear) and (target_layers is None or key in target_layers):
                model._modules[key] = _get_bitlinear(value)
            else:
                apply_bitlinear(value)

    if isinstance(model, (nn.ModuleList, nn.Sequential)) :
        for sub_model in model:
            if isinstance(sub_model, nn.Linear) and (target_layers is None or sub_model in target_layers):
                sub_model = _get_bitlinear(sub_model)
            else:
                apply_bitlinear(sub_model)

    # for name, param in model.named_parameters():
    #     param.requires_grad = True
    return model


if __name__ == "__main__":
    # Create a sample input tensor of shape (n, d)
    # For example, let n = batch size and d = features dimension
    n, d, k = 10, 5, 1024  # n: batch size, d: input features, k: output features
    input_tensor: Tensor = torch.randn(n, d)
    parameterized = True
    # Initialize the BitLinear layer with input features d and output features k
    bit_linear_layer: BitLinear = BitLinear(d, k, bias=False)
    print("bit_linear_layer: ", bit_linear_layer)

    linaer: BitLinear = nn.Linear(d, k, bias=False)
    print("linaer: ", linaer)

    bilinear: BitLinear = apply_bitlinear(linaer)
    print("bilinear: ", bilinear)

    # Run the sample input through the BitLinear layer
    output: Tensor = bit_linear_layer(input_tensor)
    
    # get some explainations on compile
    explaination =  torch._dynamo.explain(bit_linear_layer.forward, bit_linear_layer, input_tensor) 
    print("graph breaks", explaination.graph_break_count)
    # for graph in explaination.graphs:
    #     print(graph.print_tabular())

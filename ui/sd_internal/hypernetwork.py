# this is basically a cut down version of https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/c9a2cfdf2a53d37c2de1908423e4f548088667ef/modules/hypernetworks/hypernetwork.py, mostly for feature parity
# I, c0bra5, don't really understand how deep learning works. I just know how to port stuff.

import inspect
import torch
import optimizedSD.splitAttention
from . import runtime
from einops import rearrange

optimizer_dict = {optim_name : cls_obj for optim_name, cls_obj in inspect.getmembers(torch.optim, inspect.isclass) if optim_name != "Optimizer"}

loaded_hypernetwork = None

class HypernetworkModule(torch.nn.Module):
    multiplier = 0.5
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal',
                 add_layer_norm=False, use_dropout=False, activate_output=False, last_layer_dropout=False):
        super().__init__()

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):

            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func except last layer
            if activation_func == "linear" or activation_func is None or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Add dropout except last layer
            if use_dropout and (i < len(layer_structure) - 3 or last_layer_dropout and i < len(layer_structure) - 2):
                linears.append(torch.nn.Dropout(p=0.3))

        self.linear = torch.nn.Sequential(*linears)

        self.fix_old_state_dict(state_dict)
        self.load_state_dict(state_dict)

        self.to(runtime.thread_data.device)

    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x: torch.Tensor):
        return x + self.linear(x) * runtime.thread_data.hypernetwork_strength

def apply_hypernetwork(hypernetwork, context, layer=None):
    hypernetwork_layers = hypernetwork.get(context.shape[2], None)

    if hypernetwork_layers is None:
        return context, context

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    context_k = hypernetwork_layers[0](context)
    context_v = hypernetwork_layers[1](context)
    return context_k, context_v

def get_kv(context, hypernetwork):
    if hypernetwork is None:
        return context, context
    else:
        return apply_hypernetwork(runtime.thread_data.hypernetwork, context)

# This might need updating as the optimisedSD code changes
# I think yall have a system for this (patch files in sd_internal) but idk how it works and no amount of searching gave me any clue
# just in case for attribution https://github.com/easydiffusion/diffusion-kit/blob/e8ea0cadd543056059cd951e76d4744de76327d2/optimizedSD/splitAttention.py#L171
def new_cross_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    # default context
    context = context if context is not None else x() if inspect.isfunction(x) else x
    # hypernetwork!
    context_k, context_v =  get_kv(context, runtime.thread_data.hypernetwork)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))


    limit = k.shape[0]
    att_step = self.att_step
    q_chunks = list(torch.tensor_split(q, limit//att_step, dim=0))
    k_chunks = list(torch.tensor_split(k, limit//att_step, dim=0))
    v_chunks = list(torch.tensor_split(v, limit//att_step, dim=0))

    q_chunks.reverse()
    k_chunks.reverse()
    v_chunks.reverse()
    sim = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
    del k, q, v
    for i in range (0, limit, att_step):

        q_buffer = q_chunks.pop()
        k_buffer = k_chunks.pop()
        v_buffer = v_chunks.pop()
        sim_buffer = torch.einsum('b i d, b j d -> b i j', q_buffer, k_buffer) * self.scale

        del k_buffer, q_buffer
        # attention, what we cannot get enough of, by chunks

        sim_buffer = sim_buffer.softmax(dim=-1)

        sim_buffer = torch.einsum('b i j, b j d -> b i d', sim_buffer, v_buffer)
        del v_buffer
        sim[i:i+att_step,:,:] = sim_buffer

        del sim_buffer
    sim = rearrange(sim, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(sim)


def load_hypernetwork(path: str):

    state_dict = torch.load(path, map_location='cpu')

    layer_structure = state_dict.get('layer_structure', [1, 2, 1])
    activation_func = state_dict.get('activation_func', None)
    weight_init = state_dict.get('weight_initialization', 'Normal')
    add_layer_norm = state_dict.get('is_layer_norm', False)
    use_dropout = state_dict.get('use_dropout', False)
    activate_output = state_dict.get('activate_output', True)
    last_layer_dropout = state_dict.get('last_layer_dropout', False)
    # this is a bit verbose so leaving it commented out for the poor soul who ever has to debug this
    # print(f"layer_structure: {layer_structure}")
    # print(f"activation_func: {activation_func}")
    # print(f"weight_init: {weight_init}")
    # print(f"add_layer_norm: {add_layer_norm}")
    # print(f"use_dropout: {use_dropout}")
    # print(f"activate_output: {activate_output}")
    # print(f"last_layer_dropout: {last_layer_dropout}")

    layers = {}
    for size, sd in state_dict.items():
        if type(size) == int:
            layers[size] = (
                HypernetworkModule(size, sd[0], layer_structure, activation_func, weight_init, add_layer_norm,
                                   use_dropout, activate_output, last_layer_dropout=last_layer_dropout),
                HypernetworkModule(size, sd[1], layer_structure, activation_func, weight_init, add_layer_norm,
                                   use_dropout, activate_output, last_layer_dropout=last_layer_dropout),
            )
    print(f"hypernetwork loaded")
    return layers



# overriding of original function
old_cross_attention_forward = optimizedSD.splitAttention.CrossAttention.forward
# hijacks the cross attention forward function to add hyper network support
def hijack_cross_attention():
    print("hypernetwork functionality added to cross attention")
    optimizedSD.splitAttention.CrossAttention.forward = new_cross_attention_forward
# there was a cop on board
def unhijack_cross_attention_forward():
    print("hypernetwork functionality removed from cross attention")
    optimizedSD.splitAttention.CrossAttention.forward = old_cross_attention_forward

hijack_cross_attention()
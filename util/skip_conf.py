import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoConfig
from copy import deepcopy


import torch

def informed_choice_no_bumps(jsd_series, window_size=3):
    """
    Make an informed choice for the distribution with the highest JSD in the absence of bumps.
    
    Parameters:
    jsd_series (torch.Tensor): A tensor of JSD values over time.
    window_size (int): The window size for moving average and variance calculations (default is 3).
    
    Returns:
    tuple: The index and value of the chosen distribution.
    """    
    # Calculate the moving average
    if len(jsd_series) <= window_size:
        return jsd_series.argmax().item(), jsd_series.max().item()
    
    moving_avg = torch.nn.functional.avg_pool1d(jsd_series.unsqueeze(0).unsqueeze(0), kernel_size=window_size, stride=1).squeeze()
    
    # Manually replicate padding for moving_avg
    pad_size = (window_size - 1) // 2
    moving_avg_padded = torch.cat([moving_avg[0].repeat(pad_size), moving_avg, moving_avg[-1].repeat(pad_size)])
    
    # Calculate the moving variance
    moving_var = torch.nn.functional.avg_pool1d((jsd_series.unsqueeze(0).unsqueeze(0) - moving_avg_padded.unsqueeze(0).unsqueeze(0))**2, kernel_size=window_size, stride=1).squeeze()
    moving_var_padded = torch.cat([moving_var[0].repeat(pad_size), moving_var, moving_var[-1].repeat(pad_size)])
    
    # Combine the moving average and variance to make a choice
    stability_score = moving_avg_padded / (moving_var_padded + 1e-6)
    
    # Select the index with the highest stability score
    chosen_index = stability_score.argmax().item()
    chosen_value = jsd_series[chosen_index].item()
    
    return chosen_index, chosen_value, 0

def detect_and_rank_bumps(jsd_series, threshold=0):
    """
    Detect and rank significant bumps in a time-series of Jensen-Shannon Divergence values using PyTorch tensors.
    
    Parameters:
    jsd_series (torch.Tensor): A tensor of JSD values over time.
    threshold (float): The minimum increase considered as a bump to avoid noise (default is 0.01).
    
    Returns:
    tuple: The most significant bump (index, value, magnitude) based on ranking criteria or None if no significant bumps.
    """
    jsd_series = torch.tensor(jsd_series)
    
    # Calculate the differences between consecutive elements
    diffs = jsd_series[1:] - jsd_series[:-1]
    
    # Identify where the differences exceed the threshold
    bumps_mask = diffs > threshold
    
    if not bumps_mask.any():
        return informed_choice_no_bumps(jsd_series)
    
    # Extract indices and values of bumps
    bump_indices = torch.nonzero(bumps_mask).flatten() + 1
    bump_values = jsd_series[bump_indices]
    bump_magnitudes = diffs[bumps_mask]
    
    # Find the most significant bump
    max_magnitude_index = torch.argmax(bump_magnitudes)
    most_significant_bump = (bump_indices[max_magnitude_index].item(), 
                             bump_values[max_magnitude_index].item(), 
                             bump_magnitudes[max_magnitude_index].item())
    
    return most_significant_bump


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor):
        # Move p and q to CPU and ensure they are in float64 for high precision calculation
        p, q = p.cpu().double(), q.cpu().double()
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
    
def softmax_confidence(
    logits: torch.Tensor = None,
):  
    # start = datetime.datetime.now()
    assert logits is not None
    probs = torch.softmax(logits, dim=-1)
    top_2 = torch.topk(probs, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for softmax confidence", end-start)

    return (top_2[..., 0] - top_2[..., 1]).squeeze()
    

def meta_confidence(
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
):
    assert hidden_states is not None
    assert classifier is not None
    
    preds = classifier(hidden_states)
    probs = torch.softmax(preds, dim=-1)
    return probs[..., 1].squeeze()


def contrastive_confidence(  
    lm_logits: torch.Tensor = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = None,
):
    """
    Checking confidence with contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """

    assert lm_logits is not None


    ## calculate current layer probabilities
    # start = datetime.datetime.now()
    probits_exp = torch.softmax(lm_logits, dim=-1)
    probits_exp = torch.squeeze(probits_exp)
    prev_probits[layer_exp] = probits_exp
   
    # probs_exp = torch.softmax(logits_at, dim=-1)
    max_probs_exp = torch.max(probits_exp)

    ## obtaining the correct layer probit values from previous layers (the layer index is choosen to be usually half of the current layer). 
    if layer_am in prev_probits.keys():
        probits_am = prev_probits[layer_am]
    else:
        raise ValueError("Choosen layer has not been computed yet")
    
    ## calculating the scores using the plausibility constraint
    s = deepcopy(probits_exp)

    mask = probits_exp >= alpha * max_probs_exp
    s[mask] = torch.softmax(torch.log(probits_exp[mask]) - torch.log(probits_am[mask]), dim=-1) 
    top_2 = torch.topk(s, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for contrastive confidence", end-start)
    
    return (top_2[..., 0] - top_2[..., 1]).squeeze()

def reweight_contrastive_confidence(  
    lm_logits: torch.Tensor = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
):
    """
    Checking confidence with reweighted contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """
    # start = datetime.datetime.now()
    assert lm_logits is not None


    ## calculate current layer probabilities
    probits_exp = torch.softmax(lm_logits, dim=-1).squeeze_()
    prev_probits[layer_exp] = probits_exp
    # print("prev_probits", prev_probits) 
    # probs_exp = torch.softmax(logits_at, dim=-1)
    max_probs_exp = torch.max(probits_exp)

    ## obtaining the correct layer probit values from previous layers (the layer index is choosen to be usually half of the current layer). 
    if layer_am in prev_probits.keys():
        probits_am = prev_probits[layer_am]
    else:
        raise ValueError("Choosen layer has not been computed yet")


    s = torch.zeros_like(probits_exp)
    mask = probits_exp >= alpha * max_probs_exp

    # start = datetime.datetime.now()
    contrast = torch.softmax(torch.log(probits_exp[mask]) - torch.log(probits_am[mask]), dim=-1).mul_(torch.sum(probits_exp[mask]))
    s[mask] = contrast
    
    top_2 = torch.topk(s, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for contrastive confidence", end-start)
    
    return (top_2[..., 0] - top_2[..., 1]).squeeze()
 
def JSD_contrastive_confidence(  
    lm_logits: torch.Tensor = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    alpha: float = None,
    return_jsds=True,
):
    """
    Checking confidence with JSD contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """
    # start = datetime.datetime.now()
    assert lm_logits is not None
    ## calculate current layer probabilities
    probits_exp = torch.softmax(lm_logits, dim=-1).squeeze_()
    prev_probits[layer_exp] = probits_exp
    # print("prev_probits", prev_probits) 

   
    # probs_exp = torch.softmax(logits_at, dim=-1)
    max_probs_exp = torch.max(probits_exp)

    ## obtaining the correct layer probit values from previous layers (the layer index is choosen to be usually half of the current layer). 
    # if layer_am in prev_probits.keys():
    #     probits_am = prev_probits[layer_am]
    # else:
    #     raise ValueError("Choosen layer has not been computed yet")


    # Calculate Jensen-Shannon Divergence between the current and previous layer
    # probs_am = torch.softmax(logits_am, dim=-1)
    # probs_am = torch.squeeze(probs_am)
    # probs_exp = torch.squeeze(probs_exp)
    # m = 0.5 * (probs_am + probs_exp)
    # jsd = 0.5 * (torch.sum(probs_am * torch.log(probs_am / m)) + torch.sum(probs_exp * torch.log(probs_exp / m)))

    jsd = JSD()
    #jsds = {k: jsd(probits_exp, v) for k, v in prev_probits.items()}

    # only consider jsds between current and 2nd layer
    mask = probits_exp >= alpha * max_probs_exp
    jsds = {layer: jsd(probits_exp[mask], prev_probits[layer][mask]) for layer in np.arange(stop = layer_exp + 1, start=2)}
    # get the probits with the maximum jsd
    #max_jsd_layer = max(jsds, key=jsds.get)

    # Get list of jsds values
    vals = list(jsds.values())
    o = detect_and_rank_bumps(vals)
    bump_idx = o[0]
    probits_am = prev_probits[bump_idx + 2]

    # for v in prev_probits.values():
    #     probs_am = v
    #     jsd_val = jsd(probits_exp, probs_am)
    #     jsds.append(jsd_val)
    
    # max_jsd = torch.max(torch.stack(jsds))

    
    ## calculating the scores using the plausibility constraint
    # s = deepcopy(probits_exp)

    s = torch.zeros_like(probits_exp)
    contrast = torch.log(probits_exp[mask]) - torch.log(probits_am[mask])
    s.masked_fill_(mask, contrast[0])
    # DoLA Implementation:
    s.masked_fill_(~mask, -1e9)
    s = torch.softmax(s, dim=-1).mul_(torch.sum(probits_exp))

    #plot_probits(s, title='Reweighted Contrastive Confidence, layer_exp: {}, layer_am: {}'.format(layer_exp, max_jsd_layer))

    # TODO: (joan) test also against the scaling being done within the softmax 
    # TODO (joan): Assess JSD between distributions to see what is the best way to do this

    top_2 = torch.topk(s, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for contrastive confidence", end-start)
    
    if return_jsds:
        return (top_2[..., 0] - top_2[..., 1]).squeeze(), jsds
    else:
        return (top_2[..., 0] - top_2[..., 1]).squeeze()


def get_confidence_class(key):

    _conf_class_map = {
        'softmax': softmax_confidence,
        'meta': meta_confidence,
        'contrastive_decoding': contrastive_confidence,
        'reweight_contrastive_decoding': reweight_contrastive_confidence,
        'JSD_contrastive_confidence':  JSD_contrastive_confidence,
    }

    if key in _conf_class_map:
        return _conf_class_map[key]
    else:
        raise ValueError('Invalid confidence measure: {}'.format(key))

def get_skip_mask_cd(
    lm_logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = 0.1,
    adapt_threshold: float = None,
    return_conf=False,
    return_jsds=True,
):

    assert config.exit_conf_type is not None or config.shallow2deep_conf_type is not None

    if config.exit_conf_type is not None:
        key = config.exit_conf_type
        if config.exit_position_temp is not None:
            # decays the confidence threshold with decoding time stp.        
            correct_by_pos = lambda i: config.exit_conf_threshold * np.exp(
                - config.exit_position_temp * i / config.max_answer_length
            ) / 10 + 9 * config.exit_conf_threshold / 10
            threshold = correct_by_pos(pos_time)
        else:
            threshold = config.exit_conf_threshold
    elif config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = config.shallow2deep_conf_threshold if adapt_threshold is None else adapt_threshold

    conf_measure = get_confidence_class(key=key)    

    # print("Inside get_skip_mask_cd")

    if key == 'JSD_contrastive_confidence' and not return_jsds:
        conf = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            layer_am = layer_am,
            alpha = alpha,
            hidden_states = hidden_states,
            classifier = classifier,
        )
    elif key == 'JSD_contrastive_confidence' and return_jsds:
        conf, jsds = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            alpha = alpha,
            return_jsds = return_jsds,
        )
    else:
        conf = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            layer_am = layer_am,
            alpha = alpha,
            hidden_states = hidden_states,
            classifier = classifier,
        )
    


    # print("confidence return", conf)

    mask = torch.where(conf <= threshold, 0., 1.).bool()

    # print("Are we early exiting?", mask.item() == 1)
    # print('Confidence:', conf.item(), 'Threshold:', threshold, 'Mask:', mask.item())

    # print("mask", mask)
    # print("mask shape", mask.shape)

    if return_jsds:
        return mask.item(), jsds
    
    if not return_conf:
        return mask.item()  # False (0) and True (1) denote keep and exit
    else:
        return mask.item(), conf.item()
    


def get_skip_mask(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    adapt_threshold: float = None,
    return_conf=False,
):
    assert config.exit_conf_type is not None or config.shallow2deep_conf_type is not None

    if config.exit_conf_type is not None:
        key = config.exit_conf_type
        if config.exit_position_temp is not None:
            # decays the confidence threshold with decoding time stp.        
            correct_by_pos = lambda i: config.exit_conf_threshold * np.exp(
                - config.exit_position_temp * i / config.max_answer_length
            ) / 10 + 9 * config.exit_conf_threshold / 10
            threshold = correct_by_pos(pos_time)
        else:
            threshold = config.exit_conf_threshold
    elif config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = config.shallow2deep_conf_threshold if adapt_threshold is None else adapt_threshold

    conf_measure = get_confidence_class(key=key)    
    
    conf = conf_measure(
        logits=logits
        )
    
    mask = torch.where(conf <= threshold, 0., 1.).bool()

    # print(f"Confidence: {conf.item():.4f}, Threshold: {threshold:.4f}, Mask: {mask.item()}")
    
    # print("Are we early exiting?", mask.item() == 1)
    # print('Confidence:', conf.item(), 'Threshold:', threshold, 'Mask:', mask.item())


    if not return_conf:
        return mask.item()  # False (0) and True (1) denote keep and exit
    else:
        return mask.item(), conf.item()
import numpy as np
import torch

from transformers import AutoConfig
from copy import deepcopy
import datetime



def softmax_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
):  
    # start = datetime.datetime.now()
    assert logits is not None
    probs = torch.softmax(logits, dim=-1)
    top_2 = torch.topk(probs, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for softmax confidence", end-start)

    return (top_2[..., 0] - top_2[..., 1]).squeeze()


def meta_confidence(
    logits: torch.Tensor = None,
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
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
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
    Checking confidence with contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """
    # start = datetime.datetime.now()
    assert lm_logits is not None


    ## calculate current layer probabilities
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


    s[mask] = torch.softmax(torch.log(probits_exp[mask]) - torch.log(probits_am[mask]), dim=-1) * torch.sum(probits_exp[mask])


    top_2 = torch.topk(s, dim=-1, k=2)[0]
    # end = datetime.datetime.now()
    # print("Time taken for contrastive confidence", end-start)
    
    return (top_2[..., 0] - top_2[..., 1]).squeeze()


def get_confidence_class(key):

    _conf_class_map = {
        'softmax': softmax_confidence,
        'meta': meta_confidence,
        'contrastive_decoding': contrastive_confidence,
        'reweight_contrastive_decoding': reweight_contrastive_confidence,
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
        logits=logits, 
        hidden_states=hidden_states, 
        classifier=classifier,
        )
    
    mask = torch.where(conf <= threshold, 0., 1.).bool()

    # print("Are we early exiting?", mask.item() == 1)
    # print('Confidence:', conf.item(), 'Threshold:', threshold, 'Mask:', mask.item())
    
    
    if not return_conf:
        return mask.item()  # False (0) and True (1) denote keep and exit
    else:
        return mask.item(), conf.item()
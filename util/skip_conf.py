import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoConfig
from copy import deepcopy


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
    

def plot_probits(probits, title='Probability Distribution Over Large Vocabulary', layer_exp=None, layer_am=None):
    """
    Plot the probability distribution for a large vocabulary indexed from 0 to len(probits)-1.

    Parameters:
        probits (list or np.array): The probabilities of the vocabulary terms.
        title (str): Title of the plot.

    Returns:
        None. Displays the plot.
    """

    probits = probits.cpu().detach().numpy()

    # Setting the style and context for a publication-quality plot
    sns.set(style="whitegrid", context="talk", palette="muted")

    # Create a new figure and set its size for better readability in papers
    plt.figure(figsize=(12, 6))

    # Use numpy to create an array of indices for the x-axis
    indices = np.arange(len(probits))

    # Creating the line plot for large vocabulary
    plt.plot(indices, probits, marker='', color='deepskyblue', linewidth=1.5)

    # Adding title and labels with enhancements
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Vocabulary Index', fontsize=14)
    plt.ylabel('Probability', fontsize=14)

    # Limit the display of the x-axis for clarity
    plt.xlim(0, len(probits))

    # Tight layout often provides a cleaner look especially when saving figures
    plt.tight_layout()

    # Save the figure as a high-resolution PNG, which is often used in papers
    plt.savefig(title, dpi=300)

    # Show the plot
    plt.show()

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
 
def JDS_contrastive_confidence(  
    lm_logits: torch.Tensor = None,
    layer_exp: int = None, 
    prev_probits: dict = None, 
    layer_am: int = None,
    alpha: float = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    return_jsds=True,
):
    """
    Checking confidence with JDS contrastive decoding.
    First we are computing the V_head, meaning the plausibility constraint.
    Second we are getting the probits from previous iterations and comparing with a fixed one by taking the log difference.
    
    """
    # start = datetime.datetime.now()
    assert lm_logits is not None

    ## calculate current layer probabilities
    probits_exp = torch.softmax(lm_logits, dim=-1).squeeze_()
    prev_probits[layer_exp] = probits_exp
   
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

    # only consider jsds between current and current // 2 layers
    jsds = {layer: jsd(probits_exp, prev_probits[layer]) for layer in np.arange(stop = layer_exp + 1, start=1)}
    # get the probits with the maximum jsd
    max_jsd_layer = max(jsds, key=jsds.get)
    probits_am = prev_probits[max_jsd_layer]

    # for v in prev_probits.values():
    #     probs_am = v
    #     jsd_val = jsd(probits_exp, probs_am)
    #     jsds.append(jsd_val)
    
    # max_jsd = torch.max(torch.stack(jsds))

    
    ## calculating the scores using the plausibility constraint
    # s = deepcopy(probits_exp)

    s = torch.zeros_like(probits_exp)
    mask = probits_exp >= alpha * max_probs_exp
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
        'JDS_contrastive_confidence':  JDS_contrastive_confidence,
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

    if key == 'JDS_contrastive_confidence' and not return_jsds:
        conf = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            layer_am = layer_am,
            alpha = alpha,
            hidden_states = hidden_states,
            classifier = classifier,
        )
    elif key == 'JDS_contrastive_confidence' and return_jsds:
        conf, jsds = conf_measure(
            lm_logits,
            layer_exp = layer_exp, 
            prev_probits = prev_probits, 
            layer_am = layer_am,
            alpha = alpha,
            hidden_states = hidden_states,
            classifier = classifier,
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

    # print("Are we early exiting?", mask.item() == 1)
    # print('Confidence:', conf.item(), 'Threshold:', threshold, 'Mask:', mask.item())
    
    
    if not return_conf:
        return mask.item()  # False (0) and True (1) denote keep and exit
    else:
        return mask.item(), conf.item()
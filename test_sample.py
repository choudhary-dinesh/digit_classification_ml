
from utils import get_list_of_param_comination

def test_get_list_of_param_comination():
    gamma_values = [0.001, 0.002, 0.005, 0.01, 0.02]
    C_values = [0.1, 0.2, 0.5, 0.75, 1]
    list_of_param = [gamma_values, C_values]
    param_names =  ['gamma', 'C']
    assert len(get_list_of_param_comination(list_of_param, param_names)) == len(gamma_values) * len(C_values)

def test_get_list_of_param_comination_values():
    gamma_values = [0.001, 0.01] 
    C_values = [0.1]
    list_of_param = [gamma_values, C_values]
    param_names =  ['gamma', 'C']
    hparams_combs = get_list_of_param_comination(list_of_param, param_names)
    assert ({'gamma':0.001, 'C':0.1} in hparams_combs) and  ({'gamma':0.01, 'C':0.1} in hparams_combs)

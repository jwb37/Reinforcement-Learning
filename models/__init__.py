from models.DeepQ.Model import DeepQModel
from models.PolicyGradient.AC import ActorCritic

from Params import Params

def load_model():
    if Params.model == 'DeepQ':
        return DeepQModel()
    elif Params.model == 'ActorCritic':
        return ActorCritic()

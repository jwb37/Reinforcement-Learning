from Params import Params
from models import load_model

model = load_model()
model.prepare_training()
model.train()

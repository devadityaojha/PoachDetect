import torch
from model_utils import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = Model().to(device)
classifier.load_state_dict(torch.load("./model_parameters.tar", map_location=device))


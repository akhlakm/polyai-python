#!/usr/bin/env python

"""
Example usages of helpers to per NER using a BERT model.

"""

import polyai.api
from polyai.api.util import create_ssh_tunnel
from polyai.api.helpers import (
    generation_time, tok_per_sec,
    instruct_prompt, model_ner
)

# Load the env variables and set the API key.
import os
from dotenv import load_dotenv
load_dotenv()
polyai.api.api_key = os.environ.get("POLYAI_API_KEY")

# Uncomment this to use ssh tunnel. Make sure to
# add hostname, username and password in the .env file.
# create_ssh_tunnel()

print("Sending api request.")
resp = polyai.api.BERTNER.create(
    model="polyai", # currently ignored by the server.
    text="In this paper, we have studied the dynamics and relaxation of charge carriers in poly(methylmethacrylate)-lithium salt based polymer electrolytes plasticized with ethylene carbonate. Structural and thermal properties have been examined using X-ray diffraction and differential scanning calorimetry, respectively. We have analyzed the complex conductivity spectra by using power law model coupled with the contribution of electrode polarization at low frequencies and high temperatures. The temperature dependence of the ionic conductivity and crossover frequency exhibits Vogel-Tammann-Fulcher type behavior indicating a strong coupling between the ionic and the polymer chain segmental motions. The scaling of the ac conductivity indicates that relaxation dynamics of charge carriers follows a common mechanism for all temperatures and ethylene carbonate concentrations. The analysis of the ac conductivity also shows the existence of a nearly constant loss in these polymer electrolytes at low temperatures and high frequencies. The fraction of free anions and ion pairs in polymer electrolyte have been obtained from the analysis of Fourier transform infrared spectra. It is observed that these quantities influence the behavior of the composition dependence of the ionic conductivity."
)

for tag in model_ner(resp):
    print(tag['text'], "==", tag['label'])


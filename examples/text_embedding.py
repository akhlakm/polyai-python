#!/usr/bin/env python

"""
Example usages of helpers to calculate text embeddings using a LLM.

"""

import polyai.api
from polyai import sett
from polyai.api.util import create_ssh_tunnel
from polyai.api.helpers import embeddings

# Load the env variables and set the API key.
sett.load_api_settings()
polyai.api.api_key = sett.API.polyai_api_key
polyai.api.api_base = sett.API.polyai_api_base

# Add hostname, username and password in the settings.yaml file.
create_ssh_tunnel()

text="In this paper, we have studied the dynamics and relaxation of charge carriers in poly(methylmethacrylate)-lithium salt based polymer electrolytes plasticized with ethylene carbonate. Structural and thermal properties have been examined using X-ray diffraction and differential scanning calorimetry, respectively. We have analyzed the complex conductivity spectra by using power law model coupled with the contribution of electrode polarization at low frequencies and high temperatures. The temperature dependence of the ionic conductivity and crossover frequency exhibits Vogel-Tammann-Fulcher type behavior indicating a strong coupling between the ionic and the polymer chain segmental motions. The scaling of the ac conductivity indicates that relaxation dynamics of charge carriers follows a common mechanism for all temperatures and ethylene carbonate concentrations. The analysis of the ac conductivity also shows the existence of a nearly constant loss in these polymer electrolytes at low temperatures and high frequencies. The fraction of free anions and ion pairs in polymer electrolyte have been obtained from the analysis of Fourier transform infrared spectra. It is observed that these quantities influence the behavior of the composition dependence of the ionic conductivity."

print("Sending api request.")
resp = polyai.api.TextEmbedding.create(
    model="polyai", # currently ignored by the server.
    text=text,
)

print("Text =", text)
print("Embeddings =", embeddings(resp))

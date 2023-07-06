#!/usr/bin/env python

"""
Example usages of helpers to instruct the LLM
with two-shot learning.

"""

from dotenv import load_dotenv
from polyai.api.util import create_ssh_tunnel
from polyai.api.helpers import (
    generation_time, tok_per_sec,
    instruct_prompt, model_reply
)

# Load the env variables.
load_dotenv()
create_ssh_tunnel()

print("Sending api request.")
resp, ptok, ctok, req = instruct_prompt(
    model="polyai",
    instruction="Answer the user questions in a helpful way.",
    prompt="""
Here is a table, it's caption and relevant description.

Description:
PU matrixTo be able to distinguish the dielectric properties of the pure matrix from that of composites in the experimental dielectric spectra, we first consider the properties of the PU matrix.
PU UR 3420 is a thermoplastic polymer with semicrystalline morphology, which is clearly indicated by DSC (Table I).
The DSC results show a glass transition (Tg) around a80aAdegC and the following melting points (Tm) of crystalline phases: first around a13aAdegC and the second around +4aAdegC.
inset).
It should be noted that the glass transition temperature of a HC is lower than that of pure PU by about 40aAdegC (Table I).
Table I.
It should be noted that the glass transition temperature of a HC is lower than that of pure PU by about 40aAdegC (Table I).
Table I.
The main characteristic temperatures of PU and HC: DSC measurements.
FIG.
PPT|High-resolutionIn the low-temperature interval of a30a10aAdegC (see Table I for melting points), there is another relaxation process.
Figure 1 shows that the real part of permittivity in this temperature interval depends strongly on frequency; this fact manifests itself in the peaks of Iua3(f).

Caption:
Table I. The main characteristic temperatures of PU and HC: DSC measurements. (Tg is the glass transition temperature, Tc is the crystallization temperature, Tm is the melting temperature, and Tdeg is the degradation temperature.)

Table:
          1       2        3        4        5        6       7         8
0  Material  Tg (C)  Tc1 (C)  Tc2 (C)  Tm1 (C)  Tm2 (C)  Tm (C)  Tdeg (C)
1        PU      98       60       36        4       13     117       299
2        HC     136                37        1       11     112       272

Infer the method of the measurements. Do not explain your thoughts.
Finally, extract the Tg values as JSONL format with the fields: PolymerName, PropertyName, PropertyValue, PropertyUnit, MeasurementMethod.
""",
    shots = [],
    temperature = 0.0001,
    top_k = 0,
    max_tokens = 1048,
)

print("\nResponse stats:")
print("Tokens (prompt completion) =", ptok, ctok)
print("Generation time =", generation_time(resp)/1000, "sec")
print("Tokens per second =", tok_per_sec(resp), "\n")
print(model_reply(resp))

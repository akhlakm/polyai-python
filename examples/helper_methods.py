#!/usr/bin/env python

"""
Example usages of helpers to instruct the LLM
with two-shot learning.

"""

from polyai.api.helpers import (
    generation_time, tok_per_sec,
    instruct_prompt, model_reply
)

import polyai.api as polyai
import polyai.sett as sett

try:
    sett.load_api_settings()
except:
    pass
sett.save_api_settings()

polyai.create_ssh_tunnel()  # if needed

polyai.api_key = sett.API.polyai_api_key
polyai.api_base = sett.API.polyai_api_base

print("Sending api request.")
resp, ptok, ctok, req = instruct_prompt(
    model="polyai",
    instruction="Answer the user questions in a helpful way.",
    prompt="""
Here is a table, it's caption and relevant description.

Description:
PPT|High-resolutionDSC traces for different polymer electrolytes, taken during second heating process to remove any thermal history, are shown in Fig.
The glass transition temperature (Tg), melting temperature (Tm), and melting enthalpy (IHm) have been calculated from the DSC data and are listed in Table I.
K with the addition of PC to PEO-LiClO4 polymer electrolyte.
Good fits to VTF formula in the entire temperature range clearly indicate that the Li+ ionic motion is coupled with the polymer segmental motion in these polymer electrolytes.
The parameters obtained from VTF fits are shown in Table I.
It is observed in the table that the values of (TgaT0) are in the range 70 K - 80 K, which is consistent with the values observed for other PEO bases electrolytes.

Caption:
TABLE I. Glass transition and melting temperatures, percentage of crystalline phase (XC %), Ea and T0 obtained from VTF formalism for PEO-LiClO4-X wt. % PC electrolytes.

Table:
                      1             2              3       4                5           6
0  PC content X (wt. %)  Tg (K)(0.02)  Tm (K) (0.02)  XC (%)  Ea (eV) (0.002)  T0 (K) (2)
1                     0        244.79         334.38   39.08            0.094         174
2                    10        243.96         331.63   37.81            0.107         164
3                    20        242.73         330.84    37.9            0.146         154
4                    30        242.21         326.98   36.01            0.073         171
5                    40        233.81         323.27    23.7            0.092         157
6                    50        245.84         332.48    34.7            0.112         167

Infer the method of the measurements for the Tg values. Do not explain your thoughts.
Finally, extract only the Tg values as JSONL format with the fields: PolymerName, PropertyName, PropertyValue, PropertyUnit, MeasurementMethod, MeasurementComposition.
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

## Sample output: Guanaco 65B
## ---------------------------------------------------------------
# Sending api request.

# Response stats:
# Tokens (prompt completion) = 658 615
# Generation time = 129.992 sec
# Tokens per second = 9.79

# The table in question is a tabular representation of data obtained from Differential Scanning Calorimetry (DSC) traces for different polymer electrolytes. It contains information on glass transition temperature (Tg), melting temperature (Tm), percentage of crystalline phase (XC %), activation energy (Ea), and reference temperature (T0).

# The Tg value can be inferred to have been measured using differential scanning calorimetry (DSC), which is a common technique used to study thermal properties of materials. This inference is based on the fact that the table includes DSC traces for the different polymer electrolytes.

# Here's the extracted JSONL data:

# ```json
# [
#   { "PolymerName": "PEO-LiClO4", "PropertyName": "Tg", "PropertyValue": "244.79", "PropertyUnit": "K", "MeasurementMethod": "DSC", "MeasurementComposition": "(0.02)" },
#   { "PolymerName": "PEO-LiClO4-10 wt. % PC", "PropertyName": "Tg", "PropertyValue": "243.96", "PropertyUnit": "K", "MeasurementMethod": "DSC", "MeasurementComposition": "(0.02)" },
#   { "PolymerName": "PEO-LiClO4-20 wt. % PC", "PropertyName": "Tg", "PropertyValue": "242.73", "PropertyUnit": "K", "MeasurementMethod": "DSC", "MeasurementComposition": "(0.02)" },
#   { "PolymerName": "PEO-LiClO4-30 wt. % PC", "PropertyName": "Tg", "PropertyValue": "242.21", "PropertyUnit": "K", "MeasurementMethod": "DSC", "MeasurementComposition": "(0.02)" },
#   { "PolymerName": "PEO-LiClO4-40 wt. % PC", "PropertyName": "Tg", "PropertyValue": "233.81", "PropertyUnit": "K", "MeasurementMethod": "DSC", "MeasurementComposition": "(0.02)" },
#   { "PolymerName": "PEO-LiClO4-50 wt. % PC", "PropertyName": "Tg", "PropertyValue": "245.84", "PropertyUnit": "K", "MeasurementMethod": "DSC", "MeasurementComposition": "(0.02)" }
# ]
# ```

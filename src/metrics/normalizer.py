"""
normalizer.py — Rule-based ESG/BRSR metric name normalization and unit
compatibility checking.
"""

from __future__ import annotations

import re
from config.settings import cfg

# canonical_name -> list of regex patterns (case-insensitive) that match it
_METRIC_SYNONYMS: dict[str, list[str]] = {
    "Scope 1 Emissions": [
        r"scope\s*1.*emission", r"ghg.*emission.*scope\s*1", r"direct.*ghg.*emission",
    ],
    "Scope 2 Emissions": [
        r"scope\s*2.*emission", r"ghg.*emission.*scope\s*2", r"indirect.*ghg.*emission",
    ],
    "Scope 3 Emissions": [
        r"scope\s*3.*emission",
    ],
    "Total GHG Emissions": [
        r"total.*ghg.*emission", r"total.*greenhouse.*gas",
    ],
    "Emissions Intensity": [
        r"emission.*intensity", r"ghg.*intensity", r"carbon.*intensity",
    ],
    "Renewable Energy Consumption": [
        r"renewable.*energy.*consum", r"renewable.*power.*consum", r"green.*energy.*consum",
    ],
    "Total Energy Consumption": [
        r"total.*energy.*consum",
    ],
    "Water Consumption": [
        r"water.*consum", r"water.*withdraw",
    ],
    "Water Recycled/Reused": [
        r"water.*recycl", r"water.*reus",
    ],
    "Waste Generated": [
        r"waste.*generat", r"total.*waste",
    ],
    "Waste Recycled": [
        r"waste.*recycl", r"waste.*recover",
    ],
    "Hazardous Waste": [
        r"hazardous.*waste",
    ],
    "Employee Training Coverage": [
        r"employee.*train", r"training.*coverage", r"train.*hours.*employee",
    ],
    "Employee Attrition Rate": [
        r"attrition", r"turnover.*rate",
    ],
    "Gender Diversity (Workforce)": [
        r"women.*employee", r"gender.*divers", r"female.*workforce",
    ],
    "CSR Spend": [
        r"csr.*spend", r"corporate.*social.*responsibility.*expenditure",
    ],
    "Safety Incidents (LTIFR)": [
        r"ltifr", r"lost.*time.*injur", r"safety.*incident",
    ],
    "Board Independence": [
        r"independent.*director", r"board.*independence",
    ],
}

_COMPILED = {
    canon: [re.compile(p, re.IGNORECASE) for p in patterns]
    for canon, patterns in _METRIC_SYNONYMS.items()
}


def normalize_metric_name(raw_label: str) -> str:
    label = raw_label.strip()
    for canon, patterns in _COMPILED.items():
        if any(p.search(label) for p in patterns):
            return canon
    return label


def normalize_unit(raw_unit: str) -> str:
    return re.sub(r"\s+", " ", raw_unit.strip().lower())


def units_compatible(unit_a: str, unit_b: str) -> bool:
    a, b = normalize_unit(unit_a), normalize_unit(unit_b)
    if not a or not b:
        return a == b
    if a == b:
        return True
    for group in cfg.dashboard.compatible_unit_groups:
        norm_group = {normalize_unit(u) for u in group}
        if a in norm_group and b in norm_group:
            return True
    return False

import logging
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, List, Optional, TypedDict

logger = logging.getLogger("bolt_calculator")

# =============================================================================
# Reference data
# -----------------------------------------------------------------------------
# Methodology follows Shigley's Mechanical Engineering Design (Ch. 8) and the
# Rotscher/VDI 2230 frustum-cone model for member stiffness. All internal
# calculations are strictly Metric (mm, N, MPa, deg C).
# =============================================================================

# Bolt Sizes: name -> (diameter_mm, pitch_mm)
BOLT_SIZES_METRIC: Dict[str, Tuple[float, float]] = {
    "M4": (4.0, 0.7),
    "M5": (5.0, 0.8),
    "M6": (6.0, 1.0),
    "M8": (8.0, 1.25),
    "M10": (10.0, 1.5),
    "M12": (12.0, 1.75),
    "M14": (14.0, 2.0),
    "M16": (16.0, 2.0),
    "M20": (20.0, 2.5),
    "M24": (24.0, 3.0),
    "M30": (30.0, 3.5),
    "M36": (36.0, 4.0),
}

# Imperial sizes are keyed by NOMINAL DIAMETER (e.g. "1/2") and store their
# millimetre equivalents (reference data is kept metric throughout). The pitch in
# the tuple is the default coarse (UNC) pitch; the available fine/UNF pitches live
# in BOLT_THREAD_SERIES_IMPERIAL and are picked separately in the UI. e.g. 1/2 ->
# 0.5 in = 12.7 mm, UNC 13 TPI -> 25.4/13 = 1.9538 mm pitch.
BOLT_SIZES_IMPERIAL: Dict[str, Tuple[float, float]] = {
    "1/4": (6.35, 1.27),
    "5/16": (7.9375, 1.4111),
    "3/8": (9.525, 1.5875),
    "1/2": (12.7, 1.9538),
    "5/8": (15.875, 2.3091),
    "3/4": (19.05, 2.54),
    "7/8": (22.225, 2.8222),
    "1": (25.4, 3.175),
}

# Thread series available per bolt size: a list of (label, pitch_mm), coarse/UNC
# first (the default). All are the SAME 60-degree profile, so the stress-area and
# stiffness models apply unchanged -- only the pitch differs. A finer pitch raises
# the tensile-stress area (and thus the proof load) at the same nominal diameter
# and gives finer preload-per-turn control, at the cost of easier stripping in soft
# materials. The first pitch in each list must match the size table's coarse pitch.
BOLT_THREAD_SERIES_METRIC: Dict[str, List[Tuple[str, float]]] = {
    "M4":  [("Coarse — 0.70 mm", 0.70), ("Fine — 0.50 mm", 0.50)],
    "M5":  [("Coarse — 0.80 mm", 0.80), ("Fine — 0.50 mm", 0.50)],
    "M6":  [("Coarse — 1.00 mm", 1.00), ("Fine — 0.75 mm", 0.75)],
    "M8":  [("Coarse — 1.25 mm", 1.25), ("Fine — 1.00 mm", 1.00), ("Fine — 0.75 mm", 0.75)],
    "M10": [("Coarse — 1.50 mm", 1.50), ("Fine — 1.25 mm", 1.25), ("Fine — 1.00 mm", 1.00)],
    "M12": [("Coarse — 1.75 mm", 1.75), ("Fine — 1.50 mm", 1.50), ("Fine — 1.25 mm", 1.25)],
    "M14": [("Coarse — 2.00 mm", 2.00), ("Fine — 1.50 mm", 1.50)],
    "M16": [("Coarse — 2.00 mm", 2.00), ("Fine — 1.50 mm", 1.50)],
    "M20": [("Coarse — 2.50 mm", 2.50), ("Fine — 2.00 mm", 2.00), ("Fine — 1.50 mm", 1.50)],
    "M24": [("Coarse — 3.00 mm", 3.00), ("Fine — 2.00 mm", 2.00)],
    "M30": [("Coarse — 3.50 mm", 3.50), ("Fine — 2.00 mm", 2.00)],
    "M36": [("Coarse — 4.00 mm", 4.00), ("Fine — 3.00 mm", 3.00)],
}

# Unified inch series: pitch_mm = 25.4 / TPI. UNC (coarse) first, then UNF (fine)
# and, where standard, UNEF (extra-fine).
BOLT_THREAD_SERIES_IMPERIAL: Dict[str, List[Tuple[str, float]]] = {
    "1/4":  [("UNC — 20 TPI", 1.2700), ("UNF — 28 TPI", 0.9071)],
    "5/16": [("UNC — 18 TPI", 1.4111), ("UNF — 24 TPI", 1.0583)],
    "3/8":  [("UNC — 16 TPI", 1.5875), ("UNF — 24 TPI", 1.0583)],
    "1/2":  [("UNC — 13 TPI", 1.9538), ("UNF — 20 TPI", 1.2700)],
    "5/8":  [("UNC — 11 TPI", 2.3091), ("UNF — 18 TPI", 1.4111)],
    "3/4":  [("UNC — 10 TPI", 2.5400), ("UNF — 16 TPI", 1.5875)],
    "7/8":  [("UNC — 9 TPI", 2.8222), ("UNF — 14 TPI", 1.8143)],
    "1":    [("UNC — 8 TPI", 3.1750), ("UNF — 12 TPI", 2.1167), ("UNEF — 20 TPI", 1.2700)],
}

# Bolt Materials (Metric units: Sp/Sy/Sut/Se in MPa, E in MPa, CTE in 1/C).
#   Sp, Sy, Sut : proof, yield (0.2% offset) and ultimate tensile strength
#              (ISO 898-1 for metric classes, SAE J429 for imperial grades).
#   Se       : fully-corrected axial endurance strength for ROLLED threads,
#              already including the thread fatigue stress concentration
#              (Shigley Table 8-17). Do NOT re-apply a Kf in fatigue.
#   E        : Young's modulus of the bolt material. Sy feeds the Soderberg
#              fatigue criterion (Sp feeds the ASME-elliptic criterion).
# Strengths per ISO 898-1 (metric) / SAE J429 (imperial); Se (rolled-thread,
# fully corrected) per Shigley Table 8-17. Stainless per ISO 3506.
BOLT_MATERIALS_METRIC: Dict[str, BoltMaterial] = {
    "Grade 4.6": {"Sp": 225, "Sy": 240, "Sut": 400, "Se": 60, "E": 200000, "CTE": 11.5e-6,
                  "source": "ISO 898-1; Se Shigley T8-17"},
    "Grade 5.8": {"Sp": 380, "Sy": 420, "Sut": 500, "Se": 85, "E": 200000, "CTE": 11.5e-6,
                  "source": "ISO 898-1; Se est."},
    "Grade 8.8": {"Sp": 600, "Sy": 640, "Sut": 800, "Se": 129, "E": 200000, "CTE": 11.5e-6,
                  "source": "ISO 898-1; Se Shigley T8-17"},
    "Grade 9.8": {"Sp": 650, "Sy": 720, "Sut": 900, "Se": 140, "E": 200000, "CTE": 11.5e-6,
                  "source": "ISO 898-1; Se Shigley T8-17"},
    "Grade 10.9": {"Sp": 830, "Sy": 940, "Sut": 1040, "Se": 162, "E": 200000, "CTE": 11.5e-6,
                   "source": "ISO 898-1; Se Shigley T8-17"},
    "Grade 12.9": {"Sp": 970, "Sy": 1100, "Sut": 1220, "Se": 190, "E": 200000, "CTE": 11.5e-6,
                   "source": "ISO 898-1; Se Shigley T8-17"},
    "Stainless A2-70": {"Sp": 450, "Sy": 450, "Sut": 700, "Se": 105, "E": 193000, "CTE": 16.0e-6,
                        "source": "ISO 3506; Se est."},
    "Stainless A4-80": {"Sp": 600, "Sy": 600, "Sut": 800, "Se": 120, "E": 193000, "CTE": 16.0e-6,
                        "source": "ISO 3506; Se est."},
}

BOLT_MATERIALS_IMPERIAL: Dict[str, BoltMaterial] = {
    "SAE Grade 2": {"Sp": 379, "Sy": 393, "Sut": 510, "Se": 85, "E": 200000, "CTE": 11.5e-6,
                    "source": "SAE J429; Se est."},
    "SAE Grade 5": {"Sp": 586, "Sy": 634, "Sut": 827, "Se": 129, "E": 200000, "CTE": 11.5e-6,
                    "source": "SAE J429; Se Shigley T8-17"},
    "SAE Grade 7": {"Sp": 724, "Sy": 792, "Sut": 917, "Se": 142, "E": 200000, "CTE": 11.5e-6,
                    "source": "SAE J429; Se Shigley T8-17"},
    "SAE Grade 8": {"Sp": 827, "Sy": 896, "Sut": 1034, "Se": 162, "E": 200000, "CTE": 11.5e-6,
                    "source": "SAE J429; Se Shigley T8-17"},
    "Stainless 18-8": {"Sp": 448, "Sy": 448, "Sut": 689, "Se": 103, "E": 193000, "CTE": 16.0e-6,
                       "source": "ASTM F593; Se est."},
    "ASTM A325 (Structural)": {"Sp": 586, "Sy": 634, "Sut": 827, "Se": 129, "E": 200000, "CTE": 11.5e-6,
                               "source": "ASTM F3125; Se Shigley"},
    "ASTM A490 (Structural)": {"Sp": 827, "Sy": 896, "Sut": 1034, "Se": 162, "E": 200000, "CTE": 11.5e-6,
                               "source": "ASTM F3125; Se Shigley"},
    "ASTM A193 B7 (Alloy)": {"Sp": 724, "Sy": 724, "Sut": 862, "Se": 140, "E": 200000, "CTE": 11.5e-6,
                             "source": "ASTM A193; Se est."},
}

# Joint (clamped-member) materials: Syc (compressive yield, MPa), E (MPa),
# CTE (1/deg C) and a source tag for the data.
#
# IMPORTANT: these are NOMINAL, room-temperature, condition/temper-specific values
# (the condition is in the name). They are representative figures compiled from the
# references below and MUST be verified against certified data for design use.
# Syc = compressive yield, taken as the tensile yield for ductile metals and as
# the compressive strength for brittle materials (cast irons, concrete, etc.).
#
# Source tags:
#   ASM         = ASM Metals Reference Book / ASM Handbook Vols. 1-2
#   MatWeb      = MatWeb database (typical values / manufacturer datasheets)
#   MMPDS       = MMPDS-2023 (formerly MIL-HDBK-5), metallic aerospace materials
#   ASTM ...    = the cited ASTM material standard
#   CDA         = Copper Development Association
#   SpecialM    = Special Metals / Haynes alloy datasheets
#   Mfr         = manufacturer datasheet (Victrex, DuPont, etc.)
#   ETB         = The Engineering ToolBox / general engineering handbooks
JOINT_MATERIALS: Dict[str, JointMaterial] = {
    # --- Carbon, alloy & structural steels ---
    "Steel (Mild)": {"Syc": 250, "E": 200000, "CTE": 11.5e-6, "source": "ASM"},
    "Steel AISI 1018 (CD)": {"Syc": 370, "E": 205000, "CTE": 11.7e-6, "source": "ASM"},
    "Steel AISI 1020 (CD)": {"Syc": 350, "E": 205000, "CTE": 11.7e-6, "source": "ASM"},
    "Steel AISI 1045 (normalized)": {"Syc": 450, "E": 205000, "CTE": 11.5e-6, "source": "ASM"},
    "Steel AISI 1045 (Q&T)": {"Syc": 530, "E": 205000, "CTE": 11.5e-6, "source": "ASM"},
    "Steel AISI 1095 (spring)": {"Syc": 525, "E": 205000, "CTE": 11.4e-6, "source": "ASM"},
    "Steel AISI 4130 (normalized)": {"Syc": 460, "E": 205000, "CTE": 12.2e-6, "source": "ASM"},
    "Steel 4140": {"Syc": 415, "E": 205000, "CTE": 11.5e-6, "source": "ASM"},
    "Steel AISI 4140 (Q&T)": {"Syc": 655, "E": 205000, "CTE": 12.3e-6, "source": "ASM"},
    "Steel AISI 4340 (Q&T)": {"Syc": 860, "E": 205000, "CTE": 12.3e-6, "source": "ASM"},
    "Steel AISI 8620 (core)": {"Syc": 360, "E": 205000, "CTE": 11.1e-6, "source": "ASM"},
    "Steel AISI 52100 (annealed)": {"Syc": 415, "E": 210000, "CTE": 11.9e-6, "source": "ASM"},
    "Steel ASTM A36": {"Syc": 250, "E": 200000, "CTE": 11.7e-6, "source": "ASTM A36"},
    "Steel ASTM A572 Gr.50": {"Syc": 345, "E": 200000, "CTE": 11.7e-6, "source": "ASTM A572"},
    "Steel ASTM A514 (T-1)": {"Syc": 690, "E": 200000, "CTE": 11.7e-6, "source": "ASTM A514"},
    "Steel HY-80": {"Syc": 550, "E": 207000, "CTE": 11.7e-6, "source": "MIL-S-16216"},
    "Steel Maraging 250": {"Syc": 1700, "E": 186000, "CTE": 11.3e-6, "source": "MMPDS"},
    "Steel Maraging 300": {"Syc": 2000, "E": 190000, "CTE": 10.1e-6, "source": "MMPDS"},
    # --- Stainless steels ---
    "Stainless Steel 304": {"Syc": 215, "E": 193000, "CTE": 17.2e-6, "source": "ASTM A240"},
    "Stainless 303": {"Syc": 240, "E": 193000, "CTE": 17.2e-6, "source": "ASM"},
    "Stainless 316": {"Syc": 240, "E": 193000, "CTE": 16.0e-6, "source": "ASTM A240"},
    "Stainless 316L": {"Syc": 205, "E": 193000, "CTE": 16.0e-6, "source": "ASTM A240"},
    "Stainless 321": {"Syc": 240, "E": 193000, "CTE": 16.6e-6, "source": "ASTM A240"},
    "Stainless 347": {"Syc": 240, "E": 193000, "CTE": 16.6e-6, "source": "ASTM A240"},
    "Stainless 410 (annealed)": {"Syc": 275, "E": 200000, "CTE": 9.9e-6, "source": "ASTM A240"},
    "Stainless 416 (annealed)": {"Syc": 275, "E": 200000, "CTE": 9.9e-6, "source": "ASM"},
    "Stainless 420 (annealed)": {"Syc": 345, "E": 200000, "CTE": 10.3e-6, "source": "ASM"},
    "Stainless 430": {"Syc": 310, "E": 200000, "CTE": 10.4e-6, "source": "ASTM A240"},
    "Stainless 17-4 PH (H900)": {"Syc": 1170, "E": 197000, "CTE": 10.8e-6, "source": "ASTM A564"},
    "Stainless 15-5 PH (H900)": {"Syc": 1170, "E": 196000, "CTE": 10.8e-6, "source": "ASTM A564"},
    "Stainless 2205 (duplex)": {"Syc": 450, "E": 200000, "CTE": 13.7e-6, "source": "ASTM A240"},
    "Stainless 904L": {"Syc": 220, "E": 190000, "CTE": 15.3e-6, "source": "ASM"},
    "Stainless A286 (aged)": {"Syc": 590, "E": 201000, "CTE": 16.5e-6, "source": "MMPDS"},
    # --- Tool steels (hardened; Syc is compressive yield) ---
    "Tool Steel A2 (hardened)": {"Syc": 1900, "E": 203000, "CTE": 10.7e-6, "source": "ASM"},
    "Tool Steel D2 (hardened)": {"Syc": 2200, "E": 210000, "CTE": 10.4e-6, "source": "ASM"},
    "Tool Steel O1 (hardened)": {"Syc": 1800, "E": 205000, "CTE": 11.0e-6, "source": "ASM"},
    "Tool Steel H13 (hardened)": {"Syc": 1500, "E": 210000, "CTE": 10.4e-6, "source": "ASM"},
    "Tool Steel M2 HSS (hardened)": {"Syc": 3250, "E": 224000, "CTE": 9.4e-6, "source": "ASM"},
    # --- Cast irons (Syc = compressive strength) ---
    "Cast Iron": {"Syc": 820, "E": 100000, "CTE": 10.4e-6, "source": "ASM (gray, compressive)"},
    "Gray Cast Iron Class 30": {"Syc": 750, "E": 100000, "CTE": 10.5e-6, "source": "ASTM A48"},
    "Gray Cast Iron Class 40": {"Syc": 965, "E": 125000, "CTE": 10.5e-6, "source": "ASTM A48"},
    "Ductile Iron 65-45-12": {"Syc": 310, "E": 169000, "CTE": 11.5e-6, "source": "ASTM A536"},
    "Ductile Iron 80-55-06": {"Syc": 380, "E": 168000, "CTE": 11.5e-6, "source": "ASTM A536"},
    "Malleable Iron 32510": {"Syc": 220, "E": 170000, "CTE": 12.0e-6, "source": "ASTM A47"},
    "White Cast Iron": {"Syc": 1380, "E": 180000, "CTE": 9.0e-6, "source": "ASM (compressive)"},
    # --- Aluminum alloys ---
    "Aluminum 6061-O (annealed)": {"Syc": 55, "E": 69000, "CTE": 23.6e-6, "source": "Aluminum Assoc."},
    "Aluminum 6061-T4": {"Syc": 145, "E": 69000, "CTE": 23.6e-6, "source": "Aluminum Assoc."},
    "Aluminum (6061-T6)": {"Syc": 275, "E": 69000, "CTE": 23.6e-6, "source": "ASM/Aluminum Assoc."},
    "Aluminum (7075-T6)": {"Syc": 503, "E": 71000, "CTE": 23.6e-6, "source": "MMPDS"},
    "Aluminum 2014-T6": {"Syc": 414, "E": 73000, "CTE": 23.0e-6, "source": "MMPDS"},
    "Aluminum 2024-T3": {"Syc": 345, "E": 73000, "CTE": 23.2e-6, "source": "MMPDS"},
    "Aluminum 2024-T4": {"Syc": 324, "E": 73000, "CTE": 23.2e-6, "source": "MMPDS"},
    "Aluminum 3003-H14": {"Syc": 145, "E": 69000, "CTE": 23.2e-6, "source": "Aluminum Assoc."},
    "Aluminum 5052-H32": {"Syc": 195, "E": 70000, "CTE": 23.8e-6, "source": "Aluminum Assoc."},
    "Aluminum 5083-H116": {"Syc": 215, "E": 71000, "CTE": 23.8e-6, "source": "ASTM B928"},
    "Aluminum 6063-T5": {"Syc": 145, "E": 69000, "CTE": 23.4e-6, "source": "Aluminum Assoc."},
    "Aluminum 6082-T6": {"Syc": 250, "E": 70000, "CTE": 23.4e-6, "source": "EN 755"},
    "Aluminum 7050-T7451": {"Syc": 469, "E": 72000, "CTE": 23.5e-6, "source": "MMPDS"},
    "Aluminum 1100-H14": {"Syc": 117, "E": 69000, "CTE": 23.6e-6, "source": "Aluminum Assoc."},
    "Aluminum 356.0-T6 (cast)": {"Syc": 205, "E": 72400, "CTE": 21.5e-6, "source": "ASTM B26"},
    "Aluminum A380 (die cast)": {"Syc": 160, "E": 71000, "CTE": 21.8e-6, "source": "ASTM B85"},
    # --- Titanium alloys ---
    "Titanium (Grade 5)": {"Syc": 970, "E": 114000, "CTE": 8.6e-6, "source": "MMPDS"},
    "Titanium Grade 2 (CP)": {"Syc": 275, "E": 105000, "CTE": 8.6e-6, "source": "ASTM B265"},
    "Titanium Grade 9 (3Al-2.5V)": {"Syc": 485, "E": 100000, "CTE": 9.4e-6, "source": "ASTM B338"},
    "Titanium Grade 23 (6Al-4V ELI)": {"Syc": 795, "E": 113000, "CTE": 8.6e-6, "source": "ASTM F136"},
    "Titanium Beta-C (aged)": {"Syc": 1100, "E": 86000, "CTE": 9.0e-6, "source": "ASM"},
    # --- Copper alloys ---
    "Brass": {"Syc": 110, "E": 110000, "CTE": 20.0e-6, "source": "ASM/CDA"},
    "Brass C260 (cartridge, H02)": {"Syc": 310, "E": 110000, "CTE": 19.9e-6, "source": "CDA"},
    "Brass C360 (free-cutting)": {"Syc": 310, "E": 97000, "CTE": 20.5e-6, "source": "CDA"},
    "Bronze C510 (phosphor)": {"Syc": 345, "E": 110000, "CTE": 17.8e-6, "source": "CDA"},
    "Bronze C932 (bearing)": {"Syc": 125, "E": 100000, "CTE": 18.0e-6, "source": "CDA"},
    "Aluminum Bronze C954": {"Syc": 250, "E": 110000, "CTE": 16.2e-6, "source": "CDA"},
    "Copper C110 (ETP, annealed)": {"Syc": 70, "E": 117000, "CTE": 17.0e-6, "source": "CDA"},
    "Beryllium Copper C17200 (aged)": {"Syc": 1000, "E": 128000, "CTE": 16.7e-6, "source": "CDA"},
    "Cupronickel C715 (70/30)": {"Syc": 170, "E": 150000, "CTE": 16.2e-6, "source": "CDA"},
    # --- Nickel & cobalt alloys / superalloys ---
    "Nickel 200": {"Syc": 150, "E": 207000, "CTE": 13.3e-6, "source": "SpecialM"},
    "Monel 400": {"Syc": 240, "E": 179000, "CTE": 13.9e-6, "source": "SpecialM"},
    "Monel K-500 (aged)": {"Syc": 790, "E": 179000, "CTE": 13.7e-6, "source": "SpecialM"},
    "Inconel 600": {"Syc": 310, "E": 207000, "CTE": 13.3e-6, "source": "SpecialM"},
    "Inconel 625": {"Syc": 490, "E": 208000, "CTE": 12.8e-6, "source": "SpecialM"},
    "Inconel 718 (aged)": {"Syc": 1035, "E": 200000, "CTE": 13.0e-6, "source": "SpecialM"},
    "Inconel X-750 (aged)": {"Syc": 815, "E": 214000, "CTE": 12.6e-6, "source": "SpecialM"},
    "Hastelloy C-276": {"Syc": 355, "E": 205000, "CTE": 11.2e-6, "source": "Haynes"},
    "Waspaloy (aged)": {"Syc": 795, "E": 213000, "CTE": 12.2e-6, "source": "SpecialM"},
    "Invar 36": {"Syc": 280, "E": 141000, "CTE": 1.3e-6, "source": "ASM (low-CTE)"},
    "Kovar": {"Syc": 340, "E": 138000, "CTE": 5.5e-6, "source": "Mfr (CRT-alloy)"},
    # --- Other metals ---
    "Magnesium AZ31B": {"Syc": 220, "E": 45000, "CTE": 26.0e-6, "source": "ASTM B90"},
    "Magnesium AZ91D (cast)": {"Syc": 160, "E": 45000, "CTE": 26.0e-6, "source": "ASTM B94"},
    "Zinc Zamak 3 (die cast)": {"Syc": 220, "E": 85000, "CTE": 27.4e-6, "source": "ASTM B86"},
    "Tungsten (wrought)": {"Syc": 750, "E": 400000, "CTE": 4.5e-6, "source": "ASM"},
    "Molybdenum (wrought)": {"Syc": 415, "E": 329000, "CTE": 4.8e-6, "source": "ASM"},
    "Beryllium (S-200F)": {"Syc": 240, "E": 287000, "CTE": 11.3e-6, "source": "ASM"},
    "Lead (pure)": {"Syc": 12, "E": 14000, "CTE": 29.0e-6, "source": "ETB"},
    "Tin Babbitt (bearing)": {"Syc": 30, "E": 50000, "CTE": 23.0e-6, "source": "ASM"},
    # --- Engineering polymers ---
    "Plastic (Nylon)": {"Syc": 60, "E": 3000, "CTE": 80.0e-6, "source": "MatWeb"},
    "Nylon 6/6 (30% glass)": {"Syc": 145, "E": 9000, "CTE": 30.0e-6, "source": "Mfr"},
    "Acetal / POM (Delrin)": {"Syc": 70, "E": 3100, "CTE": 110.0e-6, "source": "Mfr (DuPont)"},
    "Polycarbonate (PC)": {"Syc": 62, "E": 2400, "CTE": 68.0e-6, "source": "MatWeb"},
    "PEEK (unfilled)": {"Syc": 100, "E": 3600, "CTE": 47.0e-6, "source": "Mfr (Victrex)"},
    "PEEK (30% carbon)": {"Syc": 200, "E": 13000, "CTE": 16.0e-6, "source": "Mfr (Victrex)"},
    "PTFE (Teflon)": {"Syc": 23, "E": 500, "CTE": 135.0e-6, "source": "MatWeb"},
    "UHMW-PE": {"Syc": 21, "E": 700, "CTE": 200.0e-6, "source": "MatWeb"},
    "ABS": {"Syc": 45, "E": 2300, "CTE": 90.0e-6, "source": "MatWeb"},
    "PVC (rigid)": {"Syc": 52, "E": 3000, "CTE": 70.0e-6, "source": "MatWeb"},
    "PMMA (acrylic)": {"Syc": 70, "E": 3000, "CTE": 70.0e-6, "source": "MatWeb"},
    "Epoxy (cast)": {"Syc": 70, "E": 3400, "CTE": 60.0e-6, "source": "MatWeb"},
    # --- Composites & non-metals (highly anisotropic: representative values) ---
    "G-10 / FR-4 (glass-epoxy)": {"Syc": 350, "E": 18000, "CTE": 14.0e-6, "source": "Mfr (laminate)"},
    "CFRP (quasi-isotropic)": {"Syc": 450, "E": 55000, "CTE": 2.5e-6, "source": "MatWeb (anisotropic)"},
    "Concrete (C30/37)": {"Syc": 30, "E": 30000, "CTE": 10.0e-6, "source": "ETB / EN 1992"},
    "Wood (Douglas Fir, || grain)": {"Syc": 50, "E": 13000, "CTE": 4.0e-6, "source": "ETB (anisotropic)"},
}

# Bearing-diameter factor (d_w / d) under the head/nut for crushing & frustum base.
BOLT_TYPES: Dict[str, float] = {
    "Hex Head": 1.5,
    "Socket Head Cap Screw": 1.5,
    "Flange Head": 2.0
}

# Nut factor K (torque coefficient) in T = K * F * d. These bundle thread + head
# friction and geometry; they are NOT the bare thread friction coefficient.
FRICTION_COEFFICIENTS: Dict[str, float] = {
    "Dry / as-received (K=0.20)": 0.20,
    "Zinc-Plated (K=0.16)": 0.16,
    "Lubricated / Oil (K=0.15)": 0.15,
    "Teflon / Moly (K=0.12)": 0.12
}

# Tightening method -> relative preload scatter (+/- fraction about the target).
# Representative VDI 2230 / Bickford ranges for the scatter of achieved preload.
TIGHTENING_METHODS: Dict[str, float] = {
    "Torque wrench (+/-25%)": 0.25,
    "Turn-of-nut / angle (+/-15%)": 0.15,
    "Hydraulic tensioner (+/-10%)": 0.10,
    "Hand / uncontrolled (+/-35%)": 0.35,
}

# Fatigue criteria selectable in the UI. The first six are mean-stress loci
# evaluated along the preloaded-bolt load line; the VDI 2230 entries instead use
# a diameter-based endurance amplitude (see vdi2230_endurance_fos).
FATIGUE_CRITERIA: Tuple[str, ...] = (
    "Goodman", "Gerber", "ASME-elliptic", "Soderberg", "SWT", "Morrow",
    "VDI 2230, rolled before HT", "VDI 2230, rolled after HT",
)

BOLT_MODULUS_MPA = 200000.0  # Fallback bolt modulus (steel) if material has no E.

# Morrow true-fracture-strength estimate for steels: sigma_f ~= Sut + this offset
MORROW_SIGMA_F_OFFSET_MPA = 345.0


def morrow_sigma_f(Sut: float) -> float:
    """Morrow true fracture strength estimate for steels: sigma_f ~= Sut + 345 MPa."""
    return Sut + MORROW_SIGMA_F_OFFSET_MPA


# Preferred standard bolt lengths (mm). Reference data is kept metric throughout
# (like the bolt sizes and strengths); the imperial set is the inch series stored
# as its millimetre equivalents.
STANDARD_BOLT_LENGTHS_METRIC_MM: Tuple[float, ...] = (
    6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
    90, 100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300,
)
STANDARD_BOLT_LENGTHS_IMPERIAL_MM: Tuple[float, ...] = tuple(
    round(x * 25.4, 4) for x in (
        0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0,
        2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0,
    )
)

# Typical fastener reference dimensions by bolt-size name, all in mm:
#   hex_af    : hex head / nut width across flats (ISO 272 for metric).
#   socket_af : hex-key (Allen) size for a socket-head cap screw (ISO 4762).
#   clearance : typical free-fit clearance hole (ISO 273 "medium" for metric;
#               common clearance-drill sizes for imperial).
# The tap-drill diameter is computed as nominal - pitch (the coarse-thread rule),
# so it is not stored here. Imperial wrench/key sizes are inch tools given in mm.
BOLT_HARDWARE: Dict[str, Dict[str, float]] = {
    "M4": {"hex_af": 7.0, "socket_af": 3.0, "clearance": 4.5},
    "M5": {"hex_af": 8.0, "socket_af": 4.0, "clearance": 5.5},
    "M6": {"hex_af": 10.0, "socket_af": 5.0, "clearance": 6.6},
    "M8": {"hex_af": 13.0, "socket_af": 6.0, "clearance": 9.0},
    "M10": {"hex_af": 16.0, "socket_af": 8.0, "clearance": 11.0},
    "M12": {"hex_af": 18.0, "socket_af": 10.0, "clearance": 13.5},
    "M14": {"hex_af": 21.0, "socket_af": 12.0, "clearance": 15.5},
    "M16": {"hex_af": 24.0, "socket_af": 14.0, "clearance": 17.5},
    "M20": {"hex_af": 30.0, "socket_af": 17.0, "clearance": 22.0},
    "M24": {"hex_af": 36.0, "socket_af": 19.0, "clearance": 26.0},
    "M30": {"hex_af": 46.0, "socket_af": 22.0, "clearance": 33.0},
    "M36": {"hex_af": 55.0, "socket_af": 27.0, "clearance": 39.0},
    "1/4": {"hex_af": 11.11, "socket_af": 4.76, "clearance": 6.93},
    "5/16": {"hex_af": 12.70, "socket_af": 6.35, "clearance": 8.74},
    "3/8": {"hex_af": 14.29, "socket_af": 7.94, "clearance": 10.32},
    "1/2": {"hex_af": 19.05, "socket_af": 9.53, "clearance": 13.49},
    "5/8": {"hex_af": 23.81, "socket_af": 12.70, "clearance": 16.67},
    "3/4": {"hex_af": 28.58, "socket_af": 15.88, "clearance": 19.84},
    "7/8": {"hex_af": 33.34, "socket_af": 15.88, "clearance": 23.02},
    "1": {"hex_af": 38.10, "socket_af": 19.05, "clearance": 26.19},
}


# =============================================================================
# Result types
# -----------------------------------------------------------------------------
# TypedDicts for the dict results of the main calculations. They document every
# field and let the type checker catch key typos at the call sites (the app indexes
# these dicts with string literals). At runtime a TypedDict is an ordinary dict, so
# this changes nothing about behaviour or serialisation.
# =============================================================================

class PreloadResult(TypedDict):
    """Return value of :func:`calculate_preload` (all SI: N, mm, MPa, deg C)."""
    tensile_stress_area_mm2: float
    proof_load_N: float
    target_preload_N: float
    recommended_preload_N: float
    torque_Nm: float
    bearing_area_mm2: float
    bearing_stress_MPa: float
    crushing_warning_material: str
    kb_N_mm: float
    km_N_mm: float
    joint_constant_C: float
    total_grip_length_mm: float
    thermal_delta_F_N: float
    embedment_loss_N: float
    operating_preload_N: float
    max_bolt_force_N: float
    fatigue_sigma_a_MPa: float
    fatigue_sigma_m_MPa: float
    preload_stress_MPa: float
    fatigue_fos: float
    fatigue_criterion: str
    fatigue_all_fos: Dict[str, float]
    proof_fos: float
    separation_load_N: float
    separation_fos: float
    thread_shear_fos: float
    required_engagement_mm: float
    tightening_stress_MPa: float
    tightening_utilization: float
    tightening_fos: float
    endurance_Se_MPa: float
    ultimate_Sut_MPa: float
    proof_strength_MPa: float
    yield_Sy_MPa: float
    vdi_sigma_asv_MPa: float
    vdi_sigma_asg_MPa: float


class BoltGroupResult(TypedDict):
    """Return value of :func:`analyze_bolt_group` (coords/eccentricity in mm, forces in N)."""
    centroid: Tuple[float, float]
    tensions_N: List[float]
    shears_N: List[float]
    shear_vectors_N: List[Tuple[float, float]]
    governing_index: int
    governing_tension_N: float
    governing_shear_N: float
    sum_distance_sq_mm2: float
    polar_moment_mm2: float
    moment_reactable: bool


class _Sourced(TypedDict, total=False):
    """Optional ``source`` tag shared by the material tables (custom materials omit it)."""
    source: str


class JointMaterial(_Sourced):
    """A clamped-member material: Syc compressive yield, E modulus (MPa), CTE (1/deg C)."""
    Syc: float
    E: float
    CTE: float


class BoltMaterial(_Sourced):
    """A bolt grade: Sp/Sy/Sut/Se strengths and E (MPa), CTE (1/deg C)."""
    Sp: float
    Sy: float
    Sut: float
    Se: float
    E: float
    CTE: float


class Layer(TypedDict):
    """One clamped layer fed to calculate_preload (thickness mm; Syc/E MPa; CTE 1/deg C)."""
    Material: str
    thickness: float
    Syc: float
    E: float
    CTE: float


class BoltCandidate(TypedDict):
    """One passing size/material/pitch from recommend_bolt."""
    size: str
    material: str
    pitch_mm: float
    thread: str
    stress_area_mm2: float
    proof_fos: float
    fatigue_fos: float
    separation_fos: float
    preload_N: float
    torque_Nm: float


class RecommendResult(TypedDict):
    """Return value of recommend_bolt: the ranked passing candidates, lightest first."""
    found: bool
    best: Optional[BoltCandidate]
    candidates: List[BoltCandidate]


def calculate_stress_area(d: float, p: float) -> float:
    """Tensile stress area A_t = (pi/4)(d - 0.9382 p)^2 (ISO metric)."""
    return (math.pi / 4.0) * (d - 0.9382 * p)**2


def calculate_bearing_diameter(d: float, bolt_type: str, use_washer: bool) -> float:
    """Effective bearing (washer-face) diameter under the head/nut."""
    return d * 2.0 if use_washer else d * BOLT_TYPES.get(bolt_type, 1.5)


def calculate_bearing_area(d: float, dw: float) -> float:
    """Annular bearing area between the bearing diameter and the clearance hole."""
    clearance_hole_d = d * 1.1
    return (math.pi / 4.0) * (dw**2 - clearance_hole_d**2)


def calculate_bolt_stiffness(d: float, At: float, L: float, E: float) -> float:
    """Axial bolt stiffness using the two-section (shank + thread) model.

    k_b = (A_d A_t E) / (A_d l_t + A_t l_d)   (Shigley Eq. 8-17)

    The threaded length within the grip is calculated by assuming the bolt is sized
    to protrude slightly past the nut (L_bolt_est = L_grip + H_nut).
    """
    if L <= 0:
        return 0.0
    Ad = (math.pi / 4.0) * d * d
    
    # Estimate actual bolt length (grip + nut height (approx d) + 2mm protrusion)
    bolt_len_est = L + d + 2.0
    
    # Standard total threaded length (metric ISO 888 approx for most sizes)
    if bolt_len_est <= 125.0:
        Lt_total = 2.0 * d + 6.0
    elif bolt_len_est <= 200.0:
        Lt_total = 2.0 * d + 12.0
    else:
        Lt_total = 2.0 * d + 25.0
        
    Ld_total = max(0.0, bolt_len_est - Lt_total)  # Total unthreaded shank length
    
    # The shank is entirely within the grip (since it starts at the bolt head)
    Ld = min(L, Ld_total)
    Lt = max(0.0, L - Ld)  # The rest of the grip is threaded
    
    if Ld <= 0:
        return At * E / L
    if Lt <= 0:
        return Ad * E / L
    return (Ad * At * E) / (Ad * Lt + At * Ld)


def calculate_layer_stiffness(E: float, t: float, D: float, d: float) -> float:
    """Rotscher frustum-cone stiffness of one member (30 deg half-apex cone).

    D is the cone base diameter at the loaded face of this frustum; the hole is
    taken at the clearance diameter (1.1 d). See Shigley Eq. 8-20.
    """
    clearance_hole_d = d * 1.1
    numerator = 0.5774 * math.pi * E * clearance_hole_d

    term1 = (1.155 * t + D - clearance_hole_d) * (D + clearance_hole_d)
    term2 = (1.155 * t + D + clearance_hole_d) * (D - clearance_hole_d)

    if term2 <= 0:
        return float('inf')

    ratio = term1 / term2
    if ratio <= 1.0:
        return float('inf')

    return numerator / math.log(ratio)


def _positive_quadratic_root(a: float, b: float, c: float) -> float:
    """Positive root of a t^2 + b t + c = 0 (the load-line multiplier).

    Used to find where the preloaded operating point meets a curved (Gerber /
    elliptic) fatigue locus. Returns +inf when there is no positive root.
    """
    if abs(a) < 1e-30:
        if abs(b) < 1e-30:
            return float('inf')
        t = -c / b
        return t if t > 0.0 else float('inf')
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return float('inf')
    root = (-b + math.sqrt(disc)) / (2.0 * a)
    return root if root > 0.0 else float('inf')


def fatigue_factor_of_safety(criterion: str, sigma_a: float, sigma_m: float,
                             sigma_i: float, Se: float, Sut: float,
                             Sp: float, Sy: float) -> float:
    """Preloaded-bolt fatigue factor of safety for a mean-stress criterion.

    The operating point is (sigma_m, sigma_a); the load line rises from the
    preload point (sigma_i, 0) because only the external load cycles. n_f is the
    factor by which the cyclic stress may grow before the load line reaches the
    failure locus, i.e. n_f = S_a / sigma_a at the intersection (Shigley
    Ch. 8-12). All stresses in MPa.

      Goodman       : sigma_a/Se + sigma_m/Sut     = 1   (linear, conservative)
      Gerber        : sigma_a/Se + (sigma_m/Sut)^2 = 1   (parabola, ductile fit)
      ASME-elliptic : (sigma_a/Se)^2 + (sigma_m/Sp)^2 = 1  (Sp = proof, for bolts)
      Soderberg     : sigma_a/Se + sigma_m/Sy      = 1   (yield, most conservative)
      SWT           : sqrt(sigma_max * sigma_a) = Se      (peak-stress corrected)
      Morrow        : sigma_a/Se + sigma_m/sigma_f = 1    (sigma_f ~ Sut + 345 MPa)

    VDI 2230's diameter-based endurance limit is handled by
    vdi2230_endurance_fos(), not here, because it is not a mean-stress locus.
    """
    if sigma_a <= 0.0:
        return float('inf')              # no alternating stress -> no fatigue
    delta = sigma_m - sigma_i            # mean-stress increment above preload

    if criterion == "Soderberg":
        if sigma_i >= Sy:
            return 0.0
        return (1.0 - sigma_i / Sy) / (sigma_a / Se + delta / Sy)
    if criterion == "Gerber":
        if sigma_i >= Sut:
            return 0.0
        a = (delta * delta) / (Sut * Sut)
        b = sigma_a / Se + 2.0 * sigma_i * delta / (Sut * Sut)
        c = (sigma_i * sigma_i) / (Sut * Sut) - 1.0
        return _positive_quadratic_root(a, b, c)
    if criterion == "ASME-elliptic":
        if sigma_i >= Sp:
            return 0.0
        a = (sigma_a * sigma_a) / (Se * Se) + (delta * delta) / (Sp * Sp)
        b = 2.0 * sigma_i * delta / (Sp * Sp)
        c = (sigma_i * sigma_i) / (Sp * Sp) - 1.0
        return _positive_quadratic_root(a, b, c)
    if criterion == "SWT":
        # Smith-Watson-Topper: sigma_ar = sqrt(sigma_max * sigma_a) <= Se. Along
        # the load line (sigma_max = sigma_i + t(delta + sigma_a)) this gives
        #   sigma_a(delta + sigma_a) t^2 + sigma_a*sigma_i t - Se^2 = 0
        a = sigma_a * (delta + sigma_a)
        b = sigma_a * sigma_i
        c = -Se * Se
        return _positive_quadratic_root(a, b, c)
    if criterion == "Morrow":
        sigma_f = morrow_sigma_f(Sut)   # true fracture-strength estimate for steels (MPa)
        if sigma_i >= sigma_f:
            return 0.0
        return (1.0 - sigma_i / sigma_f) / (sigma_a / Se + delta / sigma_f)
    # Goodman (default)
    if sigma_i >= Sut:
        return 0.0
    denom = Sut * sigma_a + Se * delta
    return (Se * (Sut - sigma_i)) / denom if denom > 0.0 else float('inf')


def vdi2230_endurance_amplitude(d: float, rolled_after_ht: bool,
                                mean_force: float, yield_force: float) -> float:
    """Permissible alternating stress amplitude (MPa) of a bolt thread, VDI 2230.

    Threads rolled BEFORE heat treatment:  sigma_ASV = 0.85 (150/d + 45).
    Threads rolled AFTER heat treatment keep beneficial root residual stress and
    are stronger by (2 - F_Sm/F_0.2), clamped to [1, 2]. ``d`` is the nominal
    diameter in mm, ``mean_force`` the mean bolt force and ``yield_force`` = Sy*At.
    The VDI endurance limit is treated as mean-stress independent.
    """
    if d <= 0.0:
        return 0.0
    sigma_asv = 0.85 * (150.0 / d + 45.0)
    if not rolled_after_ht:
        return sigma_asv
    ratio = mean_force / yield_force if yield_force > 0.0 else 1.0
    factor = min(2.0, max(1.0, 2.0 - ratio))
    return sigma_asv * factor


def vdi2230_endurance_fos(sigma_a: float, d: float, rolled_after_ht: bool,
                          mean_force: float, yield_force: float) -> float:
    """VDI 2230 fatigue factor of safety: permissible amplitude / actual amplitude."""
    if sigma_a <= 0.0:
        return float('inf')
    sigma_allow = vdi2230_endurance_amplitude(d, rolled_after_ht, mean_force, yield_force)
    return sigma_allow / sigma_a


def all_fatigue_factors(sigma_a: float, sigma_m: float, sigma_i: float,
                        Se: float, Sut: float, Sp: float, Sy: float, d: float,
                        bolt_mean_force: float, yield_force: float) -> Dict[str, float]:
    """Fatigue factor of safety for every criterion in FATIGUE_CRITERIA, for the
    side-by-side comparison table and the Haigh diagram."""
    out: Dict[str, float] = {}
    for crit in FATIGUE_CRITERIA:
        if crit.startswith("VDI"):
            out[crit] = vdi2230_endurance_fos(
                sigma_a, d, rolled_after_ht=("after" in crit),
                mean_force=bolt_mean_force, yield_force=yield_force)
        else:
            out[crit] = fatigue_factor_of_safety(crit, sigma_a, sigma_m, sigma_i, Se, Sut, Sp, Sy)
    return out


def tightening_von_mises_stress(preload: float, At: float, d: float, p: float,
                                nut_factor: float) -> float:
    """von Mises stress in the bolt during tightening (axial tension + thread torsion).

    The torsional part comes from the thread torque T_G = F (0.16 p + 0.58 d2 mu_G),
    with the thread friction mu_G inferred from the lumped nut factor K under the
    usual assumption mu_thread ~ mu_head and D_Km ~ 1.4 d. Returns sqrt(s^2 + 3 t^2)
    in MPa -- the peak reached while torquing (the torsion largely relaxes once the
    wrench is released, leaving mostly the axial stress).
    """
    if At <= 0.0 or d <= 0.0:
        return 0.0
    sigma = preload / At
    d2 = d - 0.6495 * p            # pitch diameter
    d3 = d - 1.2269 * p            # minor diameter (rounded root)
    ds = 0.5 * (d2 + d3)           # stress diameter
    mu_g = max(0.02, (nut_factor - 0.159 * p / d) / 1.22)
    torque_thread = preload * (0.16 * p + 0.58 * d2 * mu_g)   # N*mm twisting the bolt
    Wp = math.pi * ds ** 3 / 16.0
    tau = torque_thread / Wp if Wp > 0.0 else 0.0
    return math.sqrt(sigma * sigma + 3.0 * tau * tau)


def calculate_preload(
    d: float,
    p: float,
    bolt_material_props: BoltMaterial,
    layers: Sequence[Layer],
    bolt_type: str,
    use_washer: bool,
    is_permanent: bool,
    friction_condition: str,
    temp_assembly: float = 20.0,
    temp_operating: float = 20.0,
    external_load_max: float = 0.0,
    external_load_min: float = 0.0,
    thread_engagement_length: float = 0.0,
    internal_thread_material_props: Optional[JointMaterial] = None,
    fatigue_criterion: str = "Goodman",
    embedment_um: float = 0.0,
    load_intro_factor: float = 1.0,
) -> PreloadResult:
    """
    Calculate preload, stiffness, thermal, fatigue, separation and thread
    stripping for a bolted joint. All inputs/outputs are strictly Metric
    (mm, N, MPa, deg C). The external loads passed here are PER BOLT.

    layers: list of dicts with keys "E", "Syc", "CTE", "thickness".
    """
    Sp = bolt_material_props["Sp"]
    Sut = bolt_material_props["Sut"]
    Se = bolt_material_props["Se"]
    Sy = bolt_material_props.get("Sy", Sp / 0.9)
    bolt_cte = bolt_material_props["CTE"]
    bolt_E = bolt_material_props.get("E", BOLT_MODULUS_MPA)

    # 1. Bolt tensile mechanics
    At = calculate_stress_area(d, p)
    Fp = Sp * At                                   # proof load
    Fi = (0.90 if is_permanent else 0.75) * Fp     # target preload

    # 2. Bearing mechanics & crushing check (only the head/nut faces bear, i.e.
    #    the first and last layers in the stack).
    dw = calculate_bearing_diameter(d, bolt_type, use_washer)
    Ab = calculate_bearing_area(d, dw)
    bearing_stress = Fi / Ab if Ab > 0 else float('inf')

    max_allowed_preload = float('inf')
    crushing_warning = ""
    n_layers = len(layers)
    for i, layer in enumerate(layers):
        if n_layers > 1 and 0 < i < n_layers - 1:
            continue  # interior layers do not bear directly against head/nut
        max_layer_preload = layer["Syc"] * Ab
        if max_layer_preload < max_allowed_preload:
            max_allowed_preload = max_layer_preload
            crushing_warning = f"Layer {i + 1}"

    recommended_preload = min(Fi, max_allowed_preload)
    if recommended_preload >= Fi:
        crushing_warning = ""  # no crushing risk at the target preload

    # 3. Torque (nut-factor form)
    K = FRICTION_COEFFICIENTS.get(friction_condition)
    if K is None:
        # An unrecognised friction condition (e.g. a stale project file or a renamed
        # UI label) must not silently produce a wrong torque -- surface it, then fall
        # back to the dry/as-received default so the rest of the analysis still runs.
        logger.warning("Unknown friction condition %r; falling back to K=0.20.", friction_condition)
        K = 0.20
    torque = K * recommended_preload * (d / 1000.0)

    # 4. Stiffness (bolt two-section; members as frustums in series)
    L = sum(layer["thickness"] for layer in layers)
    km_components = []
    if L > 0:
        kb = calculate_bolt_stiffness(d, At, L, bolt_E)
        z_current = 0.0
        mid_plane = L / 2.0
        
        for layer in layers:
            t = layer["thickness"]
            if t <= 0:
                continue
            
            E_layer = layer["E"]
            z_start = z_current
            z_end = z_current + t
            
            # Split layer if it crosses the mid_plane
            sub_layers = []
            if z_start < mid_plane and z_end > mid_plane:
                sub_layers.append((z_start, mid_plane))
                sub_layers.append((mid_plane, z_end))
            else:
                sub_layers.append((z_start, z_end))
            
            for z1, z2 in sub_layers:
                sub_t = z2 - z1
                if sub_t <= 0:
                    continue
                
                # Determine depth from the nearest bearing face (bolt head or nut)
                if z1 < mid_plane:
                    depth = z1
                else:
                    depth = L - z2
                
                # The cone starts at the bearing face with diameter dw and spreads at 30 deg
                D_start = dw + 2.0 * depth * math.tan(math.radians(30))
                k_sub = calculate_layer_stiffness(E_layer, sub_t, D_start, d)
                km_components.append(k_sub)
            
            z_current += t

        if any(k == 0.0 for k in km_components):
            km = 0.0
        else:
            inv_km = sum(1.0 / k for k in km_components if k != float('inf'))
            km = 1.0 / inv_km if inv_km > 0 else float('inf')

        if km == float('inf') or km + kb == 0.0:
            C = 0.0
        else:
            C = kb / (km + kb)
    else:
        kb = 0.0
        km = 0.0
        C = 0.0

    # 5. Thermal expansion (differential free expansion reacted by series spring)
    delta_T = temp_operating - temp_assembly
    delta_F_thermal = 0.0
    if L > 0 and delta_T != 0.0:
        joint_expansion = sum(layer["thickness"] * layer["CTE"] * delta_T for layer in layers)
        bolt_expansion = L * bolt_cte * delta_T
        delta_deflection = joint_expansion - bolt_expansion
        if km == float('inf'):
            delta_F_thermal = delta_deflection * kb
        elif (km + kb) > 0:
            delta_F_thermal = delta_deflection * ((kb * km) / (kb + km))

    # Embedment / short-term relaxation loss (VDI 2230): a length loss f_z relaxes
    # the joint through the bolt-member series spring, dropping the preload.
    embedment_loss = 0.0
    if L > 0 and embedment_um > 0.0:
        if km == float('inf'):
            embedment_loss = (embedment_um / 1000.0) * kb
        elif (km + kb) > 0:
            embedment_loss = (embedment_um / 1000.0) * ((kb * km) / (kb + km))

    F_operating = max(0.0, recommended_preload + delta_F_thermal - embedment_loss)

    # 6. Bolt force bounds under the external load. The VDI 2230 load-introduction
    #    factor n scales the geometric joint constant, so the bolt sees (n*C) of the
    #    external load and the members are relieved by (1 - n*C). n = 1 puts the load
    #    at the joint interfaces (the classic Shigley assumption).
    C_load = load_intro_factor * C
    F_ext_max = external_load_max
    F_ext_min = external_load_min
    Fb_max = F_operating + C_load * F_ext_max
    Fb_min = F_operating + C_load * F_ext_min

    sigma_a = (Fb_max - Fb_min) / (2.0 * At) if At > 0 else 0.0
    sigma_m = (Fb_max + Fb_min) / (2.0 * At) if At > 0 else 0.0
    sigma_i = F_operating / At if At > 0 else 0.0   # preload (mean) stress at zero ext. load

    # 7. Fatigue: the selected mean-stress criterion evaluated along the load line
    #    rising from the preload point (sigma_i, 0); only the external load cycles.
    #    See fatigue_factor_of_safety() for the per-criterion failure loci.
    bolt_mean_force = (Fb_max + Fb_min) / 2.0
    yield_force = Sy * At
    if fatigue_criterion.startswith("VDI"):
        fatigue_fos = vdi2230_endurance_fos(
            sigma_a, d, rolled_after_ht=("after" in fatigue_criterion),
            mean_force=bolt_mean_force, yield_force=yield_force)
    else:
        fatigue_fos = fatigue_factor_of_safety(
            fatigue_criterion, sigma_a, sigma_m, sigma_i, Se, Sut, Sp, Sy)
    fatigue_all = all_fatigue_factors(
        sigma_a, sigma_m, sigma_i, Se, Sut, Sp, Sy, d, bolt_mean_force, yield_force)

    # 8. Static proof FOS of the bolt at the maximum service load
    proof_fos = Fp / Fb_max if Fb_max > 0 else float('inf')

    # 9. Joint separation (member force -> 0): F_sep = F_operating / (1 - C)
    if C_load < 1.0 and (1.0 - C_load) > 0:
        separation_load = F_operating / (1.0 - C_load)
    else:
        separation_load = float('inf')
    separation_fos = separation_load / F_ext_max if F_ext_max > 0 else float('inf')

    # 10. Thread stripping. Pitch-dependent thread shear areas per unit engagement
    #     (Federal STD H28 / Machinery's Handbook, using the basic thread diameters):
    #       nut (internal) threads shear at the major diameter:  A_sn/Le = 0.875 pi d
    #       bolt (external) threads shear at the minor diameter:  A_ss/Le = 0.75 pi (d - 1.0825 p)
    #     Differential material: the joint strips at the *weaker* of the two capacities
    #     (soft tapped hole vs the bolt threads), each with its own shear yield.
    stripping_fos = float('inf')
    required_engagement = 0.0
    if internal_thread_material_props:
        Ssy_int = 0.577 * internal_thread_material_props["Syc"]    # internal (tapped) shear yield
        Ssy_bolt = 0.577 * Sy                                      # bolt-thread shear yield
        a_sn_per_len = 0.875 * math.pi * d                        # internal thread shear area / mm
        a_ss_per_len = 0.75 * math.pi * max(0.0, d - 1.0825 * p)  # external (bolt) thread area / mm
        cap_per_len = min(Ssy_int * a_sn_per_len, Ssy_bolt * a_ss_per_len)   # weaker thread governs
        # Engagement to develop the bolt proof load (so the bolt yields before the
        # threads strip) -- the usual design target.
        if cap_per_len > 0:
            required_engagement = Fp / cap_per_len
        if thread_engagement_length > 0 and Fb_max > 0:
            stripping_fos = (cap_per_len * thread_engagement_length) / Fb_max

    # 11. von Mises stress during tightening (axial tension + thread torsion).
    tightening_stress = tightening_von_mises_stress(recommended_preload, At, d, p, K)
    tightening_utilization = tightening_stress / Sy if Sy > 0 else 0.0
    tightening_fos = Sy / tightening_stress if tightening_stress > 0 else float('inf')

    return {
        "tensile_stress_area_mm2": At,
        "proof_load_N": Fp,
        "target_preload_N": Fi,
        "recommended_preload_N": recommended_preload,
        "torque_Nm": torque,
        "bearing_area_mm2": Ab,
        "bearing_stress_MPa": bearing_stress,
        "crushing_warning_material": crushing_warning,
        "kb_N_mm": kb,
        "km_N_mm": km,
        "joint_constant_C": C,
        "total_grip_length_mm": L,
        "thermal_delta_F_N": delta_F_thermal,
        "embedment_loss_N": embedment_loss,
        "operating_preload_N": F_operating,
        "max_bolt_force_N": Fb_max,
        "fatigue_sigma_a_MPa": sigma_a,
        "fatigue_sigma_m_MPa": sigma_m,
        "preload_stress_MPa": sigma_i,
        "fatigue_fos": fatigue_fos,
        "fatigue_criterion": fatigue_criterion,
        "fatigue_all_fos": fatigue_all,
        "proof_fos": proof_fos,
        "separation_load_N": separation_load,
        "separation_fos": separation_fos,
        "thread_shear_fos": stripping_fos,
        "required_engagement_mm": required_engagement,
        "tightening_stress_MPa": tightening_stress,
        "tightening_utilization": tightening_utilization,
        "tightening_fos": tightening_fos,
        "endurance_Se_MPa": Se,
        "ultimate_Sut_MPa": Sut,
        "proof_strength_MPa": Sp,
        "yield_Sy_MPa": Sy,
        "vdi_sigma_asv_MPa": vdi2230_endurance_amplitude(d, False, bolt_mean_force, yield_force),
        "vdi_sigma_asg_MPa": vdi2230_endurance_amplitude(d, True, bolt_mean_force, yield_force),
    }


# =============================================================================
# Fastener tools (torque inversion, angle control, length & engagement, sizing,
# reference dimensions). Self-contained helpers used by the "Fastener Tools" tab.
# =============================================================================

def bolt_member_forces(preload: float, C: float, external_load: float) -> Tuple[float, float]:
    """Bolt force F_b and member (clamp) force F_m under an external tensile load.

    Before separation the load shares by stiffness: F_b = F_i + C P and
    F_m = F_i - (1 - C) P (Shigley §8). Once F_m would go negative the joint has
    separated and the bolt carries the full external load: F_b = P, F_m = 0.
    Compression (P < 0) stays on the linear branch. Used by the force-vs-load chart.
    """
    if external_load <= 0.0:
        return preload + C * external_load, preload - (1.0 - C) * external_load
    fm = preload - (1.0 - C) * external_load
    if fm <= 0.0:
        return external_load, 0.0
    return preload + C * external_load, fm


def clamp_load_budget(installation: float, embedment_loss: float, thermal_delta_F: float,
                      C: float, external_load_max: float) -> List[Dict[str, Any]]:
    """Ordered clamp-load (member force) budget from installation to residual clamp.

    Steps: installation preload -> minus embedment relaxation -> plus/minus the
    thermal preload change -> minus the member relief (1 - C) P at the maximum
    external load -> residual clamp. Returns a list of step dicts with the signed
    ``delta`` and the running ``cumulative`` (N); the final ``total`` cumulative
    equals the operating preload minus the external-load relief, i.e. F_m at P_max.
    """
    steps: List[Dict[str, Any]] = []
    cum = installation
    steps.append({"label": "Installation", "delta": installation, "cumulative": cum, "kind": "start"})
    cum -= embedment_loss
    steps.append({"label": "Embedment", "delta": -embedment_loss, "cumulative": cum, "kind": "delta"})
    cum += thermal_delta_F
    steps.append({"label": "Thermal", "delta": thermal_delta_F, "cumulative": cum, "kind": "delta"})
    relief = (1.0 - C) * external_load_max if external_load_max > 0.0 else 0.0
    cum -= relief
    steps.append({"label": "Ext. load relief", "delta": -relief, "cumulative": cum, "kind": "delta"})
    steps.append({"label": "Residual clamp", "delta": cum, "cumulative": cum, "kind": "total"})
    return steps


def preload_from_yield_percent(yield_pct: float, Sy_MPa: float, At_mm2: float) -> float:
    """
    Calculate the preload force required to reach a specific percentage
    of the bolt's yield strength (Sy).
    """
    return (yield_pct / 100.0) * Sy_MPa * At_mm2


def preload_from_torque(applied_torque_Nm: float, nut_factor: float, d: float) -> float:
    """Estimate the achieved preload (N) from an applied torque, inverting T = K F d.

    ``applied_torque_Nm`` is the wrench torque in N*m, ``nut_factor`` the dimensionless
    torque coefficient K and ``d`` the nominal diameter in mm. With T in N*m and d in
    metres, F = T / (K d), i.e. F = 1000 T / (K d_mm). Returns 0 for degenerate input.
    """
    if nut_factor <= 0.0 or d <= 0.0:
        return 0.0
    return applied_torque_Nm * 1000.0 / (nut_factor * d)


def tightening_angle(preload: float, kb: float, km: float, p: float,
                     snug_preload: float = 0.0) -> float:
    """Nut rotation (degrees) past the snug point to develop ``preload`` (N).

    The bolt stretches and the members compress by the series-spring deflection
    delta = (F - F_snug)(1/kb + 1/km); one full 360 deg turn advances the nut by one
    pitch p, so theta = 360 delta / p. Returns 0 for degenerate stiffness or pitch.
    This is the elastic angle only; real run-down and embedment add to it.
    """
    if p <= 0.0 or kb <= 0.0 or km <= 0.0 or kb == float('inf') or km == float('inf'):
        return 0.0
    delta = (preload - snug_preload) * (1.0 / kb + 1.0 / km)
    return 360.0 * delta / p


def standard_thread_length(d: float, L: float, metric: bool = True) -> float:
    """Nominal thread length b (mm) for a bolt of nominal diameter ``d`` and length
    ``L`` (both mm).

    Metric (ISO 888): b = 2d + 6 (L <= 125), 2d + 12 (125 < L <= 200), else 2d + 25.
    Inch (ASME B18.2.1): b = 2d + 1/4 in (L <= 6 in) else 2d + 1/2 in, i.e.
    2d + 6.35 mm (L <= 152.4 mm) else 2d + 12.7 mm.
    """
    if metric:
        if L <= 125.0:
            return 2.0 * d + 6.0
        if L <= 200.0:
            return 2.0 * d + 12.0
        return 2.0 * d + 25.0
    if L <= 152.4:            # 6 in
        return 2.0 * d + 6.35   # 1/4 in
    return 2.0 * d + 12.7        # 1/2 in


def thread_series_options(size_name: str, metric: bool = True) -> List[Tuple[str, float]]:
    """Available (label, pitch_mm) thread series for a bolt size; coarse/UNC first."""
    table = BOLT_THREAD_SERIES_METRIC if metric else BOLT_THREAD_SERIES_IMPERIAL
    return table.get(size_name, [])


def thread_designation(size_name: str, pitch_label: str, metric: bool, p: float) -> str:
    """Full thread designation for display, e.g. 'M12x1.5' or '1/2-20 UNF'.

    For inch threads the TPI is recovered from the pitch and the series token (UNC /
    UNF / UNEF) is taken from the start of ``pitch_label``.
    """
    if metric:
        return f"{size_name}×{p:g}"
    series = pitch_label.split()[0] if pitch_label else ""
    tpi = round(25.4 / p) if p > 0 else 0
    return f"{size_name}-{tpi} {series}".strip()


def recommend_bolt_length(grip: float, extra_stack: float,
                          lengths: Iterable[float]) -> Tuple[float, Optional[float]]:
    """Minimum required bolt length and the next standard length >= it (mm).

    ``grip`` is the clamped material thickness the bolt passes through and
    ``extra_stack`` the length consumed beyond the grip (washers + nut height +
    thread protrusion). Returns (L_min, recommended); recommended is None when no
    supplied standard length is long enough.
    """
    l_min = grip + extra_stack
    for length in sorted(lengths):
        if length >= l_min:
            return l_min, length
    return l_min, None


def grip_thread_engagement(d: float, L: float, grip: float, metric: bool = True) -> Dict[str, Any]:
    """Whether the threaded portion of a length-``L`` bolt falls within the grip.

    The unthreaded shank length is L - b (b from ISO 888 for metric, ASME B18.2.1
    for inch). If the shank is shorter than the grip, threads lie inside the clamped
    length, lowering bolt stiffness (the reduced tensile-stress-area section then
    spans part of the grip). Returns the thread length, shank length (mm) and a
    boolean ``threads_in_grip``.
    """
    b = standard_thread_length(d, L, metric)
    shank = max(0.0, L - b)
    return {"thread_length_mm": b, "shank_length_mm": shank,
            "threads_in_grip": shank < grip}


def bolt_hardware_reference(size_name: str, d: float, p: float) -> Dict[str, Optional[float]]:
    """Typical wrench/hole reference dimensions (mm) for a bolt size.

    ``hex_af`` = hex head/nut width across flats; ``socket_af`` = hex-key (Allen)
    size for a socket-head cap screw; ``clearance`` = typical (free-fit) clearance
    hole; ``tap_drill`` = the nominal-minus-pitch rule (~ standard coarse tap drill).
    Wrench/socket/clearance values come from BOLT_HARDWARE; clearance falls back to
    the 1.1 d rule used elsewhere when the size is not tabulated.
    """
    hw = BOLT_HARDWARE.get(size_name, {})
    clearance = hw.get("clearance", round(d * 1.1, 2))
    return {
        "hex_af_mm": hw.get("hex_af"),
        "socket_af_mm": hw.get("socket_af"),
        "clearance_hole_mm": clearance,
        "tap_drill_mm": max(0.0, d - p),
    }


def recommend_bolt(
    sizes: Dict[str, Tuple[float, float]],
    materials: Mapping[str, BoltMaterial],
    layers: Sequence[Layer],
    *,
    bolt_type: str,
    use_washer: bool,
    is_permanent: bool,
    friction_condition: str,
    temp_assembly: float = 20.0,
    temp_operating: float = 20.0,
    external_load_max: float = 0.0,
    external_load_min: float = 0.0,
    fatigue_criterion: str = "Goodman",
    target_proof_fos: float = 1.5,
    target_fatigue_fos: float = 1.5,
    target_separation_fos: float = 1.1,
    thread_series: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> RecommendResult:
    """Smallest bolt (by tensile-stress area) meeting the factor-of-safety targets.

    Sweeps every size x material combination through ``calculate_preload`` with the
    current joint, keeping those that meet the proof target (always), the separation
    target (only when an external load is present) and the fatigue target (only when
    the load is cyclic, i.e. max != min). When ``thread_series`` (size -> list of
    ``(label, pitch_mm)``) is supplied every available pitch of each size is tried;
    otherwise only the coarse pitch from ``sizes`` is used. Returns the lightest
    passing candidate plus the full ranked list. All loads are PER BOLT, in newtons.
    """
    cyclic = external_load_max != external_load_min
    has_ext = external_load_max > 0.0 or external_load_min > 0.0
    passing: List[BoltCandidate] = []
    for size_name, (d, coarse_p) in sizes.items():
        pitch_opts = (thread_series.get(size_name) if thread_series else None) or [("", coarse_p)]
        for thread_label, p in pitch_opts:
            for mat_name, props in materials.items():
                res = calculate_preload(
                    d=d, p=p, bolt_material_props=props, layers=layers,
                    bolt_type=bolt_type, use_washer=use_washer, is_permanent=is_permanent,
                    friction_condition=friction_condition, temp_assembly=temp_assembly,
                    temp_operating=temp_operating, external_load_max=external_load_max,
                    external_load_min=external_load_min, fatigue_criterion=fatigue_criterion)
                ok = res["proof_fos"] >= target_proof_fos
                if has_ext:
                    ok = ok and res["separation_fos"] >= target_separation_fos
                if cyclic:
                    ok = ok and res["fatigue_fos"] >= target_fatigue_fos
                if ok:
                    passing.append({
                        "size": size_name, "material": mat_name,
                        "pitch_mm": p, "thread": thread_label,
                        "stress_area_mm2": res["tensile_stress_area_mm2"],
                        "proof_fos": res["proof_fos"],
                        "fatigue_fos": res["fatigue_fos"],
                        "separation_fos": res["separation_fos"],
                        "preload_N": res["recommended_preload_N"],
                        "torque_Nm": res["torque_Nm"],
                    })
    passing.sort(key=lambda c: c["stress_area_mm2"])
    return {"found": bool(passing),
            "best": passing[0] if passing else None,
            "candidates": passing}


# =============================================================================
# Bolt-group / pattern analysis
# -----------------------------------------------------------------------------
# A bolted joint is rarely a single fastener. These helpers distribute joint
# loads over a pattern of bolts so the GOVERNING (most highly loaded) bolt can be
# fed into calculate_preload above. The elastic ("rigid member") method is used:
#   * Tension from an overturning moment is shared in proportion to each bolt's
#     distance from the centroidal bending axis (Sum of A*d^2, equal areas).
#   * In-plane shear with eccentricity is the classic torsional bolt-group:
#     direct shear V/N plus torsional shear T*r_i/J added as vectors (J is the
#     polar second moment of the bolt areas about the centroid).
# These are standard first-pass methods (Shigley Ch. 8, AISC, Machinery's
# Handbook). The centroidal tension model can be unconservative versus a
# neutral-axis-at-edge model for prying-dominated joints.
# =============================================================================

def rectangular_pattern(rows: int, cols: int, pitch_x: float, pitch_y: float) -> List[Tuple[float, float]]:
    """Bolt (x, y) coordinates (mm) for a rows x cols grid, centred on the origin."""
    coords: List[Tuple[float, float]] = []
    for r in range(max(0, int(rows))):
        for c in range(max(0, int(cols))):
            x = (c - (cols - 1) / 2.0) * pitch_x
            y = (r - (rows - 1) / 2.0) * pitch_y
            coords.append((x, y))
    return coords


def circular_pattern(n: int, bolt_circle_dia: float, start_angle_deg: float = 0.0) -> List[Tuple[float, float]]:
    """Bolt (x, y) coordinates (mm) for n bolts equally spaced on a bolt circle."""
    coords: List[Tuple[float, float]] = []
    n = max(0, int(n))
    radius = bolt_circle_dia / 2.0
    for i in range(n):
        angle = math.radians(start_angle_deg) + 2.0 * math.pi * i / n
        coords.append((radius * math.cos(angle), radius * math.sin(angle)))
    return coords


def analyze_bolt_group(
    coords: List[Tuple[float, float]],
    axial_load: float = 0.0,
    moment: float = 0.0,
    moment_axis: str = "x",
    shear_load: float = 0.0,
    shear_eccentricity: float = 0.0,
) -> BoltGroupResult:
    """Elastic bolt-group analysis (rigid members, rotation about the centroid).

    All inputs are Metric: coordinates and eccentricity in mm, forces in N,
    moment in N*mm. Returns per-bolt tensile/shear forces (N) and governing
    values for the most highly loaded bolt.

    axial_load          : concentric tensile load, shared equally (N).
    moment              : overturning moment (N*mm); tension varies linearly with
                          distance from the centroidal bending axis.
    moment_axis         : 'x' -> bending about the centroidal x-axis (tension
                          varies with y); 'y' -> about the y-axis (varies with x).
    shear_load          : in-plane shear force (N).
    shear_eccentricity  : distance (mm) from the centroid to the shear line of
                          action -> torque T = shear_load * shear_eccentricity.
    """
    n = len(coords)
    if n == 0:
        return {
            "centroid": (0.0, 0.0),
            "tensions_N": [],
            "shears_N": [],
            "shear_vectors_N": [],
            "governing_index": -1,
            "governing_tension_N": 0.0,
            "governing_shear_N": 0.0,
            "sum_distance_sq_mm2": 0.0,
            "polar_moment_mm2": 0.0,
            "moment_reactable": False,
        }

    c_arr = np.array(coords, dtype=float)
    centroid = np.mean(c_arr, axis=0)
    cx, cy = float(centroid[0]), float(centroid[1])
    rel = c_arr - centroid

    # --- Tension distribution (axial share + moment * d / Sum d^2) ---
    if moment_axis == "y":
        dist = rel[:, 0]   # tension varies with x
    else:
        dist = rel[:, 1]   # bending about x-axis: tension varies with y

    sum_d2 = float(np.sum(dist * dist))
    moment_reactable = sum_d2 > 0.0

    tensions = np.full(n, axial_load / n)
    if moment_reactable:
        tensions += moment * dist / sum_d2

    # --- Shear distribution (direct V/N + torsional T*r/J, vector sum) ---
    J = float(np.sum(rel[:, 0]**2 + rel[:, 1]**2))
    torque = shear_load * shear_eccentricity
    direct = shear_load / n

    if J > 0.0:
        fx = -torque * rel[:, 1] / J
        fy = torque * rel[:, 0] / J
    else:
        fx = np.zeros(n)
        fy = np.zeros(n)

    vx = fx
    vy = direct + fy
    shears = np.hypot(vx, vy)

    gov_idx = int(np.argmax(tensions))
    return {
        "centroid": (cx, cy),
        "tensions_N": tensions.tolist(),
        "shears_N": shears.tolist(),
        "shear_vectors_N": list(zip(vx.tolist(), vy.tolist())),
        "governing_index": gov_idx,
        "governing_tension_N": float(tensions[gov_idx]),
        "governing_shear_N": float(np.max(shears)),
        "sum_distance_sq_mm2": sum_d2,
        "polar_moment_mm2": J,
        "moment_reactable": moment_reactable,
    }


# =============================================================================
# External FE results import
# -----------------------------------------------------------------------------
# Evaluate per-bolt results exported from a finite-element model. The CSV gives the
# TOTAL bolt forces (already including preload and contact load-sharing), so the
# factors of safety are computed directly from those forces -- the joint-stiffness
# (joint-constant C) model used elsewhere is NOT re-applied here. All inputs are SI
# (N, mm, MPa). See evaluate_fe_rows for the accepted columns/aliases.
# =============================================================================

# Accepted CSV column names per logical field (matched case-insensitively).
_FE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "bolt_id": ("bolt_id", "id", "bolt", "label", "name"),
    "diameter": ("diameter_mm", "diameter", "d", "d_mm"),
    "pitch": ("pitch_mm", "pitch", "p", "p_mm"),
    "grade": ("bolt_grade", "grade"),
    "proof": ("proof_mpa", "proof", "sp"),
    "yield": ("yield_mpa", "yield", "sy"),
    "ultimate": ("ultimate_mpa", "ultimate", "sut", "uts"),
    "endurance": ("endurance_mpa", "endurance", "se"),
    "axial_max": ("axial_force_max_n", "axial_max", "axial_force_max", "tension_max"),
    "axial_min": ("axial_force_min_n", "axial_min", "axial_force_min", "tension_min"),
    "shear_max": ("shear_force_max_n", "shear_max", "shear", "shear_force_max", "v_max"),
    "preload": ("preload_n", "preload", "fi"),
}


def combined_tension_shear_fos(axial_force: float, shear_force: float, At: float,
                               Sp: float, Sy: float) -> float:
    """Elliptic tension-shear interaction factor of safety for a bolt.

    The operating point is scaled onto the unit ellipse
    (sigma/Sp)^2 + (tau/(0.577 Sy))^2 = 1 and the FoS is that scale factor. Shear
    uses the tensile-stress area (threads assumed in the shear plane -- conservative).
    Returns +inf when there is no load. Shared by the FE-import per-bolt check and
    the interactive bolt-group governing-bolt check.
    """
    if At <= 0.0:
        return float('inf')
    sigma = axial_force / At
    tau = shear_force / At
    shear_allow = 0.577 * Sy
    util = 0.0
    if Sp > 0:
        util += (sigma / Sp) ** 2
    if shear_allow > 0:
        util += (tau / shear_allow) ** 2
    return (1.0 / math.sqrt(util)) if util > 0 else float('inf')


def evaluate_fe_bolt(d: float, p: float, Sp: float, Sy: float, Sut: float, Se: float,
                     axial_max: float, axial_min: float, preload: float = 0.0,
                     shear_max: float = 0.0, fatigue_criterion: str = "Goodman",
                     target_fos: float = 1.5) -> Dict[str, Any]:
    """Factor-of-safety evaluation of one bolt from external FE results (SI units).

    ``axial_max``/``axial_min`` are the TOTAL bolt tension over the duty cycle.
    Returns the stress area, alternating/mean stresses and the proof, fatigue, shear
    and combined tension-shear factors of safety, plus the governing (minimum) FoS
    and a pass/fail flag against ``target_fos``. Shear uses the tensile-stress area
    (threads assumed in the shear plane -- conservative).
    """
    At = calculate_stress_area(d, p)
    sigma_max = axial_max / At if At > 0 else 0.0
    sigma_min = axial_min / At if At > 0 else 0.0
    sigma_a = max(0.0, (sigma_max - sigma_min) / 2.0)
    sigma_m = (sigma_max + sigma_min) / 2.0
    # Load line starts at the steady (preload) point; fall back to the minimum stress.
    sigma_i = (preload / At) if (preload > 0.0 and At > 0) else sigma_min

    Fp = Sp * At
    proof_fos = Fp / axial_max if axial_max > 0 else float('inf')

    if fatigue_criterion.startswith("VDI"):
        fatigue_fos = vdi2230_endurance_fos(
            sigma_a, d, rolled_after_ht=("after" in fatigue_criterion),
            mean_force=(axial_max + axial_min) / 2.0, yield_force=Sy * At)
    else:
        fatigue_fos = fatigue_factor_of_safety(
            fatigue_criterion, sigma_a, sigma_m, sigma_i, Se, Sut, Sp, Sy)

    shear_allow = 0.577 * Sy
    tau = shear_max / At if At > 0 else 0.0
    shear_fos = shear_allow / tau if tau > 0 else float('inf')

    # Combined tension-shear: elliptic interaction (see combined_tension_shear_fos).
    combined_fos = combined_tension_shear_fos(axial_max, shear_max, At, Sp, Sy)

    checks = [("Proof", proof_fos), ("Fatigue", fatigue_fos),
              ("Shear", shear_fos), ("Combined", combined_fos)]
    governing, min_fos = min(checks, key=lambda t: t[1])
    return {
        "stress_area_mm2": At,
        "sigma_max_MPa": sigma_max, "sigma_a_MPa": sigma_a, "sigma_m_MPa": sigma_m,
        "proof_fos": proof_fos, "fatigue_fos": fatigue_fos, "shear_fos": shear_fos,
        "combined_fos": combined_fos, "min_fos": min_fos, "governing": governing,
        "passes": min_fos >= target_fos,
    }


def evaluate_fe_rows(rows: Any, bolt_grades: Mapping[str, BoltMaterial],
                     fatigue_criterion: str = "Goodman",
                     target_fos: float = 1.5) -> List[Dict[str, Any]]:
    """Resolve and evaluate a list of raw FE rows or a Pandas DataFrame.
    Rows sharing a ``bolt_id`` are enveloped (max tension / min tension / max
    shear). Material comes from ``bolt_grades`` when ``bolt_grade`` matches, else from
    explicit proof/yield/ultimate/endurance columns. Returns one result dict per
    unique bolt, each carrying the evaluation plus an ``error`` string ('' when ok).
    """
    if isinstance(rows, list):
        if not rows:
            return []
        df = pd.DataFrame(rows)
    else:
        df = rows.copy()

    if df.empty:
        return []

    # Lowercase column names and strip whitespace
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map columns based on _FE_ALIASES
    col_map = {}
    for logical, aliases in _FE_ALIASES.items():
        for col in df.columns:
            if col in aliases and logical not in col_map:
                col_map[logical] = col
                break

    if "bolt_id" not in col_map:
        return []

    # Standardize columns to the logical names
    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Convert numeric columns
    numeric_cols = ["diameter", "pitch", "proof", "yield", "ultimate", "endurance",
                    "axial_max", "axial_min", "shear_max", "preload"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Group by bolt_id
    agg_dict = {}
    for col in ["diameter", "pitch", "grade", "preload", "proof", "yield", "ultimate", "endurance"]:
        if col in df.columns:
            agg_dict[col] = "first"

    if "axial_max" in df.columns:
        agg_dict["axial_max"] = "max"
    if "axial_min" in df.columns:
        agg_dict["axial_min"] = "min"
    if "shear_max" in df.columns:
        agg_dict["shear_max"] = "max"

    grouped = df.groupby("bolt_id", as_index=False).agg(agg_dict)

    results = []
    for _, row in grouped.iterrows():
        bid = str(row["bolt_id"])
        base = {"bolt_id": bid}

        d = float(row.get("diameter", np.nan))
        p = float(row.get("pitch", np.nan))

        sp = sy = sut = se = np.nan
        grade = str(row.get("grade", "")).strip()
        if grade and grade in bolt_grades:
            props = bolt_grades[grade]
            sp = float(props["Sp"])
            sy = float(props.get("Sy", props["Sp"] / 0.9))
            sut = float(props["Sut"])
            se = float(props["Se"])
        else:
            sp = float(row.get("proof", np.nan))
            sy = float(row.get("yield", np.nan))
            sut = float(row.get("ultimate", np.nan))
            se = float(row.get("endurance", np.nan))

        axial_max = float(row.get("axial_max", np.nan))
        axial_min = float(row.get("axial_min", np.nan))
        shear_max = float(row.get("shear_max", 0.0))
        if pd.isna(shear_max):
            shear_max = 0.0
        preload = float(row.get("preload", 0.0))
        if pd.isna(preload):
            preload = 0.0

        if pd.isna(d) or pd.isna(p) or d <= 0 or p <= 0:
            results.append({**base, "error": "missing/invalid diameter or pitch"})
            continue

        if pd.isna(sp) or pd.isna(sy) or pd.isna(sut) or pd.isna(se) or min(sp, sy, sut, se) <= 0:
            msg = "missing/invalid bolt strength (bolt_grade or proof/yield/ultimate/endurance)"
            results.append({**base, "error": msg})
            continue

        if pd.isna(axial_max):
            results.append({**base, "error": "missing axial_force_max"})
            continue

        if pd.isna(axial_min):
            axial_min = 0.0

        ev = evaluate_fe_bolt(d, p, sp, sy, sut, se, axial_max, axial_min,
                              preload=preload, shear_max=shear_max,
                              fatigue_criterion=fatigue_criterion, target_fos=target_fos)

        ev.update({**base, "error": "", "diameter_mm": d, "pitch_mm": p,
                   "axial_max_N": axial_max, "axial_min_N": axial_min,
                   "shear_max_N": shear_max, "preload_N": preload,
                   "Sp_MPa": sp, "Sy_MPa": sy, "Sut_MPa": sut, "Se_MPa": se})
        results.append(ev)

    return results

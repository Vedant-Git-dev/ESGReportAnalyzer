"""
config/constants.py
-------------------
Shared constants: XBRL namespaces, fiscal year bounds, Groq model,
KPI metadata, benchmark thresholds and display-group definitions.

Metric type taxonomy
--------------------
OPERATIONAL_SCALE_METRICS  — absolute totals; context/size only, never ranked
ESG_EFFICIENCY_EMP_METRICS — per-employee intensities; lower=better; ranked
ESG_EFFICIENCY_REV_METRICS — per-revenue intensities; lower=better; ranked (new)
ESG_POLICY_METRICS         — rates/shares; higher=better; ranked
SOCIAL_METRICS             — diversity ratios; higher=better; ranked
SAFETY_METRICS             — incident counts/rates; lower=better; ranked
GOVERNANCE_METRICS         — complaint counts; lower=better; ranked

ESG_EFFICIENCY_METRICS     — union of EMP + REV efficiency metrics

RANKABLE_KPIS              — efficiency + policy + social + safety + governance
                             (NO operational scale metrics)

ESG Category Weights (for composite score)
-------------------------------------------
Environment  = 40%  (efficiency intensities + policy rates)
Social       = 30%  (diversity ratios)
Governance   = 30%  (safety + complaints)

Benchmarking method
-------------------
  industry mode → median of peer values (robust to outliers)
  Also computes Q1 (25th pctile) and Q3 (75th pctile) for quartile banding
"""

# ── XBRL namespace URIs ───────────────────────────────────────────────────────
CAPMKT = "https://www.sebi.gov.in/xbrl/2025-05-31/in-capmkt"
XBRLI  = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"

# ── Fiscal year bounds ────────────────────────────────────────────────────────
CY_START = "2024-04-01"
CY_END   = "2025-03-31"

# ── Groq model ────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# ══════════════════════════════════════════════════════════════════════════════
# METRIC TYPE TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

# Absolute totals — scale / impact indicators ONLY.
# NEVER ranked, NEVER used in gap analysis.
OPERATIONAL_SCALE_METRICS: set[str] = {
    "TotalGHG_tCO2e",
    "Scope1_tCO2e",
    "Scope2_tCO2e",
    "TotalEnergy_GJ",
    "RenewableEnergy_GJ",
    "NonRenewableEnergy_GJ",
    "WaterConsumption_KL",
    "WaterWithdrawal_KL",
    "WasteGenerated_MT",
    "WasteRecovered_MT",
    "WasteDisposed_MT",
    "PlasticWaste_MT",
    "EWaste_MT",
    "HazardousWaste_MT",
    "Turnover_INR",
    "NetWorth_INR",
    "PaidUpCapital_INR",
    # raw headcounts
    "perm_emp_total", "perm_emp_male", "perm_emp_female",
    "perm_wkr_total", "perm_wkr_male", "perm_wkr_female",
    "contract_wkr_total", "total_emp_total", "total_wkr_total",
    "nonperm_emp_total", "plants_national", "offices_national",
    "Emp_HS_Training", "Wkr_HS_Training",
}

# Per-employee environmental intensities — lower = better — RANKED
ESG_EFFICIENCY_EMP_METRICS: set[str] = {
    "GHG_tCO2e_perEmployee",
    "Energy_GJ_perEmployee",
    "Water_KL_perEmployee",
    "Waste_MT_perEmployee",
}

# Per-revenue environmental intensities — lower = better — RANKED
# Used when companies differ greatly in workforce size (capital-intensive sectors)
# Units: tCO2e/₹Cr, GJ/₹Cr, KL/₹Cr, MT/₹Cr
ESG_EFFICIENCY_REV_METRICS: set[str] = {
    "GHG_tCO2e_perRevCr",
    "Energy_GJ_perRevCr",
    "Water_KL_perRevCr",
    "Waste_MT_perRevCr",
}

# Combined efficiency set
ESG_EFFICIENCY_METRICS: set[str] = (
    ESG_EFFICIENCY_EMP_METRICS | ESG_EFFICIENCY_REV_METRICS
)

# Policy / commitment rates — higher = better — RANKED
ESG_POLICY_METRICS: set[str] = {
    "RenewableEnergyShare_Pct",
    "WasteRecoveryRate_Pct",
    "Pct_Emp_HS_Training",
    "Pct_Wkr_HS_Training",
}

# Social / workforce diversity — higher = better — RANKED
SOCIAL_METRICS: set[str] = {
    "Female_Ratio_PermanentEmp",
    "Female_Ratio_PermanentWorkers",
    "Female_Ratio_ContractWorkers",
    "Female_Ratio_AllEmployees",
    "Female_Ratio_AllWorkers",
}

# Safety — lower = better — RANKED
SAFETY_METRICS: set[str] = {
    "Fatalities",
    "LTIFR",
    "HighConsequenceInjuries",
    "RecordableInjuries",
}

# Governance — lower = better — RANKED
GOVERNANCE_METRICS: set[str] = {
    "Complaints_Filed",
    "Complaints_Pending",
}

# Denominator floor for ratio reliability
MIN_RATIO_DENOMINATOR: int = 30

# ── RANKABLE_KPIS = all typed sets EXCEPT operational scale ───────────────────
RANKABLE_KPIS: set[str] = (
    ESG_EFFICIENCY_METRICS
    | ESG_POLICY_METRICS
    | SOCIAL_METRICS
    | SAFETY_METRICS
    | GOVERNANCE_METRICS
)

# ── Lower = better ────────────────────────────────────────────────────────────
LOWER_IS_BETTER: set[str] = (
    ESG_EFFICIENCY_METRICS
    | SAFETY_METRICS
    | GOVERNANCE_METRICS
)

# ══════════════════════════════════════════════════════════════════════════════
# ESG COMPOSITE SCORING WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

ESG_WEIGHTS: dict[str, float] = {
    "Environment": 0.40,
    "Social":      0.30,
    "Governance":  0.30,
}

# Which KPIs belong to each ESG pillar (for composite scoring)
ESG_PILLAR_KPIS: dict[str, list[str]] = {
    "Environment": [
        "GHG_tCO2e_perEmployee",
        "Energy_GJ_perEmployee",
        "Water_KL_perEmployee",
        "Waste_MT_perEmployee",
        "GHG_tCO2e_perRevCr",
        "Energy_GJ_perRevCr",
        "RenewableEnergyShare_Pct",
        "WasteRecoveryRate_Pct",
    ],
    "Social": [
        "Female_Ratio_PermanentEmp",
        "Female_Ratio_PermanentWorkers",
        "Female_Ratio_ContractWorkers",
        "Female_Ratio_AllEmployees",
        "Female_Ratio_AllWorkers",
        "Pct_Emp_HS_Training",
        "Pct_Wkr_HS_Training",
    ],
    "Governance": [
        "Fatalities",
        "LTIFR",
        "HighConsequenceInjuries",
        "Complaints_Filed",
        "Complaints_Pending",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARKING METHOD
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_METHOD = "median"   # "median" (robust) or "mean"

# Scale-ratio threshold: if max_turnover / min_turnover > this, add disclaimer
SCALE_RATIO_THRESHOLD: float = 10.0

# ── ESG scoring thresholds ────────────────────────────────────────────────────
# Legacy ±10% thresholds — used only when < 4 peers (percentile method needs ≥4)
GOOD_THRESHOLD: float = 0.10
BAD_THRESHOLD:  float = 0.10

# Percentile bands for executive summary scoring (used when ≥4 peers available)
PERCENTILE_GOOD:  float = 75.0   # top quartile
PERCENTILE_BAD:   float = 25.0   # bottom quartile

# ── Static industry benchmarks (used when computed benchmark unavailable) ─────
STATIC_BENCHMARKS: dict[str, float] = {
    # Social
    "Female_Ratio_PermanentEmp":     15.0,
    "Female_Ratio_PermanentWorkers": 10.0,
    "Female_Ratio_ContractWorkers":   5.0,
    "Female_Ratio_AllEmployees":     15.0,
    "Female_Ratio_AllWorkers":        8.0,
    # Policy
    "RenewableEnergyShare_Pct":      30.0,
    "WasteRecoveryRate_Pct":         85.0,
    "Pct_Emp_HS_Training":           90.0,
    "Pct_Wkr_HS_Training":           85.0,
    # Safety
    "LTIFR":                          0.30,
    "Fatalities":                     0.0,
    # Efficiency (Indian energy sector reference values, per employee)
    "GHG_tCO2e_perEmployee":          5.0,
    "Energy_GJ_perEmployee":         80.0,
    "Water_KL_perEmployee":         150.0,
    "Waste_MT_perEmployee":           3.0,
    # Efficiency (Indian energy sector reference values, per ₹Cr revenue)
    "GHG_tCO2e_perRevCr":            0.05,
    "Energy_GJ_perRevCr":            0.80,
    "Water_KL_perRevCr":             1.50,
    "Waste_MT_perRevCr":             0.03,
}

# ── Human-readable KPI labels ─────────────────────────────────────────────────
KPI_LABELS: dict[str, str] = {
    # Per-employee efficiency
    "GHG_tCO2e_perEmployee":         "GHG Intensity / Employee (tCO2e)  ↓better",
    "Energy_GJ_perEmployee":         "Energy Intensity / Employee (GJ)  ↓better",
    "Water_KL_perEmployee":          "Water Intensity / Employee (KL)  ↓better",
    "Waste_MT_perEmployee":          "Waste Intensity / Employee (MT)  ↓better",
    # Per-revenue efficiency
    "GHG_tCO2e_perRevCr":            "GHG Intensity / ₹Cr Revenue (tCO2e)  ↓better",
    "Energy_GJ_perRevCr":            "Energy Intensity / ₹Cr Revenue (GJ)  ↓better",
    "Water_KL_perRevCr":             "Water Intensity / ₹Cr Revenue (KL)  ↓better",
    "Waste_MT_perRevCr":             "Waste Intensity / ₹Cr Revenue (MT)  ↓better",
    # Policy
    "RenewableEnergyShare_Pct":      "Renewable Energy Share (%)",
    "WasteRecoveryRate_Pct":         "Waste Recovery Rate (%)",
    "Pct_Emp_HS_Training":           "Employees H&S Trained (%)",
    "Pct_Wkr_HS_Training":           "Workers H&S Trained (%)",
    # Social
    "Female_Ratio_PermanentEmp":     "Female % — Permanent Employees",
    "Female_Ratio_PermanentWorkers": "Female % — Permanent Workers",
    "Female_Ratio_ContractWorkers":  "Female % — Contract Workers",
    "Female_Ratio_AllEmployees":     "Female % — All Employees",
    "Female_Ratio_AllWorkers":       "Female % — All Workers",
    # Safety
    "Fatalities":                    "Fatalities  ↓better",
    "LTIFR":                         "Lost-Time Injury Rate  ↓better",
    "HighConsequenceInjuries":       "High-Consequence Injuries  ↓better",
    # Governance
    "Complaints_Filed":              "Complaints Filed  ↓better",
    "Complaints_Pending":            "Complaints Pending  ↓better",
    # Operational scale (display-only)
    "TotalGHG_tCO2e":                "Total GHG Scope1+2 (tCO2e)  [operational scale]",
    "TotalEnergy_GJ":                "Total Energy (GJ)  [operational scale]",
    "WaterConsumption_KL":           "Water Consumption (KL)  [operational scale]",
    "WasteGenerated_MT":             "Waste Generated (MT)  [operational scale]",
}

# ── Display groups for Section 1 KPI table ────────────────────────────────────
DISPLAY_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Workforce — Headcount [operational scale]", [
        ("perm_emp_total",    "Permanent Employees"),
        ("perm_wkr_total",    "Permanent Workers"),
        ("contract_wkr_total","Contract Workers"),
        ("total_emp_total",   "Total Employees"),
        ("total_wkr_total",   "Total Workers"),
    ]),
    ("Workforce — Diversity [social / ranked]", [
        ("Female_Ratio_PermanentEmp",     "Female % (Perm Employees)"),
        ("Female_Ratio_PermanentWorkers", "Female % (Perm Workers)"),
        ("Female_Ratio_ContractWorkers",  "Female % (Contract Workers)"),
        ("Female_Ratio_AllEmployees",     "Female % (All Employees)"),
        ("Female_Ratio_AllWorkers",       "Female % (All Workers)"),
    ]),
    ("Environmental — Operational Scale [context only, not ranked]", [
        ("TotalEnergy_GJ",    "Total Energy (GJ)"),
        ("RenewableEnergy_GJ","Renewable Energy (GJ)"),
        ("TotalGHG_tCO2e",   "Total GHG Scope1+2 (tCO2e)"),
        ("Scope1_tCO2e",      "GHG Scope 1 (tCO2e)"),
        ("Scope2_tCO2e",      "GHG Scope 2 (tCO2e)"),
        ("WaterConsumption_KL","Water Consumption (KL)"),
        ("WasteGenerated_MT", "Waste Generated (MT)"),
        ("WasteRecovered_MT", "Waste Recovered (MT)"),
    ]),
    ("Environmental — ESG Efficiency per Employee [ranked ↓]", [
        ("GHG_tCO2e_perEmployee",  "GHG / Employee (tCO2e)"),
        ("Energy_GJ_perEmployee",  "Energy / Employee (GJ)"),
        ("Water_KL_perEmployee",   "Water / Employee (KL)"),
        ("Waste_MT_perEmployee",   "Waste / Employee (MT)"),
    ]),
    ("Environmental — ESG Efficiency per ₹Cr Revenue [ranked ↓]", [
        ("GHG_tCO2e_perRevCr",  "GHG / ₹Cr Revenue (tCO2e)"),
        ("Energy_GJ_perRevCr",  "Energy / ₹Cr Revenue (GJ)"),
        ("Water_KL_perRevCr",   "Water / ₹Cr Revenue (KL)"),
        ("Waste_MT_perRevCr",   "Waste / ₹Cr Revenue (MT)"),
    ]),
    ("Environmental — ESG Policy [ranked ↑]", [
        ("RenewableEnergyShare_Pct", "Renewable Energy Share (%)"),
        ("WasteRecoveryRate_Pct",    "Waste Recovery Rate (%)"),
    ]),
    ("Safety [ranked ↓]", [
        ("Fatalities",           "Fatalities"),
        ("LTIFR",                "Lost-Time Injury Rate"),
        ("HighConsequenceInjuries","High-Consequence Injuries"),
    ]),
    ("Training [ranked ↑]", [
        ("Pct_Emp_HS_Training", "Employees H&S Trained (%)"),
        ("Pct_Wkr_HS_Training", "Workers H&S Trained (%)"),
    ]),
    ("Financial — Operational Scale [context only, not ranked]", [
        ("Turnover_INR",  "Annual Turnover (INR)"),
        ("NetWorth_INR",  "Net Worth (INR)"),
    ]),
    ("Governance [ranked ↓]", [
        ("Complaints_Filed",   "Complaints Filed"),
        ("Complaints_Pending", "Complaints Pending"),
    ]),
]
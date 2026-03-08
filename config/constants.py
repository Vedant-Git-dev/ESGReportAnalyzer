"""
config/constants.py
-------------------
Shared constants: XBRL namespaces, fiscal year bounds, Groq model,
KPI metadata, benchmark thresholds and display-group definitions.

Metric type taxonomy
--------------------
OPERATIONAL_SCALE_METRICS  — absolute totals; context/size only, never ranked
ESG_EFFICIENCY_METRICS     — per-employee intensities; lower = better; ranked
ESG_POLICY_METRICS         — rates/shares; higher = better; ranked
SOCIAL_METRICS             — diversity ratios; higher = better; ranked
SAFETY_METRICS             — incident counts/rates; lower = better; ranked
GOVERNANCE_METRICS         — complaint counts; lower = better; ranked

RANKABLE_KPIS              — union of efficiency + policy + social + safety + governance
                             (NO operational scale metrics)
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
# Displayed in "Operational Scale / Environmental Impact" section.
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

# Per-employee intensities — lower = better — RANKED & gap-analysed
ESG_EFFICIENCY_METRICS: set[str] = {
    "GHG_tCO2e_perEmployee",
    "Energy_GJ_perEmployee",
    "Water_KL_perEmployee",
    "Waste_MT_perEmployee",
}

# Policy / commitment rates — higher = better — RANKED & gap-analysed
ESG_POLICY_METRICS: set[str] = {
    "RenewableEnergyShare_Pct",
    "WasteRecoveryRate_Pct",
    "Pct_Emp_HS_Training",
    "Pct_Wkr_HS_Training",
}

# Social / workforce diversity — higher = better — RANKED & gap-analysed
SOCIAL_METRICS: set[str] = {
    "Female_Ratio_PermanentEmp",
    "Female_Ratio_PermanentWorkers",
    "Female_Ratio_ContractWorkers",
    "Female_Ratio_AllEmployees",
    "Female_Ratio_AllWorkers",
    # Female_Ratio_NonPermEmployees is computed but subject to denominator check
}

# Safety — lower = better — RANKED & gap-analysed
SAFETY_METRICS: set[str] = {
    "Fatalities",
    "LTIFR",
    "HighConsequenceInjuries",
    "RecordableInjuries",
}

# Governance — lower = better — RANKED & gap-analysed
GOVERNANCE_METRICS: set[str] = {
    "Complaints_Filed",
    "Complaints_Pending",
}

# Minimum denominator for ratio metrics to be statistically reliable
MIN_RATIO_DENOMINATOR: int = 30

# ── KPIs eligible for ranking and gap analysis ────────────────────────────────
# Strict union of typed sets — operational scale metrics are EXCLUDED
RANKABLE_KPIS: set[str] = (
    ESG_EFFICIENCY_METRICS
    | ESG_POLICY_METRICS
    | SOCIAL_METRICS
    | SAFETY_METRICS
    | GOVERNANCE_METRICS
)

# ── KPIs where LOWER value = BETTER performance ───────────────────────────────
LOWER_IS_BETTER: set[str] = (
    ESG_EFFICIENCY_METRICS
    | SAFETY_METRICS
    | GOVERNANCE_METRICS
)

# ── Static industry benchmarks ────────────────────────────────────────────────
# Only for RANKABLE_KPIS — no operational scale metrics here
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
    # Efficiency (Indian energy sector reference values)
    "GHG_tCO2e_perEmployee":          5.0,
    "Energy_GJ_perEmployee":         80.0,
    "Water_KL_perEmployee":         150.0,
    "Waste_MT_perEmployee":           3.0,
}

# ── ESG category scoring thresholds ──────────────────────────────────────────
# GOOD         : company ≥ benchmark × (1 + GOOD_THRESHOLD)    [higher-is-better]
#              : company ≤ benchmark × (1 − GOOD_THRESHOLD)    [lower-is-better]
# BELOW AVERAGE: company ≤ benchmark × (1 − BAD_THRESHOLD)     [higher-is-better]
#              : company ≥ benchmark × (1 + BAD_THRESHOLD)     [lower-is-better]
# AVERAGE      : within ±GOOD_THRESHOLD of benchmark
GOOD_THRESHOLD: float = 0.10   # ±10%
BAD_THRESHOLD:  float = 0.10   # same band, opposite direction

# ── Human-readable KPI labels ─────────────────────────────────────────────────
KPI_LABELS: dict[str, str] = {
    # Efficiency
    "GHG_tCO2e_perEmployee":         "GHG Intensity / Employee (tCO2e)  ↓better",
    "Energy_GJ_perEmployee":         "Energy Intensity / Employee (GJ)  ↓better",
    "Water_KL_perEmployee":          "Water Intensity / Employee (KL)  ↓better",
    "Waste_MT_perEmployee":          "Waste Intensity / Employee (MT)  ↓better",
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
    "TotalGHG_tCO2e":                "Total GHG Scope1+2 (tCO2e)  [scale]",
    "TotalEnergy_GJ":                "Total Energy (GJ)  [scale]",
    "WaterConsumption_KL":           "Water Consumption (KL)  [scale]",
    "WasteGenerated_MT":             "Waste Generated (MT)  [scale]",
    "Turnover_INR":                  "Turnover (INR)  [scale]",
}

# ── Display groups for Section 1 KPI table ───────────────────────────────────
# Groups are labelled with their metric type for clarity
DISPLAY_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Workforce — Headcount [scale]", [
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
    ("Environmental — ESG Efficiency [ranked, lower = better]", [
        ("GHG_tCO2e_perEmployee",  "GHG / Employee (tCO2e)"),
        ("Energy_GJ_perEmployee",  "Energy / Employee (GJ)"),
        ("Water_KL_perEmployee",   "Water / Employee (KL)"),
        ("Waste_MT_perEmployee",   "Waste / Employee (MT)"),
    ]),
    ("Environmental — ESG Policy [ranked, higher = better]", [
        ("RenewableEnergyShare_Pct", "Renewable Energy Share (%)"),
        ("WasteRecoveryRate_Pct",    "Waste Recovery Rate (%)"),
    ]),
    ("Safety [ranked, lower = better]", [
        ("Fatalities",           "Fatalities"),
        ("LTIFR",                "Lost-Time Injury Rate"),
        ("HighConsequenceInjuries","High-Consequence Injuries"),
    ]),
    ("Training [ranked, higher = better]", [
        ("Pct_Emp_HS_Training", "Employees H&S Trained (%)"),
        ("Pct_Wkr_HS_Training", "Workers H&S Trained (%)"),
    ]),
    ("Financial — Operational Scale [context only, not ranked]", [
        ("Turnover_INR",  "Turnover (INR)"),
        ("NetWorth_INR",  "Net Worth (INR)"),
    ]),
    ("Governance [ranked, lower = better]", [
        ("Complaints_Filed",   "Complaints Filed"),
        ("Complaints_Pending", "Complaints Pending"),
    ]),
]
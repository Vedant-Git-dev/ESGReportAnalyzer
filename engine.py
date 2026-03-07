"""
ESG Benchmarking Analytics Engine  v4
======================================
AI-assisted multi-agent system for SEBI BRSR XBRL analysis.

Agents
------
  XMLParsingAgent       → parse real SEBI XBRL files
  MetricExtractionAgent → workforce + environmental KPIs (flexible keyword search)
  NormalizationAgent    → per-employee / intensity metrics
  BenchmarkAgent        → industry & peer benchmarks + KPI rankings
  GapAnalysisAgent      → gap classification (Above / Below / Aligned)
  InsightAgent          → rule-based + Groq LLM narrative insights

LLM
---
  Uses the official `groq` Python library.
  Set GROQ_API_KEY environment variable to enable LLM insights.
  Falls back to rich rule-based insights if the library / key is unavailable.

Usage
-----
  engine = ESGBenchmarkingEngine(mode="industry")
  engine.run(["/path/to/castrol.xml", "/path/to/bpcl.xml"])
  engine.query("Which company leads in gender diversity?")
"""

from __future__ import annotations

import json
import os
import textwrap
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional

import pandas as pd

# ── Groq client (official library, graceful fallback) ────────────────────────
try:
    from groq import Groq as _GroqLib
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqLib = None
    _GROQ_AVAILABLE = False

# ── XBRL namespace constants ─────────────────────────────────────────────────
CAPMKT = "https://www.sebi.gov.in/xbrl/2025-05-31/in-capmkt"
XBRLI  = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"

CY_START, CY_END = "2024-04-01", "2025-03-31"


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — XML Parsing
# ══════════════════════════════════════════════════════════════════════════════
class XMLParsingAgent:
    """
    Input : filepath (str)
    Output: dict with keys
              company_info  → {company_name, cin, industry, …}
              context_map   → {ctx_id: {dim_local: member_local}}
              period_map    → {ctx_id: (start_or_instant, end_or_None)}
              data_elements → [(tag_local, ctx_id, unit_ref, value_str)]
    """

    def parse(self, filepath: str) -> dict:
        print(f"  [XMLParsingAgent] Parsing: {os.path.basename(filepath)}")
        tree = ET.parse(filepath)
        root = tree.getroot()

        context_map   = self._parse_contexts(root)
        period_map    = self._parse_periods(root)
        data_elements = self._parse_data(root)
        company_info  = self._parse_company_info(root)

        name = company_info.get("company_name", os.path.splitext(os.path.basename(filepath))[0])
        print(f"    → {name} | contexts: {len(context_map)} | elements: {len(data_elements)}")
        return dict(
            company_info=company_info,
            context_map=context_map,
            period_map=period_map,
            data_elements=data_elements,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _parse_contexts(self, root) -> dict:
        result = {}
        for ctx in root.findall(f"{{{XBRLI}}}context"):
            ctx_id = ctx.get("id", "")
            dims = {
                self._local(em.get("dimension", "")): self._local((em.text or "").strip())
                for em in ctx.findall(f".//{{{XBRLDI}}}explicitMember")
            }
            result[ctx_id] = dims
        return result

    def _parse_periods(self, root) -> dict:
        result = {}
        for ctx in root.findall(f"{{{XBRLI}}}context"):
            ctx_id = ctx.get("id", "")
            period = ctx.find(f"{{{XBRLI}}}period")
            if period is None:
                result[ctx_id] = (None, None)
                continue
            instant = period.find(f"{{{XBRLI}}}instant")
            if instant is not None:
                result[ctx_id] = (instant.text, None)
            else:
                s = period.find(f"{{{XBRLI}}}startDate")
                e = period.find(f"{{{XBRLI}}}endDate")
                result[ctx_id] = (s.text if s is not None else None,
                                  e.text if e is not None else None)
        return result

    def _parse_data(self, root) -> list:
        elements = []
        for child in root:
            if child.tag.startswith(f"{{{CAPMKT}}}"):
                elements.append((
                    child.tag[len(f"{{{CAPMKT}}}"):]  ,
                    child.get("contextRef", ""),
                    child.get("unitRef", ""),
                    (child.text or "").strip(),
                ))
        return elements

    def _parse_company_info(self, root) -> dict:
        wanted = {
            "NameOfTheCompany":        "company_name",
            "CorporateIdentityNumber": "cin",
            "NameOfIndustry":          "industry",
            "ReportingPeriod":         "reporting_period",
            "TypeOfOrganization":      "org_type",
        }
        info = {}
        for child in root:
            if child.tag.startswith(f"{{{CAPMKT}}}"):
                local = child.tag[len(f"{{{CAPMKT}}}"):]
                if local in wanted and local not in info:
                    info[wanted[local]] = (child.text or "").strip()
        return info

    @staticmethod
    def _local(qname: str) -> str:
        if ":" in qname:
            return qname.split(":", 1)[1]
        if "}" in qname:
            return qname.split("}", 1)[1]
        return qname


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Metric Extraction
# ══════════════════════════════════════════════════════════════════════════════
class MetricExtractionAgent:
    """
    Input : parsed dict from XMLParsingAgent
    Output: dict of raw metric_name → float

    Environmental extraction uses flexible keyword matching so any tag
    containing e.g. "Energy", "Scope", "Water", "Waste" is captured
    even if tag names differ between companies.
    """

    # ── Workforce axes ────────────────────────────────────────────────────────
    EMP_AXIS    = "DetailsOfEmployeesAndWorkersIncludingDifferentlyAbledAxis"
    GENDER_AXIS = "GenderAxis"
    LOC_AXIS    = "LocationAxis"
    GEO_AXIS    = "GeographicalAxis"
    HEADCOUNT   = "NumberOfEmployeesOrWorkersIncludingDifferentlyAbled"

    # ── Environmental tag keyword groups ─────────────────────────────────────
    ENV_KEYWORD_GROUPS: dict[str, list[str]] = {
        # key = metric name we produce,  value = priority-ordered substrings to search
        "TotalEnergy_GJ":             ["TotalEnergyConsumedFromRenewableAndNonRenewable"],
        "RenewableEnergy_GJ":         ["TotalEnergyConsumedFromRenewableSources"],
        "NonRenewableEnergy_GJ":      ["TotalEnergyConsumedFromNonRenewableSources"],
        "TotalElectricity_Renewable": ["TotalElectricityConsumptionFromRenewable"],
        "TotalElectricity_NonRenewable": ["TotalElectricityConsumptionFromNonRenewable"],
        "TotalFuel_Renewable":        ["TotalFuelConsumptionFromRenewable"],
        "TotalFuel_NonRenewable":     ["TotalFuelConsumptionFromNonRenewable"],
        "EnergyIntensity_perRupee":   ["EnergyIntensityPerRupeeOfTurnover"],
        "EnergyIntensity_Physical":   ["EnergyIntensityInTermOfPhysicalOutput"],
        "Scope1_tCO2e":               ["TotalScope1Emissions"],
        "Scope2_tCO2e":               ["TotalScope2Emissions"],
        "GHGIntensity_perRupee":      ["TotalScope1AndScope2EmissionsIntensityPerRupeeOfTurnover"],
        "GHGIntensity_Physical":      ["TotalScope1AndScope2EmissionsIntensityInTermOfPhysicalOutput"],
        "WaterWithdrawal_KL":         ["TotalVolumeOfWaterWithdrawal"],
        "WaterConsumption_KL":        ["TotalVolumeOfWaterConsumption"],
        "WaterDischarged_KL":         ["TotalWaterDischargedInKilolitres"],
        "WaterIntensity_perRupee":    ["WaterIntensityPerRupeeOfTurnover"],
        "WasteGenerated_MT":          ["TotalWasteGenerated"],
        "WasteRecovered_MT":          ["TotalWasteRecovered"],
        "WasteDisposed_MT":           ["TotalWasteDisposed"],
        "WasteDisposed_Incineration": ["WasteDisposedByIncineration"],
        "WasteDisposed_Landfill":     ["WasteDisposedByLandfilling"],
        "WasteIntensity_perRupee":    ["WasteIntensityPerRupeeOfTurnover"],
        "WasteIntensity_Physical":    ["WasteIntensityInTermOfPhysicalOutput"],
        "PlasticWaste_MT":            ["PlasticWaste"],
        "EWaste_MT":                  ["EWaste"],
        "HazardousWaste_MT":          ["OtherHazardousWaste"],
        "NonHazardousWaste_MT":       ["OtherNonHazardousWasteGenerated"],
    }

    def extract(self, parsed: dict) -> dict:
        ctx_map  = parsed["context_map"]
        per_map  = parsed["period_map"]
        elements = parsed["data_elements"]

        # Index: (tag_local, ctx_id) → value_str
        val_idx = {(tag, cid): val for tag, cid, unit, val in elements}

        metrics: dict[str, float] = {}
        metrics.update(self._headcount(ctx_map, per_map, val_idx))
        metrics.update(self._locations(ctx_map, per_map, val_idx))
        metrics.update(self._financials(ctx_map, per_map, val_idx))
        metrics.update(self._environment_flexible(ctx_map, per_map, val_idx, elements))
        metrics.update(self._safety(ctx_map, per_map, val_idx))
        metrics.update(self._training(ctx_map, per_map, val_idx))
        metrics.update(self._complaints(ctx_map, per_map, val_idx))

        print(f"    → Extracted {len(metrics)} raw metrics")
        return metrics

    # ── context helpers ───────────────────────────────────────────────────────

    def _is_cy(self, ctx_id: str, period_map: dict) -> bool:
        s, e = period_map.get(ctx_id, (None, None))
        return bool(s and e and s >= CY_START and e <= CY_END)

    def _find_ctxs(self, ctx_map: dict, period_map: dict,
                   required: dict, prefer_cy: bool = True) -> list:
        """Return context IDs where all required dim→member pairs match."""
        matches = [
            cid for cid, dims in ctx_map.items()
            if all(
                dims.get(dim) == val if val else dim in dims
                for dim, val in required.items()
            )
        ]
        if prefer_cy:
            cy = [c for c in matches if self._is_cy(c, period_map)]
            return cy if cy else matches
        return matches

    def _first_float(self, val_idx: dict, tag: str, ctx_ids: list) -> Optional[float]:
        for cid in ctx_ids:
            v = val_idx.get((tag, cid))
            if v:
                try:
                    return float(v)
                except ValueError:
                    pass
        return None

    def _simple_ctxs(self, ctx_map: dict, period_map: dict) -> list:
        """Contexts with no dimensions, preferring current year."""
        all_s = [c for c, d in ctx_map.items() if not d]
        cy    = [c for c in all_s if self._is_cy(c, period_map)]
        return cy if cy else all_s

    # ── workforce ─────────────────────────────────────────────────────────────

    def _headcount(self, ctx_map, period_map, val_idx) -> dict:
        m: dict[str, float] = {}
        EA, GA = self.EMP_AXIS, self.GENDER_AXIS
        combos = [
            ("perm_emp_male",          "PermanentEmployeesMember",          "MaleMember"),
            ("perm_emp_female",        "PermanentEmployeesMember",          "FemaleMember"),
            ("perm_emp_other",         "PermanentEmployeesMember",          "OtherGenderMember"),
            ("perm_emp_total",         "PermanentEmployeesMember",          "GenderMember"),
            ("nonperm_emp_male",       "OtherThanPermanentEmployeesMember", "MaleMember"),
            ("nonperm_emp_female",     "OtherThanPermanentEmployeesMember", "FemaleMember"),
            ("nonperm_emp_total",      "OtherThanPermanentEmployeesMember", "GenderMember"),
            ("total_emp_male",         "EmployeesMember",                   "MaleMember"),
            ("total_emp_female",       "EmployeesMember",                   "FemaleMember"),
            ("total_emp_total",        "EmployeesMember",                   "GenderMember"),
            ("perm_wkr_male",          "PermanentWorkersMember",            "MaleMember"),
            ("perm_wkr_female",        "PermanentWorkersMember",            "FemaleMember"),
            ("perm_wkr_total",         "PermanentWorkersMember",            "GenderMember"),
            ("contract_wkr_male",      "OtherThanPermanentWorkersMember",   "MaleMember"),
            ("contract_wkr_female",    "OtherThanPermanentWorkersMember",   "FemaleMember"),
            ("contract_wkr_total",     "OtherThanPermanentWorkersMember",   "GenderMember"),
            ("total_wkr_male",         "WorkersMember",                     "MaleMember"),
            ("total_wkr_female",       "WorkersMember",                     "FemaleMember"),
            ("total_wkr_total",        "WorkersMember",                     "GenderMember"),
        ]
        for name, emp_mem, gender_mem in combos:
            ctxs = self._find_ctxs(ctx_map, period_map, {EA: emp_mem, GA: gender_mem})
            v = self._first_float(val_idx, self.HEADCOUNT, ctxs)
            if v is not None:
                m[name] = v
        return m

    def _locations(self, ctx_map, period_map, val_idx) -> dict:
        m: dict[str, float] = {}
        for name, loc_mem, geo_mem in [
            ("plants_national",  "PlantMember",  "NationalMember"),
            ("plants_intl",      "PlantMember",  "InternationalMember"),
            ("offices_national", "OfficeMember", "NationalMember"),
            ("offices_intl",     "OfficeMember", "InternationalMember"),
        ]:
            ctxs = self._find_ctxs(ctx_map, period_map,
                                   {self.LOC_AXIS: loc_mem, self.GEO_AXIS: geo_mem})
            v = self._first_float(val_idx, "NumberOfLocations", ctxs)
            if v is not None:
                m[name] = v
        return m

    def _financials(self, ctx_map, period_map, val_idx) -> dict:
        m: dict[str, float] = {}
        sc = self._simple_ctxs(ctx_map, period_map)
        for tag, name in [
            ("Turnover",      "Turnover_INR"),
            ("NetWorth",      "NetWorth_INR"),
            ("PaidUpCapital", "PaidUpCapital_INR"),
        ]:
            v = self._first_float(val_idx, tag, sc)
            if v is not None:
                m[name] = v
        return m

    # ── environment — flexible keyword search ─────────────────────────────────

    def _environment_flexible(self, ctx_map, period_map, val_idx, elements) -> dict:
        """
        For each ENV_KEYWORD_GROUPS entry, scan all tags in the XML for
        the first substring match and extract the value from a no-dimension,
        current-year context.  Falls back to any context if CY not available.
        """
        m: dict[str, float] = {}

        # Build fast lookup: tag_local → list of (cid, value_str)
        tag_to_entries: dict[str, list] = defaultdict(list)
        for tag, cid, unit, val in elements:
            if val:
                tag_to_entries[tag].append((cid, val))

        # All simple (no-dim) contexts, CY preferred
        simple_cy  = [c for c, d in ctx_map.items() if not d and self._is_cy(c, period_map)]
        simple_all = [c for c, d in ctx_map.items() if not d]
        all_ctxs   = list(ctx_map.keys())

        for metric_name, keywords in self.ENV_KEYWORD_GROUPS.items():
            if metric_name in m:
                continue
            for kw in keywords:
                # Find all tags containing this keyword (case-insensitive)
                matching_tags = [t for t in tag_to_entries if kw.lower() in t.lower()]
                if not matching_tags:
                    continue
                # Prefer exact keyword match, then longest match
                matching_tags.sort(key=lambda t: (-(kw.lower() == t.lower()), len(t)))
                for tag in matching_tags:
                    v = self._first_float(val_idx, tag, simple_cy) or \
                        self._first_float(val_idx, tag, simple_all) or \
                        self._first_float(val_idx, tag, all_ctxs)
                    if v is not None:
                        m[metric_name] = v
                        break
                if metric_name in m:
                    break

        return m

    # ── safety ────────────────────────────────────────────────────────────────

    def _safety(self, ctx_map, period_map, val_idx) -> dict:
        m: dict[str, float] = {}
        all_ctxs = list(ctx_map.keys())
        for tag, name in [
            ("NumberOfFatalities",                    "Fatalities"),
            ("NumberOfHighConsequenceInjury",          "HighConsequenceInjuries"),
            ("NumberOfRecordableWorkRelatedInjuries",  "RecordableInjuries"),
            ("LostTimeInjuryFrequencyRate",            "LTIFR"),
        ]:
            v = self._first_float(val_idx, tag, all_ctxs)
            if v is not None:
                m[name] = v
        return m

    # ── training ──────────────────────────────────────────────────────────────

    def _training(self, ctx_map, period_map, val_idx) -> dict:
        m: dict[str, float] = {}
        all_ctxs = list(ctx_map.keys())
        for tag, name in [
            ("NumberOfEmployeesGivenHealthAndSafetyTraining", "Emp_HS_Training"),
            ("NumberOfWorkersGivenHealthAndSafetyTraining",   "Wkr_HS_Training"),
        ]:
            v = self._first_float(val_idx, tag, all_ctxs)
            if v is not None:
                m[name] = v
        return m

    # ── complaints ────────────────────────────────────────────────────────────

    def _complaints(self, ctx_map, period_map, val_idx) -> dict:
        m: dict[str, float] = {}
        all_ctxs = list(ctx_map.keys())
        for tag, name in [
            ("NumberOfComplaintsFiledDuringTheYear",    "Complaints_Filed"),
            ("NumberOfComplaintsPendingResolution",     "Complaints_Pending"),
            ("NumberOfComplaintsResolvedDuringTheYear", "Complaints_Resolved"),
        ]:
            v = self._first_float(val_idx, tag, all_ctxs)
            if v is not None:
                m[name] = v
        return m


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — Normalization
# ══════════════════════════════════════════════════════════════════════════════
class NormalizationAgent:
    """
    Input : raw metrics dict
    Output: extended metrics dict with derived ratios, shares, intensities.

    Per-employee metrics use (perm_emp_total + perm_wkr_total) as denominator,
    falling back to whichever headcount is available.
    """

    def normalize(self, metrics: dict) -> dict:
        m = dict(metrics)

        emp   = m.get("perm_emp_total", 0)
        wkr   = m.get("perm_wkr_total", 0)
        total = emp + wkr or m.get("total_emp_total", 0)
        rev   = m.get("Turnover_INR")

        # ── Gender ratios ──────────────────────────────────────────────────────
        def ratio(num: str, den: str, out: str):
            n, d = m.get(num), m.get(den)
            if n is not None and d and d > 0:
                m[out] = round(n / d * 100, 2)

        ratio("perm_emp_female",     "perm_emp_total",     "Female_Ratio_PermanentEmp")
        ratio("perm_wkr_female",     "perm_wkr_total",     "Female_Ratio_PermanentWorkers")
        ratio("contract_wkr_female", "contract_wkr_total", "Female_Ratio_ContractWorkers")
        ratio("total_emp_female",    "total_emp_total",    "Female_Ratio_AllEmployees")
        ratio("total_wkr_female",    "total_wkr_total",    "Female_Ratio_AllWorkers")
        ratio("nonperm_emp_female",  "nonperm_emp_total",  "Female_Ratio_NonPermEmployees")

        # ── Energy ────────────────────────────────────────────────────────────
        ren = m.get("RenewableEnergy_GJ")
        tot = m.get("TotalEnergy_GJ")
        if ren is not None and tot and tot > 0:
            m["RenewableEnergyShare_Pct"] = round(ren / tot * 100, 2)

        # ── GHG ───────────────────────────────────────────────────────────────
        s1 = m.get("Scope1_tCO2e")
        s2 = m.get("Scope2_tCO2e")
        if s1 is not None and s2 is not None:
            m["TotalGHG_tCO2e"] = round(s1 + s2, 2)

        # ── Waste ─────────────────────────────────────────────────────────────
        rec = m.get("WasteRecovered_MT")
        gen = m.get("WasteGenerated_MT")
        if rec is not None and gen and gen > 0:
            m["WasteRecoveryRate_Pct"] = round(rec / gen * 100, 2)

        # ── Per-employee env intensities ───────────────────────────────────────
        if total > 0:
            for src, out in [
                ("TotalEnergy_GJ",     "Energy_GJ_perEmployee"),
                ("TotalGHG_tCO2e",     "GHG_tCO2e_perEmployee"),
                ("WasteGenerated_MT",  "Waste_MT_perEmployee"),
                ("WaterConsumption_KL","Water_KL_perEmployee"),
            ]:
                if m.get(src) is not None:
                    m[out] = round(m[src] / total, 2)

        # ── Revenue intensity ──────────────────────────────────────────────────
        if rev and rev > 0:
            rev_cr = rev / 1e7   # convert INR to crores
            if emp:
                m["EmpIntensity_per_CrTurnover"] = round(emp / rev_cr, 4)

        # ── Training coverage ──────────────────────────────────────────────────
        if m.get("Emp_HS_Training") and emp and emp > 0:
            m["Pct_Emp_HS_Training"] = round(m["Emp_HS_Training"] / emp * 100, 2)
        if m.get("Wkr_HS_Training") and wkr and wkr > 0:
            m["Pct_Wkr_HS_Training"] = round(m["Wkr_HS_Training"] / wkr * 100, 2)

        print(f"    → {len(m)} metrics after normalization")
        return m


# ══════════════════════════════════════════════════════════════════════════════
# Agent 4 — Benchmark + KPI Ranking
# ══════════════════════════════════════════════════════════════════════════════
class BenchmarkAgent:
    """
    Input : companies dict {name: metrics}, mode, optional focal/peer
    Output: {benchmarks: dict, rankings: DataFrame, mode_info: str}

    Mode 'industry' → benchmark = mean of all companies
    Mode 'peer'     → benchmark = peer company's metrics
    """

    # KPIs where lower = better performance
    LOWER_IS_BETTER: set[str] = {
        "LTIFR", "Fatalities", "HighConsequenceInjuries", "RecordableInjuries",
        "GHG_tCO2e_perEmployee", "Energy_GJ_perEmployee",
        "Waste_MT_perEmployee", "Water_KL_perEmployee",
        "EnergyIntensity_perRupee", "GHGIntensity_perRupee",
        "WasteIntensity_perRupee", "Complaints_Filed", "Complaints_Pending",
    }

    # KPIs eligible for rankings & gap analysis (ratios/rates — not raw counts)
    RANKABLE_KPIS: set[str] = {
        "Female_Ratio_PermanentEmp", "Female_Ratio_PermanentWorkers",
        "Female_Ratio_ContractWorkers", "Female_Ratio_AllEmployees",
        "Female_Ratio_AllWorkers", "Female_Ratio_NonPermEmployees",
        "RenewableEnergyShare_Pct", "WasteRecoveryRate_Pct",
        "Pct_Emp_HS_Training", "Pct_Wkr_HS_Training",
        "LTIFR", "Fatalities", "HighConsequenceInjuries",
        "GHG_tCO2e_perEmployee", "Energy_GJ_perEmployee",
        "Waste_MT_perEmployee", "Water_KL_perEmployee",
        "EnergyIntensity_perRupee", "GHGIntensity_perRupee",
        "WasteIntensity_perRupee",
        "EmpIntensity_per_CrTurnover",
        "Complaints_Filed", "Complaints_Pending",
        # Absolute env metrics — rankable when available for multiple companies
    }

    # Static industry benchmarks (used to supplement computed benchmarks)
    STATIC_BENCHMARKS: dict[str, float] = {
        "Female_Ratio_PermanentEmp":     15.0,
        "Female_Ratio_PermanentWorkers": 10.0,
        "Female_Ratio_ContractWorkers":   5.0,
        "RenewableEnergyShare_Pct":      30.0,
        "WasteRecoveryRate_Pct":         85.0,
        "Pct_Emp_HS_Training":           90.0,
        "Pct_Wkr_HS_Training":           85.0,
        "LTIFR":                          0.3,
        "Fatalities":                     0.0,
    }

    def compute(self, companies: dict, mode: str = "industry",
                focal: Optional[str] = None,
                peer: Optional[str] = None) -> dict:
        print(f"  [BenchmarkAgent] mode={mode}" +
              (f" | focal={focal} | peer={peer}" if mode == "peer" else ""))

        all_kpis = {k for m in companies.values()
                    for k, v in m.items() if isinstance(v, (int, float))}

        # ── benchmarks ────────────────────────────────────────────────────────
        if mode == "industry":
            benchmarks: dict[str, float] = {}
            for kpi in all_kpis:
                vals = [m[kpi] for m in companies.values()
                        if isinstance(m.get(kpi), (int, float))]
                if vals:
                    benchmarks[kpi] = round(sum(vals) / len(vals), 6)
            for k, v in self.STATIC_BENCHMARKS.items():
                benchmarks.setdefault(k, v)
            mode_info = "Industry average benchmark"

        elif mode == "peer":
            if not focal or not peer:
                raise ValueError("focal and peer required for peer mode")
            peer_metrics = companies.get(peer, {})
            benchmarks   = {k: v for k, v in peer_metrics.items()
                            if isinstance(v, (int, float))}
            mode_info = f"Peer benchmark: {focal} vs {peer}"

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # ── KPI rankings (only RANKABLE_KPIS present in ≥2 companies) ─────────
        rows = []
        for kpi in sorted(self.RANKABLE_KPIS):
            entries = [(name, m[kpi]) for name, m in companies.items()
                       if isinstance(m.get(kpi), (int, float))]
            if len(entries) < 2:
                continue
            reverse = kpi not in self.LOWER_IS_BETTER
            ranked  = sorted(entries, key=lambda x: x[1], reverse=reverse)
            for rank, (name, val) in enumerate(ranked, 1):
                rows.append({"KPI": kpi, "Rank": rank, "Company": name,
                             "Value": val, "Higher_is_Better": reverse})

        rankings_df = pd.DataFrame(rows)
        print(f"    → benchmarks: {len(benchmarks)} KPIs | "
              f"rankings: {len(rankings_df)} rows")
        return dict(benchmarks=benchmarks, rankings=rankings_df, mode_info=mode_info)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 5 — Gap Analysis
# ══════════════════════════════════════════════════════════════════════════════
class GapAnalysisAgent:
    """
    Input : companies dict, benchmarks dict
    Output: DataFrame with Company, KPI, Value, Benchmark, Gap, Gap_Pct, Status
    """

    LOWER_IS_BETTER = BenchmarkAgent.LOWER_IS_BETTER
    MEANINGFUL_KPIS = BenchmarkAgent.RANKABLE_KPIS

    def analyze(self, companies: dict, benchmarks: dict) -> pd.DataFrame:
        rows = []
        for company, metrics in companies.items():
            for kpi, benchmark in benchmarks.items():
                if kpi not in self.MEANINGFUL_KPIS:
                    continue
                val = metrics.get(kpi)
                if not isinstance(val, (int, float)):
                    continue
                gap     = round(val - benchmark, 6)
                lower   = kpi in self.LOWER_IS_BETTER
                if abs(gap) < 1e-6:
                    status = "Aligned"
                elif (gap > 0 and not lower) or (gap < 0 and lower):
                    status = "Above"
                else:
                    status = "Below"
                pct_gap = round(gap / benchmark * 100, 1) if benchmark else None
                rows.append(dict(
                    Company=company, KPI=kpi,
                    Value=round(val, 4), Benchmark=round(benchmark, 4),
                    Gap=round(gap, 4), Gap_Pct=pct_gap, Status=status,
                ))
        df = pd.DataFrame(rows)
        print(f"  [GapAnalysisAgent] → {len(df)} gap records")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Agent 6 — Insight Agent  (rule-based + optional Groq LLM)
# ══════════════════════════════════════════════════════════════════════════════
class InsightAgent:
    """
    Input : companies dict, gap_df, rankings_df, mode_info
    Output: structured insight text

    Uses python `groq` library for LLM insights when GROQ_API_KEY is set.
    Always produces complete rule-based insights as well.
    """

    GROQ_MODEL = "llama-3.3-70b-versatile"

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior ESG analyst specialising in Indian energy sector companies.
        You receive structured benchmarking data from SEBI BRSR XBRL filings.

        Rules:
        - Reason only from the numbers provided — never invent data.
        - Write clear, substantive bullet points.
        - Highlight both strengths and specific gaps.
        - Frame recommendations as concrete, actionable steps.
        - Keep the response under 500 words.
    """).strip()

    # ── Groq call ─────────────────────────────────────────────────────────────

    def _call_groq(self, prompt: str, max_tokens: int = 800) -> Optional[str]:
        if not _GROQ_AVAILABLE:
            return None
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None
        try:
            client = _GroqLib(api_key=api_key)
            response = client.chat.completions.create(
                model=self.GROQ_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [InsightAgent] Groq API error: {e}")
            return None

    # ── public methods ─────────────────────────────────────────────────────────

    def generate(self, companies: dict, gap_df: pd.DataFrame,
                 rankings_df: pd.DataFrame, mode_info: str) -> str:
        rule_insights = self._rule_based(companies, gap_df, rankings_df)

        if _GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
            payload = self._build_payload(companies, gap_df, rankings_df, mode_info)
            prompt  = (
                f"Benchmark mode: {mode_info}\n"
                f"Companies: {', '.join(companies.keys())}\n\n"
                f"Benchmarking data:\n{json.dumps(payload, indent=2)}\n\n"
                "Provide:\n"
                "1. KEY INSIGHTS — 3-5 bullets comparing companies\n"
                "2. STRENGTHS — what each company excels at\n"
                "3. IMPROVEMENT AREAS — specific gaps + recommended actions\n"
                "4. ENVIRONMENTAL PERFORMANCE — if env data available\n"
                "5. OVERALL ESG RANKING — brief justification"
            )
            llm_text = self._call_groq(prompt, max_tokens=800)
            if llm_text:
                return "── Groq LLM Insights ──\n\n" + llm_text + \
                       "\n\n── Rule-Based Insights ──\n" + rule_insights
        return rule_insights

    def answer_query(self, query: str, companies: dict,
                     gap_df: pd.DataFrame, rankings_df: pd.DataFrame) -> str:
        if _GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
            payload = self._build_payload(companies, gap_df, rankings_df, "")
            prompt  = (
                f"Benchmarking data:\n{json.dumps(payload, indent=2)}\n\n"
                f"Question: {query}\n\n"
                "Answer concisely using only the data above."
            )
            result = self._call_groq(prompt, max_tokens=400)
            if result:
                return result
        return self._rule_query(query, companies, gap_df, rankings_df)

    # ── rule-based insight generation ─────────────────────────────────────────

    def _rule_based(self, companies: dict, gap_df: pd.DataFrame,
                    rankings_df: pd.DataFrame) -> str:
        lines: list[str] = []
        company_names = list(companies.keys())

        # ── 1. Per-company performance summary ────────────────────────────────
        lines.append("  Performance Summary")
        lines.append("  " + "-" * 56)
        for company in company_names:
            sub    = gap_df[gap_df["Company"] == company]
            above  = sub[sub["Status"] == "Above"]
            below  = sub[sub["Status"] == "Below"]
            lines.append(f"\n  {company}")
            lines.append(f"  KPIs above benchmark: {len(above)}  |  Below: {len(below)}")
            if not above.empty:
                for _, r in above.sort_values("Gap_Pct", ascending=False).head(4).iterrows():
                    lines.append(f"    ✓ {r['KPI']}: {r['Value']:.2f} "
                                 f"(benchmark {r['Benchmark']:.2f}, {r['Gap_Pct']:+.1f}%)")
            if not below.empty:
                for _, r in below.sort_values("Gap_Pct").head(4).iterrows():
                    lines.append(f"    ✗ {r['KPI']}: {r['Value']:.2f} "
                                 f"(benchmark {r['Benchmark']:.2f}, {r['Gap_Pct']:+.1f}%)")

        # ── 2. Industry-wide observations ─────────────────────────────────────
        lines.append("\n  Industry-Wide Observations")
        lines.append("  " + "-" * 56)

        if not gap_df.empty:
            # KPIs where ALL companies are below benchmark
            all_below = (
                gap_df.groupby("KPI")["Status"]
                .apply(lambda s: all(v == "Below" for v in s))
            )
            universal_gaps = all_below[all_below].index.tolist()
            if universal_gaps:
                lines.append("  ⚠  All companies below benchmark:")
                for k in universal_gaps:
                    lines.append(f"       • {k}")
            else:
                lines.append("  ✓  No KPIs where every company lags benchmark")

            # KPIs where ALL are above benchmark
            all_above = (
                gap_df.groupby("KPI")["Status"]
                .apply(lambda s: all(v == "Above" for v in s))
            )
            universal_strengths = all_above[all_above].index.tolist()
            if universal_strengths:
                lines.append("  ★  All companies outperform benchmark:")
                for k in universal_strengths:
                    lines.append(f"       • {k}")

        # ── 3. Strongest and weakest per KPI ──────────────────────────────────
        if not rankings_df.empty:
            lines.append("\n  KPI Leaders & Laggards")
            lines.append("  " + "-" * 56)
            for kpi in sorted(rankings_df["KPI"].unique()):
                sub  = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                best = sub.iloc[0]
                worst = sub.iloc[-1]
                lines.append(
                    f"  {kpi}")
                lines.append(
                    f"    Leader : {best['Company']} ({best['Value']:.3f})")
                if len(sub) > 1:
                    lines.append(
                        f"    Laggard: {worst['Company']} ({worst['Value']:.3f})")

        # ── 4. Environmental highlights ────────────────────────────────────────
        env_keys = ["TotalGHG_tCO2e", "RenewableEnergyShare_Pct",
                    "WasteRecoveryRate_Pct", "GHG_tCO2e_perEmployee",
                    "Energy_GJ_perEmployee"]
        env_data = {c: {k: m[k] for k in env_keys if k in m}
                    for c, m in companies.items()}
        if any(env_data.values()):
            lines.append("\n  Environmental Highlights")
            lines.append("  " + "-" * 56)
            for company, ev in env_data.items():
                if ev:
                    lines.append(f"  {company}")
                    for k, v in ev.items():
                        lines.append(f"    {k}: {v:.2f}")

        return "\n".join(lines)

    # ── rule-based query answering ─────────────────────────────────────────────

    def _rule_query(self, query: str, companies: dict,
                    gap_df: pd.DataFrame, rankings_df: pd.DataFrame) -> str:
        q = query.lower()
        lines: list[str] = []

        if any(k in q for k in ["gender", "diversity", "female", "women"]):
            gender_kpis = [k for k in ["Female_Ratio_PermanentEmp",
                                       "Female_Ratio_PermanentWorkers",
                                       "Female_Ratio_ContractWorkers",
                                       "Female_Ratio_AllEmployees"]
                           if not rankings_df.empty and
                           k in rankings_df["KPI"].values]
            for kpi in gender_kpis:
                sub = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                if not sub.empty:
                    b = sub.iloc[0]
                    lines.append(f"  {kpi} leader: {b['Company']} ({b['Value']:.2f}%)")

        if any(k in q for k in ["lag", "below", "gap", "worst", "weak"]):
            if not gap_df.empty:
                worst = gap_df[gap_df["Status"] == "Below"].sort_values("Gap_Pct")
                for _, r in worst.head(6).iterrows():
                    lines.append(f"  {r['Company']} — {r['KPI']}: "
                                 f"gap={r['Gap']:+.2f} ({r['Gap_Pct']:+.1f}%)")

        if any(k in q for k in ["env", "energy", "emission", "ghg", "carbon", "water", "waste"]):
            for name, m in companies.items():
                env = {k: m[k] for k in ["TotalGHG_tCO2e", "RenewableEnergyShare_Pct",
                                         "WasteRecoveryRate_Pct"] if k in m}
                if env:
                    lines.append(f"  {name}: {env}")

        if any(k in q for k in ["lead", "best", "top", "strongest"]):
            if not rankings_df.empty:
                for kpi in rankings_df["KPI"].unique():
                    sub  = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                    best = sub.iloc[0]
                    lines.append(f"  {kpi}: {best['Company']} ({best['Value']:.3f})")

        return "\n".join(lines) if lines else \
            "  Please review the gap_analysis.csv for detailed data."

    # ── payload builder ───────────────────────────────────────────────────────

    def _build_payload(self, companies: dict, gap_df: pd.DataFrame,
                       rankings_df: pd.DataFrame, mode_info: str) -> dict:
        KEY_METRICS = {
            "perm_emp_total", "Female_Ratio_PermanentEmp",
            "Female_Ratio_PermanentWorkers", "Female_Ratio_ContractWorkers",
            "RenewableEnergyShare_Pct", "TotalGHG_tCO2e",
            "GHG_tCO2e_perEmployee", "Energy_GJ_perEmployee",
            "WasteRecoveryRate_Pct", "Waste_MT_perEmployee",
            "Water_KL_perEmployee", "LTIFR", "Fatalities",
            "Pct_Emp_HS_Training", "Turnover_INR",
        }
        payload: dict = {"mode": mode_info, "companies": {}}
        for company, m in companies.items():
            sub   = gap_df[gap_df["Company"] == company] if not gap_df.empty else pd.DataFrame()
            payload["companies"][company] = {
                "key_metrics":       {k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in m.items() if k in KEY_METRICS},
                "above_benchmark":   sub[sub["Status"] == "Above"]["KPI"].tolist(),
                "below_benchmark":   sub[sub["Status"] == "Below"]["KPI"].tolist(),
            }
        if not rankings_df.empty:
            payload["rankings"] = {
                kpi: [{"company": r["Company"], "value": r["Value"], "rank": r["Rank"]}
                      for _, r in rankings_df[rankings_df["KPI"] == kpi]
                                             .sort_values("Rank").iterrows()]
                for kpi in rankings_df["KPI"].unique()
            }
        return payload


# ══════════════════════════════════════════════════════════════════════════════
# Report Builder  (not a separate agent, called by orchestrator)
# ══════════════════════════════════════════════════════════════════════════════
class ReportBuilder:
    """Assembles the five-section ESG benchmarking report."""

    # KPIs to display in the summary table, grouped by category
    DISPLAY_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
        ("Workforce", [
            ("perm_emp_total",               "Permanent Employees"),
            ("perm_wkr_total",               "Permanent Workers"),
            ("contract_wkr_total",           "Contract Workers"),
            ("total_emp_total",              "Total Employees"),
            ("total_wkr_total",              "Total Workers"),
            ("Female_Ratio_PermanentEmp",    "Female % (Perm Employees)"),
            ("Female_Ratio_PermanentWorkers","Female % (Perm Workers)"),
            ("Female_Ratio_ContractWorkers", "Female % (Contract Workers)"),
            ("Female_Ratio_AllEmployees",    "Female % (All Employees)"),
        ]),
        ("Environment", [
            # ("TotalEnergy_GJ",              "Total Energy (GJ)"),
            # ("RenewableEnergy_GJ",          "Renewable Energy (GJ)"),
            ("RenewableEnergyShare_Pct",    "Renewable Energy Share (%)"),
            # ("TotalGHG_tCO2e",             "Total GHG Scope1+2 (tCO2e)"),
            # ("Scope1_tCO2e",               "Scope 1 (tCO2e)"),
            # ("Scope2_tCO2e",               "Scope 2 (tCO2e)"),
            ("GHG_tCO2e_perEmployee",       "GHG per Employee (tCO2e)"),
            # ("WaterConsumption_KL",         "Water Consumption (KL)"),
            # ("WaterWithdrawal_KL",          "Water Withdrawal (KL)"),
            # ("WasteGenerated_MT",           "Waste Generated (MT)"),
            # ("WasteRecovered_MT",           "Waste Recovered (MT)"),
            # ("WasteRecoveryRate_Pct",       "Waste Recovery Rate (%)"),
            ("Energy_GJ_perEmployee",       "Energy per Employee (GJ)"),
            ("Water_KL_perEmployee",        "Water per Employee (KL)"),
            ("Waste_MT_perEmployee",        "Waste per Employee (MT)"),
        ]),
        ("Safety & Training", [
            ("Fatalities",                  "Fatalities"),
            ("LTIFR",                       "Lost-Time Injury Rate"),
            ("HighConsequenceInjuries",     "High-Consequence Injuries"),
            ("Pct_Emp_HS_Training",         "Employees H&S Trained (%)"),
        ]),
        ("Financial", [
            ("Turnover_INR",                "Turnover (INR)"),
            ("NetWorth_INR",                "Net Worth (INR)"),
            ("EmpIntensity_per_CrTurnover", "Employees per Cr Turnover"),
        ]),
    ]

    def build(self, companies: dict, gap_df: pd.DataFrame,
              rankings_df: pd.DataFrame, benchmark_result: dict,
              insights: str, output_dir: str) -> str:

        os.makedirs(output_dir, exist_ok=True)
        names = list(companies.keys())
        col_w = 30  # column width per company

        def hr(char="━"): return char * (70 + col_w * len(names))

        lines: list[str] = []

        # ══ HEADER ═══════════════════════════════════════════════════════════
        lines += [
            "=" * (70 + col_w * len(names)),
            "  ESG BENCHMARKING REPORT — SEBI BRSR XBRL  |  FY 2024-25",
            f"  Mode    : {benchmark_result['mode_info']}",
            f"  Companies: {' | '.join(names)}",
            "=" * (70 + col_w * len(names)),
            "",
        ]

        # ══ SECTION 1 — Company KPI Summary ══════════════════════════════════
        lines += [hr(), "  SECTION 1 — COMPANY KPI SUMMARY", hr(), ""]
        header = f"  {'Metric':<48}" + "".join(f"{n[:col_w-2]:<{col_w}}" for n in names)
        lines.append(header)
        lines.append("  " + "-" * (48 + col_w * len(names)))

        for group_name, kpi_pairs in self.DISPLAY_GROUPS:
            lines.append(f"\n  [{group_name}]")
            for key, label in kpi_pairs:
                # only show row if at least one company has data
                if not any(companies[c].get(key) is not None for c in names):
                    continue
                row = f"  {label:<48}"
                for c in names:
                    val = companies[c].get(key)
                    if val is None:
                        row += f"{'—':<{col_w}}"
                    elif isinstance(val, float) and val > 1e9:
                        row += f"{'₹{:,.0f}'.format(val):<{col_w}}"
                    elif isinstance(val, float):
                        row += f"{val:<{col_w}.2f}"
                    else:
                        row += f"{int(val):<{col_w},}"
                lines.append(row)
        lines.append("")

        # ══ SECTION 2 — KPI Rankings ══════════════════════════════════════════
        lines += [hr(), "  SECTION 2 — KPI RANKINGS", hr(), ""]
        if rankings_df.empty:
            lines.append("  Rankings require ≥2 companies with overlapping KPIs.")
        else:
            MEDALS = ["🥇", "🥈", "🥉"]
            for kpi in sorted(rankings_df["KPI"].unique()):
                sub = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                hib = sub.iloc[0]["Higher_is_Better"]
                direction = "higher = better" if hib else "lower = better"
                lines.append(f"  {kpi}  ({direction})")
                for _, row in sub.iterrows():
                    rank = int(row["Rank"])
                    medal = MEDALS[rank - 1] if rank <= 3 else f"   {rank}."
                    lines.append(f"    {medal}  {row['Company']:<48} {row['Value']:.4f}")
                lines.append("")

        # ══ SECTION 3 — Gap Analysis ══════════════════════════════════════════
        lines += [hr(), "  SECTION 3 — GAP ANALYSIS", hr(), ""]
        if gap_df.empty:
            lines.append("  No gap data available.")
        else:
            for company in names:
                sub     = gap_df[gap_df["Company"] == company]
                above   = sub[sub["Status"] == "Above"]
                below   = sub[sub["Status"] == "Below"]
                aligned = sub[sub["Status"] == "Aligned"]
                lines.append(f"  {company}")
                lines.append(f"  {'─' * 62}")
                lines.append(
                    f"  Above benchmark: {len(above):2d}  |  "
                    f"Below: {len(below):2d}  |  Aligned: {len(aligned):2d}")
                if not below.empty:
                    lines.append("  Gaps to close:")
                    for _, r in below.sort_values("Gap_Pct").iterrows():
                        lines.append(
                            f"    ✗  {r['KPI']:<44} "
                            f"val={r['Value']:.3f}  "
                            f"ref={r['Benchmark']:.3f}  "
                            f"gap={r['Gap']:+.3f} ({r['Gap_Pct']:+.1f}%)")
                if not above.empty:
                    lines.append("  Outperforming:")
                    for _, r in above.sort_values("Gap_Pct", ascending=False).iterrows():
                        lines.append(
                            f"    ✓  {r['KPI']:<44} "
                            f"val={r['Value']:.3f}  "
                            f"ref={r['Benchmark']:.3f}  "
                            f"gap={r['Gap']:+.3f} ({r['Gap_Pct']:+.1f}%)")
                lines.append("")

        # ══ SECTION 4 — Company Comparison ═══════════════════════════════════
        lines += [hr(), "  SECTION 4 — COMPANY COMPARISON", hr(), ""]
        lines.append(f"  {'  |  '.join(names)}\n")

        # ── KPI leaders
        if not rankings_df.empty:
            lines.append("  Strengths — KPI Leaders")
            lines.append("  " + "─" * 62)
            for kpi in sorted(rankings_df["KPI"].unique()):
                sub  = rankings_df[rankings_df["KPI"] == kpi].sort_values("Rank")
                best = sub.iloc[0]
                lines.append(f"  {kpi:<46} → {best['Company']} ({best['Value']:.3f})")
            lines.append("")

        # ── KPI laggards
        if not gap_df.empty:
            lines.append("  Weaknesses — Companies Below Benchmark")
            lines.append("  " + "─" * 62)
            for company in names:
                below = gap_df[(gap_df["Company"] == company) &
                               (gap_df["Status"] == "Below")]
                if not below.empty:
                    lines.append(f"  {company}")
                    for _, r in below.sort_values("Gap_Pct").iterrows():
                        lines.append(f"    • {r['KPI']}: {r['Gap_Pct']:+.1f}%")
            lines.append("")

        # ── Operational scale differences
        lines.append("  Operational Scale Comparison")
        lines.append("  " + "─" * 62)
        scale_keys = [
            ("perm_emp_total",   "Permanent Employees"),
            ("total_wkr_total",  "Total Workers"),
            ("Turnover_INR",     "Turnover (INR)"),
            ("plants_national",  "National Plants"),
        ]
        for key, label in scale_keys:
            vals = [(c, companies[c][key]) for c in names if key in companies[c]]
            if len(vals) >= 2:
                hi  = max(vals, key=lambda x: x[1])
                lo  = min(vals, key=lambda x: x[1])
                ratio = hi[1] / lo[1] if lo[1] else float("inf")
                lines.append(
                    f"  {label:<32} "
                    f"{hi[0]} is {ratio:.1f}× larger than {lo[0]}")
        lines.append("")

        # ── Environmental comparison (only if env data available)
        env_keys = ["TotalGHG_tCO2e", "RenewableEnergyShare_Pct",
                    "WasteRecoveryRate_Pct", "TotalEnergy_GJ"]
        env_data = {c: {k: companies[c][k] for k in env_keys
                        if k in companies[c]} for c in names}
        if any(env_data.values()):
            lines.append("  Environmental Performance Snapshot")
            lines.append("  " + "─" * 62)
            env_header = f"  {'Metric':<40}" + "".join(f"{c[:col_w-2]:<{col_w}}"
                                                        for c in names)
            lines.append(env_header)
            for key in env_keys:
                if not any(key in env_data[c] for c in names):
                    continue
                row = f"  {key:<40}"
                for c in names:
                    val = env_data[c].get(key)
                    row += f"{val:<{col_w}.2f}" if val is not None else f"{'—':<{col_w}}"
                lines.append(row)
            lines.append("")

        # ══ SECTION 5 — Key ESG Insights ══════════════════════════════════════
        lines += [hr(), "  SECTION 5 — KEY ESG INSIGHTS", hr(), ""]
        lines.append(insights)
        lines.append("")

        # ══ FOOTER ════════════════════════════════════════════════════════════
        lines += ["=" * (70 + col_w * len(names)), "  END OF REPORT",
                  "=" * (70 + col_w * len(names))]

        report_text = "\n".join(lines)

        # ── persist outputs ───────────────────────────────────────────────────
        with open(os.path.join(output_dir, "esg_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)
        gap_df.to_csv(os.path.join(output_dir, "gap_analysis.csv"), index=False)
        rankings_df.to_csv(os.path.join(output_dir, "kpi_rankings.csv"), index=False)
        pd.DataFrame(companies).T.rename_axis("Company").to_csv(
            os.path.join(output_dir, "normalized_metrics.csv"))
        print(f"  [ReportBuilder] Report → {os.path.join(output_dir, 'esg_report.txt')}")
        return report_text


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════════════
class ESGBenchmarkingEngine:
    """
    Orchestrates all agents end-to-end.

    Parameters
    ----------
    mode : 'industry' | 'peer'
    focal, peer : required when mode='peer'
    output_dir  : directory for report & CSVs
    include_static : inject pre-extracted static data for companies
                     not currently uploaded (e.g. BPCL from previous session)
    """

    def __init__(self, mode: str = "industry",
                 focal: Optional[str] = None,
                 peer: Optional[str] = None,
                 output_dir: str = "./outputs/esg_v4",
                 include_static: bool = True):
        self.mode           = mode
        self.focal          = focal
        self.peer           = peer
        self.output_dir     = output_dir
        self.include_static = include_static
        os.makedirs(output_dir, exist_ok=True)

        self.parser       = XMLParsingAgent()
        self.extractor    = MetricExtractionAgent()
        self.normalizer   = NormalizationAgent()
        self.benchmarker  = BenchmarkAgent()
        self.gap_agent    = GapAnalysisAgent()
        self.insight_agent = InsightAgent()
        self.report_builder = ReportBuilder()

        self._companies:   dict  = {}
        self._gap_df:      pd.DataFrame = pd.DataFrame()
        self._rankings_df: pd.DataFrame = pd.DataFrame()

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, file_paths: list[str],
            nl_queries: Optional[list[str]] = None) -> str:
        print("\n" + "=" * 60)
        print("  ESG BENCHMARKING ENGINE v4")
        print("=" * 60)

        # Parse + extract + normalize from XML files
        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"  Skipping missing file: {fp}")
                continue
            parsed  = self.parser.parse(fp)
            name    = parsed["company_info"].get("company_name") or \
                      os.path.splitext(os.path.basename(fp))[0]
            raw     = self.extractor.extract(parsed)
            normed  = self.normalizer.normalize(raw)
            self._companies[name] = normed

        # Inject static data for companies not in current uploads
        # if self.include_static:
        #     for name, data in self.STATIC_COMPANY_DATA.items():
        #         if name not in self._companies:
        #             print(f"  Injecting static data: {name}")
        #             self._companies[name] = data

        if not self._companies:
            print("  No data available. Check file paths.")
            return ""

        # Benchmark + rank
        bench_result      = self.benchmarker.compute(
            self._companies, mode=self.mode, focal=self.focal, peer=self.peer)
        self._gap_df      = self.gap_agent.analyze(
            self._companies, bench_result["benchmarks"])
        self._rankings_df = bench_result["rankings"]

        # Insights
        insights = self.insight_agent.generate(
            self._companies, self._gap_df,
            self._rankings_df, bench_result["mode_info"])

        # Report
        report = self.report_builder.build(
            self._companies, self._gap_df, self._rankings_df,
            bench_result, insights, self.output_dir)
        print(report)

        # Natural language queries
        if nl_queries:
            print("\n" + "═" * 60)
            print("  NATURAL LANGUAGE QUERIES")
            print("═" * 60)
            query_log = []
            for q in nl_queries:
                print(f"\n  Q: {q}")
                answer = self.query(q)
                print(f"  A:\n{answer}")
                query_log.append({"question": q, "answer": answer})
            with open(os.path.join(self.output_dir, "nl_queries.json"), "w") as f:
                json.dump(query_log, f, indent=2)

        return report

    def query(self, question: str) -> str:
        """Answer a natural language question about the benchmarking data."""
        if not self._companies:
            return "No data loaded. Call run() first."
        return self.insight_agent.answer_query(
            question, self._companies, self._gap_df, self._rankings_df)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    FILES = [
        "./sample_data/castrol.xml",
        "./sample_data/bpcl.xml",
        "./sample_data/indianoil.xml",
    ]
    QUERIES = [
        "Which company leads in gender diversity among permanent employees?",
        "Where is each company lagging the most relative to benchmarks?",
        "How does Castrol perform on environmental metrics?",
        "What are the top ESG improvement priorities?",
    ]
    engine = ESGBenchmarkingEngine(
        mode="industry",
        output_dir="./output/",
    )
    engine.run(FILES, nl_queries=QUERIES)
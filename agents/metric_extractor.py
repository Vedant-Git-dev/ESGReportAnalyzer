"""
agents/metric_extractor.py
--------------------------
Agent 2 — MetricExtractionAgent

Extracts workforce, environmental, safety, training, financial and
complaints KPIs from a parsed XBRL document.

Environmental extraction uses flexible keyword matching so tags are
captured even when their exact names differ slightly between companies.

Input : parsed dict from XMLParsingAgent
Output: dict of raw metric_name → float
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from config.constants import CY_START, CY_END


class MetricExtractionAgent:

    # ── Workforce axis names ──────────────────────────────────────────────────
    EMP_AXIS    = "DetailsOfEmployeesAndWorkersIncludingDifferentlyAbledAxis"
    GENDER_AXIS = "GenderAxis"
    LOC_AXIS    = "LocationAxis"
    GEO_AXIS    = "GeographicalAxis"
    HEADCOUNT   = "NumberOfEmployeesOrWorkersIncludingDifferentlyAbled"

    # ── Environmental tag keyword groups ──────────────────────────────────────
    # Each entry: output_metric_name → [priority-ordered keyword substrings]
    ENV_KEYWORD_GROUPS: dict[str, list[str]] = {
        "TotalEnergy_GJ":               ["TotalEnergyConsumedFromRenewableAndNonRenewable"],
        "RenewableEnergy_GJ":           ["TotalEnergyConsumedFromRenewableSources"],
        "NonRenewableEnergy_GJ":        ["TotalEnergyConsumedFromNonRenewableSources"],
        "TotalElectricity_Renewable":   ["TotalElectricityConsumptionFromRenewable"],
        "TotalElectricity_NonRenewable":["TotalElectricityConsumptionFromNonRenewable"],
        "TotalFuel_Renewable":          ["TotalFuelConsumptionFromRenewable"],
        "TotalFuel_NonRenewable":       ["TotalFuelConsumptionFromNonRenewable"],
        "EnergyIntensity_perRupee":     ["EnergyIntensityPerRupeeOfTurnover"],
        "EnergyIntensity_Physical":     ["EnergyIntensityInTermOfPhysicalOutput"],
        "Scope1_tCO2e":                 ["TotalScope1Emissions"],
        "Scope2_tCO2e":                 ["TotalScope2Emissions"],
        "GHGIntensity_perRupee":        ["TotalScope1AndScope2EmissionsIntensityPerRupeeOfTurnover"],
        "GHGIntensity_Physical":        ["TotalScope1AndScope2EmissionsIntensityInTermOfPhysicalOutput"],
        "WaterWithdrawal_KL":           ["TotalVolumeOfWaterWithdrawal"],
        "WaterConsumption_KL":          ["TotalVolumeOfWaterConsumption"],
        "WaterDischarged_KL":           ["TotalWaterDischargedInKilolitres"],
        "WaterIntensity_perRupee":      ["WaterIntensityPerRupeeOfTurnover"],
        "WasteGenerated_MT":            ["TotalWasteGenerated"],
        "WasteRecovered_MT":            ["TotalWasteRecovered"],
        "WasteDisposed_MT":             ["TotalWasteDisposed"],
        "WasteDisposed_Incineration":   ["WasteDisposedByIncineration"],
        "WasteDisposed_Landfill":       ["WasteDisposedByLandfilling"],
        "WasteIntensity_perRupee":      ["WasteIntensityPerRupeeOfTurnover"],
        "WasteIntensity_Physical":      ["WasteIntensityInTermOfPhysicalOutput"],
        "PlasticWaste_MT":              ["PlasticWaste"],
        "EWaste_MT":                    ["EWaste"],
        "HazardousWaste_MT":            ["OtherHazardousWaste"],
        "NonHazardousWaste_MT":         ["OtherNonHazardousWasteGenerated"],
    }

    # ── public entry point ────────────────────────────────────────────────────

    def extract(self, parsed: dict) -> dict:
        ctx_map  = parsed["context_map"]
        per_map  = parsed["period_map"]
        elements = parsed["data_elements"]

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
                   required: dict) -> list:
        matches = [
            cid for cid, dims in ctx_map.items()
            if all(dims.get(dim) == val for dim, val in required.items())
        ]
        cy = [c for c in matches if self._is_cy(c, period_map)]
        return cy if cy else matches

    def _first_float(self, val_idx: dict, tag: str,
                     ctx_ids: list) -> Optional[float]:
        for cid in ctx_ids:
            v = val_idx.get((tag, cid))
            if v:
                try:
                    return float(v)
                except ValueError:
                    pass
        return None

    def _simple_ctxs(self, ctx_map: dict, period_map: dict) -> list:
        """No-dimension contexts, current-year preferred."""
        no_dim = [c for c, d in ctx_map.items() if not d]
        cy = [c for c in no_dim if self._is_cy(c, period_map)]
        return cy if cy else no_dim

    # ── workforce ─────────────────────────────────────────────────────────────

    def _headcount(self, ctx_map, period_map, val_idx) -> dict:
        m: dict[str, float] = {}
        EA, GA = self.EMP_AXIS, self.GENDER_AXIS
        combos = [
            ("perm_emp_male",       "PermanentEmployeesMember",          "MaleMember"),
            ("perm_emp_female",     "PermanentEmployeesMember",          "FemaleMember"),
            ("perm_emp_other",      "PermanentEmployeesMember",          "OtherGenderMember"),
            ("perm_emp_total",      "PermanentEmployeesMember",          "GenderMember"),
            ("nonperm_emp_male",    "OtherThanPermanentEmployeesMember", "MaleMember"),
            ("nonperm_emp_female",  "OtherThanPermanentEmployeesMember", "FemaleMember"),
            ("nonperm_emp_total",   "OtherThanPermanentEmployeesMember", "GenderMember"),
            ("total_emp_male",      "EmployeesMember",                   "MaleMember"),
            ("total_emp_female",    "EmployeesMember",                   "FemaleMember"),
            ("total_emp_total",     "EmployeesMember",                   "GenderMember"),
            ("perm_wkr_male",       "PermanentWorkersMember",            "MaleMember"),
            ("perm_wkr_female",     "PermanentWorkersMember",            "FemaleMember"),
            ("perm_wkr_total",      "PermanentWorkersMember",            "GenderMember"),
            ("contract_wkr_male",   "OtherThanPermanentWorkersMember",   "MaleMember"),
            ("contract_wkr_female", "OtherThanPermanentWorkersMember",   "FemaleMember"),
            ("contract_wkr_total",  "OtherThanPermanentWorkersMember",   "GenderMember"),
            ("total_wkr_male",      "WorkersMember",                     "MaleMember"),
            ("total_wkr_female",    "WorkersMember",                     "FemaleMember"),
            ("total_wkr_total",     "WorkersMember",                     "GenderMember"),
        ]
        for name, emp_mem, gender_mem in combos:
            ctxs = self._find_ctxs(ctx_map, period_map, {EA: emp_mem, GA: gender_mem})
            v = self._first_float(val_idx, self.HEADCOUNT, ctxs)
            if v is not None:
                m[name] = v
        return m

    # ── locations ─────────────────────────────────────────────────────────────

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

    # ── financials ────────────────────────────────────────────────────────────

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

    # ── environment (flexible keyword search) ─────────────────────────────────

    def _environment_flexible(self, ctx_map, period_map, val_idx, elements) -> dict:
        m: dict[str, float] = {}

        tag_to_entries: dict[str, list] = defaultdict(list)
        for tag, cid, unit, val in elements:
            if val:
                tag_to_entries[tag].append((cid, val))

        simple_cy  = [c for c, d in ctx_map.items() if not d and self._is_cy(c, period_map)]
        simple_all = [c for c, d in ctx_map.items() if not d]
        all_ctxs   = list(ctx_map.keys())

        for metric_name, keywords in self.ENV_KEYWORD_GROUPS.items():
            if metric_name in m:
                continue
            for kw in keywords:
                matching = [t for t in tag_to_entries if kw.lower() in t.lower()]
                if not matching:
                    continue
                matching.sort(key=lambda t: (-(kw.lower() == t.lower()), len(t)))
                for tag in matching:
                    v = (self._first_float(val_idx, tag, simple_cy) or
                         self._first_float(val_idx, tag, simple_all) or
                         self._first_float(val_idx, tag, all_ctxs))
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
            ("NumberOfFatalities",                   "Fatalities"),
            ("NumberOfHighConsequenceInjury",         "HighConsequenceInjuries"),
            ("NumberOfRecordableWorkRelatedInjuries", "RecordableInjuries"),
            ("LostTimeInjuryFrequencyRate",           "LTIFR"),
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
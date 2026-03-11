"""
agents/metric_extractor.py
--------------------------
Agent 2 — MetricExtractionAgent

Extracts workforce, environmental, safety, training, financial and
complaints KPIs from a parsed XBRL document.

Changes for unit normalisation (v8)
------------------------------------
• extract() now returns a 2-tuple: (metrics_dict, unit_ref_map)
  - metrics_dict : {metric_name: raw_float}   ← unchanged
  - unit_ref_map : {metric_name: xbrl_unit_ref_string}
    e.g. {"TotalEnergy_GJ": "Gigajoule", "Turnover_INR": "INR", ...}

• The unit_ref is captured from the XBRL data_elements tuple at extraction
  time so the UnitConverter can use it for deterministic conversion without
  re-parsing the XML.

• Environmental extraction uses flexible keyword matching so tags are
  captured even when exact names differ between companies.

Input : parsed dict from XMLParsingAgent
Output: (metrics_dict, unit_ref_map)
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

    def extract(self, parsed: dict) -> tuple[dict, dict]:
        """
        Returns
        -------
        metrics     : {metric_name: raw_float}
        unit_ref_map: {metric_name: xbrl_unit_ref_string}
        """
        ctx_map  = parsed["context_map"]
        per_map  = parsed["period_map"]
        elements = parsed["data_elements"]

        # Build both lookup indices:
        #   val_idx : (tag, ctx_id) → value_str
        #   unit_idx: (tag, ctx_id) → unit_ref
        val_idx:  dict[tuple, str] = {}
        unit_idx: dict[tuple, str] = {}
        for tag, cid, unit, val in elements:
            if val:
                key = (tag, cid)
                if key not in val_idx:
                    val_idx[key]  = val
                    unit_idx[key] = unit

        metrics:      dict[str, float] = {}
        unit_ref_map: dict[str, str]   = {}

        def merge(m: dict, u: dict) -> None:
            metrics.update(m)
            unit_ref_map.update(u)

        merge(*self._headcount(ctx_map, per_map, val_idx, unit_idx))
        merge(*self._locations(ctx_map, per_map, val_idx, unit_idx))
        merge(*self._financials(ctx_map, per_map, val_idx, unit_idx))
        merge(*self._environment_flexible(ctx_map, per_map, val_idx, unit_idx, elements))
        merge(*self._safety(ctx_map, per_map, val_idx, unit_idx))
        merge(*self._training(ctx_map, per_map, val_idx, unit_idx))
        merge(*self._complaints(ctx_map, per_map, val_idx, unit_idx))

        print(f"    → Extracted {len(metrics)} raw metrics "
              f"({sum(1 for u in unit_ref_map.values() if u)} with explicit unit_ref)")
        return metrics, unit_ref_map

    # ── context helpers ───────────────────────────────────────────────────────

    def _is_cy(self, ctx_id: str, period_map: dict) -> bool:
        s, e = period_map.get(ctx_id, (None, None))
        return bool(s and e and s >= CY_START and e <= CY_END)

    def _find_ctxs(self, ctx_map, period_map, required: dict) -> list:
        matches = [
            cid for cid, dims in ctx_map.items()
            if all(dims.get(dim) == val for dim, val in required.items())
        ]
        cy = [c for c in matches if self._is_cy(c, period_map)]
        return cy if cy else matches

    def _first_float_and_unit(
        self,
        val_idx:  dict,
        unit_idx: dict,
        tag:      str,
        ctx_ids:  list,
    ) -> tuple[Optional[float], str]:
        """Return (float_value, unit_ref) for first matching context."""
        for cid in ctx_ids:
            v = val_idx.get((tag, cid))
            if v:
                try:
                    return float(v), unit_idx.get((tag, cid), "")
                except ValueError:
                    pass
        return None, ""

    def _simple_ctxs(self, ctx_map, period_map) -> list:
        no_dim = [c for c, d in ctx_map.items() if not d]
        cy = [c for c in no_dim if self._is_cy(c, period_map)]
        return cy if cy else no_dim

    # ── workforce ─────────────────────────────────────────────────────────────

    def _headcount(self, ctx_map, period_map, val_idx, unit_idx) -> tuple[dict, dict]:
        m: dict[str, float] = {}
        u: dict[str, str]   = {}
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
            v, unit = self._first_float_and_unit(val_idx, unit_idx, self.HEADCOUNT, ctxs)
            if v is not None:
                m[name] = v
                u[name] = unit
        return m, u

    # ── locations ─────────────────────────────────────────────────────────────

    def _locations(self, ctx_map, period_map, val_idx, unit_idx) -> tuple[dict, dict]:
        m: dict[str, float] = {}
        u: dict[str, str]   = {}
        for name, loc_mem, geo_mem in [
            ("plants_national",  "PlantMember",  "NationalMember"),
            ("plants_intl",      "PlantMember",  "InternationalMember"),
            ("offices_national", "OfficeMember", "NationalMember"),
            ("offices_intl",     "OfficeMember", "InternationalMember"),
        ]:
            ctxs = self._find_ctxs(ctx_map, period_map,
                                   {self.LOC_AXIS: loc_mem, self.GEO_AXIS: geo_mem})
            v, unit = self._first_float_and_unit(val_idx, unit_idx, "NumberOfLocations", ctxs)
            if v is not None:
                m[name] = v
                u[name] = unit
        return m, u

    # ── financials ────────────────────────────────────────────────────────────

    def _financials(self, ctx_map, period_map, val_idx, unit_idx) -> tuple[dict, dict]:
        m: dict[str, float] = {}
        u: dict[str, str]   = {}
        sc = self._simple_ctxs(ctx_map, period_map)
        for tag, name in [
            ("Turnover",      "Turnover_INR"),
            ("NetWorth",      "NetWorth_INR"),
            ("PaidUpCapital", "PaidUpCapital_INR"),
        ]:
            v, unit = self._first_float_and_unit(val_idx, unit_idx, tag, sc)
            if v is not None:
                m[name] = v
                u[name] = unit
        return m, u

    # ── environment (flexible keyword search, captures unit_ref) ──────────────

    def _environment_flexible(
        self, ctx_map, period_map, val_idx, unit_idx, elements
    ) -> tuple[dict, dict]:
        m: dict[str, float] = {}
        u: dict[str, str]   = {}

        # Build tag → [(ctx_id, val, unit_ref)] index
        tag_to_entries: dict[str, list] = defaultdict(list)
        for tag, cid, unit, val in elements:
            if val:
                tag_to_entries[tag].append((cid, val, unit))

        simple_cy  = [c for c, d in ctx_map.items() if not d and self._is_cy(c, period_map)]
        simple_all = [c for c, d in ctx_map.items() if not d]
        all_ctxs   = list(ctx_map.keys())

        def _first_from_tag(tag: str, ctx_priority: list) -> tuple[Optional[float], str]:
            """Return (float_value, unit_ref) for the first valid match."""
            entries_for_tag = tag_to_entries.get(tag, [])
            ctx_set = set(ctx_priority)
            for cid, val, unit in entries_for_tag:
                if cid in ctx_set:
                    try:
                        return float(val), unit
                    except ValueError:
                        pass
            return None, ""

        for metric_name, keywords in self.ENV_KEYWORD_GROUPS.items():
            if metric_name in m:
                continue
            for kw in keywords:
                matching = [t for t in tag_to_entries if kw.lower() in t.lower()]
                if not matching:
                    continue
                matching.sort(key=lambda t: (-(kw.lower() == t.lower()), len(t)))
                for tag in matching:
                    for ctx_group in [simple_cy, simple_all, all_ctxs]:
                        v, unit = _first_from_tag(tag, ctx_group)
                        if v is not None:
                            m[metric_name] = v
                            u[metric_name] = unit
                            break
                    if metric_name in m:
                        break
                if metric_name in m:
                    break
        return m, u

    # ── safety ────────────────────────────────────────────────────────────────

    def _safety(self, ctx_map, period_map, val_idx, unit_idx) -> tuple[dict, dict]:
        m: dict[str, float] = {}
        u: dict[str, str]   = {}
        all_ctxs = list(ctx_map.keys())
        for tag, name in [
            ("NumberOfFatalities",                   "Fatalities"),
            ("NumberOfHighConsequenceInjury",         "HighConsequenceInjuries"),
            ("NumberOfRecordableWorkRelatedInjuries", "RecordableInjuries"),
            ("LostTimeInjuryFrequencyRate",           "LTIFR"),
        ]:
            v, unit = self._first_float_and_unit(val_idx, unit_idx, tag, all_ctxs)
            if v is not None:
                m[name] = v
                u[name] = unit
        return m, u

    # ── training ──────────────────────────────────────────────────────────────

    def _training(self, ctx_map, period_map, val_idx, unit_idx) -> tuple[dict, dict]:
        m: dict[str, float] = {}
        u: dict[str, str]   = {}
        all_ctxs = list(ctx_map.keys())
        for tag, name in [
            ("NumberOfEmployeesGivenHealthAndSafetyTraining", "Emp_HS_Training"),
            ("NumberOfWorkersGivenHealthAndSafetyTraining",   "Wkr_HS_Training"),
        ]:
            v, unit = self._first_float_and_unit(val_idx, unit_idx, tag, all_ctxs)
            if v is not None:
                m[name] = v
                u[name] = unit
        return m, u

    # ── complaints ────────────────────────────────────────────────────────────

    def _complaints(self, ctx_map, period_map, val_idx, unit_idx) -> tuple[dict, dict]:
        m: dict[str, float] = {}
        u: dict[str, str]   = {}
        all_ctxs = list(ctx_map.keys())
        for tag, name in [
            ("NumberOfComplaintsFiledDuringTheYear",    "Complaints_Filed"),
            ("NumberOfComplaintsPendingResolution",     "Complaints_Pending"),
            ("NumberOfComplaintsResolvedDuringTheYear", "Complaints_Resolved"),
        ]:
            v, unit = self._first_float_and_unit(val_idx, unit_idx, tag, all_ctxs)
            if v is not None:
                m[name] = v
                u[name] = unit
        return m, u
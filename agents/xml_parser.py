"""
agents/xml_parser.py
--------------------
Agent 1 — XMLParsingAgent

Parses a real SEBI BRSR XBRL XML file into structured Python objects.

Input : filepath (str)
Output: dict with keys
          company_info  → {company_name, cin, industry, …}
          context_map   → {ctx_id: {dim_local: member_local}}
          period_map    → {ctx_id: (start_or_instant, end_or_None)}
          data_elements → [(tag_local, ctx_id, unit_ref, value_str)]
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from config.constants import CAPMKT, XBRLI, XBRLDI


class XMLParsingAgent:

    def parse(self, filepath: str) -> dict:
        print(f"  [XMLParsingAgent] Parsing: {os.path.basename(filepath)}")
        tree = ET.parse(filepath)
        root = tree.getroot()

        context_map   = self._parse_contexts(root)
        period_map    = self._parse_periods(root)
        data_elements = self._parse_data(root)
        company_info  = self._parse_company_info(root)

        name = company_info.get(
            "company_name",
            os.path.splitext(os.path.basename(filepath))[0],
        )
        print(
            f"    → {name} | "
            f"contexts: {len(context_map)} | "
            f"elements: {len(data_elements)}"
        )
        return dict(
            company_info=company_info,
            context_map=context_map,
            period_map=period_map,
            data_elements=data_elements,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _parse_contexts(self, root) -> dict:
        """Returns {ctx_id: {dim_local: member_local}}"""
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
        """Returns {ctx_id: (start_or_instant, end_or_None)}"""
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
                result[ctx_id] = (
                    s.text if s is not None else None,
                    e.text if e is not None else None,
                )
        return result

    def _parse_data(self, root) -> list:
        """Returns [(tag_local, ctx_id, unit_ref, value_str)]"""
        elements = []
        prefix = f"{{{CAPMKT}}}"
        for child in root:
            if child.tag.startswith(prefix):
                elements.append((
                    child.tag[len(prefix):],
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
        info: dict = {}
        prefix = f"{{{CAPMKT}}}"
        for child in root:
            if child.tag.startswith(prefix):
                local = child.tag[len(prefix):]
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
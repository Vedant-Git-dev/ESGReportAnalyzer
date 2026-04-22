"""
api/routers/export.py

Generates a downloadable PDF report from benchmark data.
Verbatim reportlab logic from ui.py _export_pdf_report.
"""
from __future__ import annotations

import io

from fastapi import APIRouter
from fastapi.responses import Response

from api.schemas import ExportPdfRequest, KPI_GROUPS

router = APIRouter(tags=["export"])


@router.post("/export/pdf")
def export_pdf(req: ExportPdfRequest) -> Response:
    """
    Generate and return a PDF report as binary.
    The frontend sends back the comparison data it received from /api/compare.
    """
    pdf_bytes = _build_pdf(req)
    filename = f"ESG_{req.company1}_vs_{req.company2}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _build_pdf(req: ExportPdfRequest) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm, bottomMargin=2*cm)

    BLUE  = colors.HexColor("#3B82F6")
    GREEN = colors.HexColor("#10B981")
    GRAY  = colors.HexColor("#64748B")
    LIGHT = colors.HexColor("#F8F9FB")
    BDR   = colors.HexColor("#E2E8F0")
    BLK   = colors.HexColor("#1A202C")

    ss = getSampleStyleSheet()
    s_title = ParagraphStyle("t",  parent=ss["Title"],   fontSize=22, textColor=BLK, spaceAfter=4, leading=28, fontName="Helvetica-Bold")
    s_sub   = ParagraphStyle("s",  parent=ss["Normal"],  fontSize=11, textColor=GRAY, spaceAfter=14, leading=16)
    s_h2    = ParagraphStyle("h2", parent=ss["Heading2"],fontSize=13, textColor=BLK, spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold")
    s_body  = ParagraphStyle("b",  parent=ss["Normal"],  fontSize=10, textColor=BLK, leading=16, spaceAfter=8)
    s_note  = ParagraphStyle("n",  parent=ss["Normal"],  fontSize=9,  textColor=GRAY, leading=14)

    label_a = f"{req.company1} FY{req.fy1}"
    label_b = f"{req.company2} FY{req.fy2}"

    story = [
        Paragraph("ESG Competitive Intelligence Report", s_title),
        Paragraph(f"{label_a} vs {label_b}", s_sub),
        Paragraph(f"Sector: {req.sector}", s_note),
        HRFlowable(width="100%", thickness=1, color=BDR, spaceAfter=14),
        Paragraph("Methodology", s_h2),
        Paragraph(
            "Environmental & Social KPIs: normalized by annual revenue (INR Crore). "
            "Governance KPIs (complaints): shown as absolute counts. "
            "Percentage KPIs (women %, renewable %): shown as-is.",
            s_body,
        ),
        Spacer(1, 8),
        Paragraph("KPI Comparison", s_h2),
    ]

    tdata = [["Group", "Metric", "Unit", label_a, label_b, "Gap", "Leader"]]
    for comp in req.comparisons:
        meta  = KPI_GROUPS.get(comp.get("kpi_name", ""), {})
        vals  = {e["company_label"]: e["value"] for e in comp.get("entries", [])}
        v0    = vals.get(label_a)
        v1    = vals.get(label_b)
        fmt   = lambda v: f"{v:.2e}" if (v and 0 < v < 0.001) else (f"{v:,.2f}" if v else "N/A")
        tdata.append([
            meta.get("group", ""),
            meta.get("label", comp.get("kpi_name", "")),
            meta.get("ratio_unit", ""),
            fmt(v0), fmt(v1),
            f"{comp.get('pct_gap', 0):.1f}%",
            str(comp.get("winner", "")).split(" FY")[0],
        ])

    tbl = Table(
        tdata,
        colWidths=[2.8*cm, 4.0*cm, 2.2*cm, 2.4*cm, 2.4*cm, 1.5*cm, 2.2*cm],
        repeatRows=1,
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  7),
        ("TOPPADDING",    (0, 0), (-1, 0),  7),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TOPPADDING",    (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, BDR),
        ("ALIGN",         (3, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR",     (6, 1), ( 6, -1), GREEN),
        ("FONTNAME",      (6, 1), ( 6, -1), "Helvetica-Bold"),
    ]))

    story += [tbl, Spacer(1, 16), Paragraph("Summary", s_h2)]
    for para in req.summary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), s_body))

    story += [
        Spacer(1, 12),
        HRFlowable(width="100%", thickness=0.5, color=BDR),
        Spacer(1, 6),
        Paragraph(
            "Generated by ESG Competitive Intelligence Pipeline. "
            "Source: public BRSR, ESG, and Integrated Annual Reports.",
            s_note,
        ),
    ]
    doc.build(story)
    return buf.getvalue()

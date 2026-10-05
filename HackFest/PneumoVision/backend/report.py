"""
PneumoVision Report Generator
Generates clean, standardized research and decision-support analysis reports:
- Print-ready HTML report
- Formatted Plain Text report
- Structured JSON export
Never includes personal patient health information (PHI).
Prominently features research prototype disclaimers.
"""

from datetime import datetime
from typing import Dict, Any


def generate_text_report(analysis_data: Dict[str, Any]) -> str:
    """Generate a clean ASCII/Text format report."""
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    filename = analysis_data.get("filename", "radiograph.dcm")
    finding = analysis_data.get("finding", "Unknown")
    confidence = analysis_data.get("confidence_percent", "0.0%")
    num_det = analysis_data.get("num_detections", 0)
    inf_time = analysis_data.get("inference_time_ms", 0)
    localization = analysis_data.get("localization", "None")
    img_info = analysis_data.get("image", {})
    detections = analysis_data.get("detections", [])

    clf_data = analysis_data.get("classification", {})
    clf_prob = clf_data.get("probability_percent", "N/A")
    clf_verdict = "Positive (Pneumonia)" if clf_data.get("is_positive") else "Negative / Control"

    lines = [
        "=" * 70,
        "PNEUMOVISION — AI-ASSISTED CHEST RADIOGRAPH ANALYSIS REPORT",
        "Decision-Support & Research Prototype",
        "=" * 70,
        f"Generated Timestamp : {now_str}",
        f"Analyzed File       : {filename}",
        f"Image Dimensions    : {img_info.get('width', 'N/A')} x {img_info.get('height', 'N/A')} px",
        f"Modality / Format   : {img_info.get('format', 'Radiograph')}",
        f"Detection Engine    : Faster R-CNN (ResNet50-FPN v2 Backbone)",
        f"Classifier Engine   : DenseNet-121 (pneumonia_final_best.keras)",
        f"Classifier Prob     : {clf_prob} (Verdict: {clf_verdict})",
        f"Inference Latency   : {inf_time} ms",
        "-" * 70,
        "FINDINGS & LOCALIZATION SUMMARY",
        "-" * 70,
        f"Primary Finding     : {finding}",
        f"Maximum Confidence  : {confidence}",
        f"Detected Opacities  : {num_det} region(s)",
        f"Anatomical Zone     : {localization}",
        ""
    ]

    if detections:
        lines.append("DETAILED DETECTED REGIONS (Image Coordinates):")
        lines.append(f"{'Region':<8} | {'Confidence':<12} | {'Zone':<24} | {'Box (x, y, w, h)'}")
        lines.append("-" * 70)
        for i, det in enumerate(detections, 1):
            box = det.get("box", {})
            box_str = f"({box.get('x')}, {box.get('y')}, {box.get('width')}, {box.get('height')})"
            lines.append(f"#{i:<7} | {det.get('confidence_percent'):<12} | {det.get('anatomical_zone', 'N/A'):<24} | {box_str}")
        lines.append("")

    lines.extend([
        "-" * 70,
        "SAFETY & REGULATORY DISCLAIMER:",
        "This system is a research and decision-support prototype and is not a",
        "medical diagnosis. Model predictions must not replace clinical evaluation",
        "by a licensed healthcare provider or radiologist.",
        "=" * 70
    ])

    return "\n".join(lines)


def generate_html_report(analysis_data: Dict[str, Any]) -> str:
    """Generate an elegant, printable HTML medical AI report."""
    now_str = datetime.utcnow().strftime("%B %d, %Y - %H:%M:%S UTC")
    filename = analysis_data.get("filename", "radiograph.dcm")
    finding = analysis_data.get("finding", "Unknown")
    is_positive = "detected" in finding.lower() and "no" not in finding.lower()
    confidence = analysis_data.get("confidence_percent", "0.0%")
    num_det = analysis_data.get("num_detections", 0)
    inf_time = analysis_data.get("inference_time_ms", 0)
    localization = analysis_data.get("localization", "None")
    img_info = analysis_data.get("image", {})
    detections = analysis_data.get("detections", [])

    det_rows = ""
    if detections:
        for idx, det in enumerate(detections, 1):
            box = det.get("box", {})
            det_rows += f"""
            <tr>
                <td style="padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-weight: 600;">Region #{idx}</td>
                <td style="padding: 10px 14px; border-bottom: 1px solid #e2e8f0; color: #0d9488; font-weight: 700;">{det.get('confidence_percent')}</td>
                <td style="padding: 10px 14px; border-bottom: 1px solid #e2e8f0;">{det.get('anatomical_zone', 'N/A')}</td>
                <td style="padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-family: monospace;">X: {box.get('x')}, Y: {box.get('y')}, W: {box.get('width')}, H: {box.get('height')}</td>
            </tr>
            """
    else:
        det_rows = """
        <tr>
            <td colspan="4" style="padding: 14px; text-align: center; color: #64748b; font-style: italic;">No regions exceeded the 0.75 confidence threshold.</td>
        </tr>
        """

    status_badge_bg = "#fef2f2" if is_positive else "#f0fdf4"
    status_badge_color = "#dc2626" if is_positive else "#16a34a"
    status_border = "#fecaca" if is_positive else "#bbf7d0"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PneumoVision Analysis Report — {filename}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0; padding: 40px; background-color: #f8fafc; color: #0f172a; line-height: 1.5;
        }}
        .report-card {{
            max-width: 820px; margin: 0 auto; background: #ffffff; border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06); padding: 36px; border: 1px solid #e2e8f0;
        }}
        .header {{
            display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0d9488; padding-bottom: 20px;
        }}
        .brand {{ font-size: 26px; font-weight: 800; color: #0f172a; }}
        .brand span {{ color: #0d9488; }}
        .badge {{
            display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 700;
            background: {status_badge_bg}; color: {status_badge_color}; border: 1px solid {status_border};
        }}
        .meta-grid {{
            display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin: 24px 0;
            background: #f8fafc; padding: 18px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 14px;
        }}
        .table-wrap {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
        .table-wrap th {{
            background: #f1f5f9; padding: 10px 14px; text-align: left; font-weight: 700; color: #334155; border-bottom: 2px solid #cbd5e1;
        }}
        .disclaimer {{
            margin-top: 30px; padding: 16px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
            color: #92400e; font-size: 12.5px;
        }}
        @media print {{
            body {{ padding: 0; background: #fff; }}
            .report-card {{ box-shadow: none; border: none; padding: 20px; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div>
                <div class="brand">Pneumo<span>Vision</span></div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">AI-Assisted Chest Radiograph Analysis Report</div>
            </div>
            <div style="text-align: right;">
                <button onclick="window.print()" class="no-print" style="background: #0d9488; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600;">Print Report</button>
                <div style="font-size: 12px; color: #94a3b8; margin-top: 6px;">ID: PV-{int(datetime.utcnow().timestamp())}</div>
            </div>
        </div>

        <div style="margin-top: 24px;">
            <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px;">Diagnostic Finding</div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 8px;">
                <div style="font-size: 20px; font-weight: 800; color: #0f172a;">{finding}</div>
                <div class="badge">{confidence} Max Confidence</div>
            </div>
        </div>

        <div class="meta-grid">
            <div><strong>Analyzed File:</strong> {filename}</div>
            <div><strong>Report Date:</strong> {now_str}</div>
            <div><strong>Resolution:</strong> {img_info.get('width', 'N/A')} × {img_info.get('height', 'N/A')} px</div>
            <div><strong>Modality / Format:</strong> {img_info.get('format', 'Radiograph')}</div>
            <div><strong>Localization Model:</strong> Faster R-CNN (ResNet50-FPN v2)</div>
            <div><strong>Classification Model:</strong> DenseNet-121 ({analysis_data.get('classification', {}).get('probability_percent', 'N/A')} probability)</div>
            <div><strong>Inference Latency:</strong> {inf_time} ms</div>
            <div><strong>Anatomical Localization:</strong> {localization}</div>
        </div>

        <div style="font-size: 15px; font-weight: 700; color: #1e293b; margin-top: 24px;">Detected Regions of Interest</div>
        <table class="table-wrap">
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Confidence</th>
                    <th>Anatomical Localization</th>
                    <th>Coordinates (x, y, w, h)</th>
                </tr>
            </thead>
            <tbody>
                {det_rows}
            </tbody>
        </table>

        <div class="disclaimer">
            <strong>RESEARCH & DECISION-SUPPORT PROTOTYPE DISCLAIMER:</strong><br>
            This system is developed strictly for research, decision support, and demonstration purposes. Model predictions do not constitute a definitive medical diagnosis and must never supersede diagnostic evaluation by a licensed physician or radiologist.
        </div>
    </div>
</body>
</html>"""
    return html

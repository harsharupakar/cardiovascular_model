"""
echo_agent.py — Echocardiography report parsing pipeline.

EXTRACTION GUARDRAILS (strictly enforced):
  • Strict Textual Fidelity  — only values explicitly written in the report are extracted.
  • Zero Hallucination       — no inferring, guessing, or calculating.
  • Null for Missing         — if a metric is not in the text at all, the raw extraction
                               returns None for that field.  Defaults are applied
                               AFTER extraction, are clearly labelled, and are NEVER
                               injected silently as if they came from the report.

Extraction priority (per field):
  1. Regex parser  — deterministic pattern matching on PDF text (zero-hallucination).
  2. LLM (Gemini)  — receives FULL report text + images; explicit zero-hallucination
                     instructed in the prompt; uses EchoExtractionRaw (nullable schema).
  3. Per-field clinical default — applied ONLY when both methods return None for a field.
     Every defaulted field is recorded separately in `echo_fields_defaulted`.
"""
import os
import re
import fitz  # PyMuPDF
import base64
import logging
import pickle
import numpy as np
from typing import Optional
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
if not os.getenv("GOOGLE_API_KEY"):
    log.warning("GOOGLE_API_KEY is not set. LLM text/vision extraction will be unavailable.")

# ── Per-field pipeline defaults (applied only for missing fields, clearly logged) ──
_FIELD_DEFAULTS: dict = {
    "LVEF": 55.0,
    "LVEDD": 50.0,
    "IVSd": 9.0,
    "LVPWd": 9.0,
    "WallMotion": 0,
    "MitralRegurgitation": 0,
    "EA_Ratio": 1.2,
    "PASP": 25.0,
    "LAVI": 28.0,
    "AorticValveArea": 3.5,
}

_MR_GRADE_TO_INT: dict = {
    "None": 0, "Trace": 0, "Mild": 1, "Moderate": 2, "Severe": 3
}


# ─── Strict Extraction Schema (nullable — respects zero-hallucination rule) ───

class EchoExtractionRaw(BaseModel):
    """
    Raw extraction output.  Every field is Optional[...] and defaults to None.
    A None value means the metric was NOT found in the report — it must NOT be
    populated with a 'normal' baseline value by the extraction layer.
    """
    LVEF: Optional[float] = Field(
        default=None,
        description=(
            "Left Ventricular Ejection Fraction as a percentage. "
            "Copy the exact number from the report (e.g. 42 from '42%'). "
            "Output null if not mentioned."
        ),
    )
    LVEDD: Optional[float] = Field(
        default=None,
        description=(
            "Left Ventricular End-Diastolic Dimension in mm. "
            "May appear as LVEDD or LVIDd. Convert cm to mm if needed. "
            "Copy the exact measurement value, not a machine setting. "
            "Output null if not mentioned."
        ),
    )
    IVSd: Optional[float] = Field(
        default=None,
        description=(
            "Interventricular Septum thickness in diastole, in mm. "
            "May appear as IVSd. Convert cm to mm if needed. Output null if absent."
        ),
    )
    LVPWd: Optional[float] = Field(
        default=None,
        description=(
            "Left Ventricular Posterior Wall thickness in diastole, in mm. "
            "May appear as LVPWd. Convert cm to mm if needed. Output null if absent."
        ),
    )
    WallMotion_Hypokinesia: Optional[bool] = Field(
        default=None,
        description=(
            "True  = report explicitly mentions hypokinesia, akinesia, or any wall motion abnormality. "
            "False = report explicitly states wall motion is normal. "
            "null  = wall motion is not mentioned at all in the report."
        ),
    )
    MitralRegurgitation_Grade: Optional[str] = Field(
        default=None,
        description=(
            "Exact severity grade of mitral regurgitation as written. "
            "Must be one of: 'None', 'Trace', 'Mild', 'Moderate', 'Severe'. "
            "Output null if mitral regurgitation is not mentioned."
        ),
    )
    EA_Ratio: Optional[float] = Field(
        default=None,
        description=(
            "E/A mitral inflow ratio. "
            "Copy the exact decimal (e.g. 0.8 from 'E/A Ratio: 0.8'). "
            "Output null if not mentioned."
        ),
    )
    PASP: Optional[float] = Field(
        default=None,
        description=(
            "Pulmonary Artery Systolic Pressure in mmHg. "
            "Copy the exact number (e.g. 42 from '42 mmHg'). "
            "Output null if not mentioned."
        ),
    )
    LAVI: Optional[float] = Field(
        default=None,
        description=(
            "Left Atrial Volume Index in mL/m². "
            "Copy the exact number (e.g. 45 from '45 mL/m2'). "
            "Output null if not mentioned."
        ),
    )
    AorticValveArea: Optional[float] = Field(
        default=None,
        description=(
            "Aortic Valve Area in cm². "
            "Copy the exact number (e.g. 1.2 from '1.2 cm2'). "
            "Output null if not mentioned."
        ),
    )


# ─── Downstream Pipeline Schema (non-nullable, uses defaults for missing) ───

class EchoMetrics(BaseModel):
    LVEF: float = Field(default=55.0, description="Left Ventricular Ejection Fraction (%)")
    LVEDD: float = Field(default=50.0, description="Left Ventricular End-Diastolic Dimension (mm)")
    IVSd: float = Field(default=9.0, description="Interventricular Septum thickness in diastole (mm)")
    LVPWd: float = Field(default=9.0, description="Left Ventricular Posterior Wall thickness in diastole (mm)")
    WallMotion: int = Field(default=0, description="0=Normal, 1=Hypokinesia/Akinesia")
    MitralRegurgitation: int = Field(default=0, description="0=None/Trace, 1=Mild, 2=Moderate, 3=Severe")
    EA_Ratio: float = Field(default=1.2, description="E/A mitral inflow ratio (diastolic function)")
    PASP: float = Field(default=25.0, description="Pulmonary Artery Systolic Pressure (mmHg)")
    LAVI: float = Field(default=28.0, description="Left Atrial Volume Index (mL/m²)")
    AorticValveArea: float = Field(default=3.5, description="Aortic Valve Area (cm²)")


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 0 — PDF → raw text + images
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Return all searchable text from a PDF as a single string."""
    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def extract_images_from_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    """
    Render PDF pages into high-fidelity PNG variants for vision extraction.

    Strategy:
      - full-page at high DPI
      - 4 quadrant crops at high DPI (helps tiny chart labels/values)
    """
    images_b64: list[str] = []
    max_images = 12

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # 1) Full page (high DPI)
            full_pix = page.get_pixmap(dpi=300, alpha=False)
            images_b64.append(base64.b64encode(full_pix.tobytes("png")).decode())
            if len(images_b64) >= max_images:
                break

            # 2) Quadrants (zoom into small table/chart text)
            rect = page.rect
            mid_x = rect.x0 + rect.width / 2
            mid_y = rect.y0 + rect.height / 2
            quads = [
                fitz.Rect(rect.x0, rect.y0, mid_x, mid_y),
                fitz.Rect(mid_x, rect.y0, rect.x1, mid_y),
                fitz.Rect(rect.x0, mid_y, mid_x, rect.y1),
                fitz.Rect(mid_x, mid_y, rect.x1, rect.y1),
            ]
            for clip_rect in quads:
                pix = page.get_pixmap(dpi=300, clip=clip_rect, alpha=False)
                images_b64.append(base64.b64encode(pix.tobytes("png")).decode())
                if len(images_b64) >= max_images:
                    break
            if len(images_b64) >= max_images:
                break

    log.info("[echo_agent] Prepared %d image variants for vision extraction", len(images_b64))
    return images_b64


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 — Regex parser  (deterministic, zero-hallucination)
#  Returns EchoExtractionRaw — None for every field not explicitly found.
# ═══════════════════════════════════════════════════════════════════════════

def _first_float(pattern: str, text: str, flags: int = re.IGNORECASE) -> Optional[float]:
    m = re.search(pattern, text, flags)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            pass
    return None


def _extract_dimension_mm(text: str, labels: list[str]) -> Optional[float]:
    """Extract a cardiac dimension/thickness and normalise to mm from cm or mm units."""
    label_group = "|".join(re.escape(label) for label in labels)
    m = re.search(
        rf"(?:{label_group})[^0-9]{{0,20}}(\d{{1,2}}(?:\.\d+)?)\s*(mm|cm)\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "cm":
        value *= 10.0
    return value


def parse_metrics_with_regex(text: str) -> EchoExtractionRaw:
    """
    Deterministic pattern-match extraction.
    Returns EchoExtractionRaw; every unmatched field stays None.
    No defaults are injected here.
    """
    t = text
    tl = t.lower()

    lvef = _first_float(r"(?:\bLVEF\b|\bEF\b|ejection\s+fraction)[^0-9%]{0,20}(\d{1,3}(?:\.\d+)?)\s*%", t)
    if lvef is not None:
        lvef = max(0.0, min(100.0, lvef))

    lvedd = _extract_dimension_mm(t, ["LVEDD", "LVIDd", "LVIDD"])
    if lvedd is not None:
        lvedd = max(20.0, min(100.0, lvedd))

    ivsd = _extract_dimension_mm(t, ["IVSd", "IVSD"])
    if ivsd is not None:
        ivsd = max(3.0, min(30.0, ivsd))

    lvpwd = _extract_dimension_mm(t, ["LVPWd", "LVPWD", "PWd", "LVPW"])
    if lvpwd is not None:
        lvpwd = max(3.0, min(30.0, lvpwd))

    ea = _first_float(r"E[/:]A\s*[Rr]atio[^0-9]{0,20}(\d{1,2}(?:\.\d+)?)", t)
    if ea is None:
        ea = _first_float(r"\bE[/:]A\b[^0-9]{0,10}(\d{1,2}(?:\.\d+)?)", t)
    if ea is not None:
        ea = max(0.1, min(5.0, ea))

    pasp = _first_float(r"PASP[^0-9]{0,20}(\d{1,3}(?:\.\d+)?)\s*mmHg", t)
    if pasp is not None:
        pasp = max(5.0, min(150.0, pasp))

    lavi = _first_float(
        r"(?:LAVI|left\s+atrial\s+volume\s+index)[^0-9]{0,20}(\d{1,3}(?:\.\d+)?)\s*mL", t
    )
    if lavi is not None:
        lavi = max(10.0, min(100.0, lavi))

    ava = _first_float(
        r"(?:aortic\s+valve\s+area|AVA)[^0-9]{0,20}(\d{1,2}(?:\.\d+)?)\s*cm", t
    )
    if ava is not None:
        ava = max(0.1, min(5.0, ava))

    # Wall motion: True=abnormal, False=explicitly normal, None=not mentioned
    wall_motion_hypokinesia: Optional[bool] = None
    if re.search(r"\b(akinesia|akinetic|hypokinesia|hypokinetic|dyskinesia|wall\s+motion\s+abnormali)", tl):
        wall_motion_hypokinesia = True
    elif re.search(r"wall\s+motion[^.]{0,60}normal", tl):
        wall_motion_hypokinesia = False

    # Mitral regurgitation grade string
    mr_grade: Optional[str] = None
    mr_match = re.search(
        r"(none|trace|trivial|mild|moderate|severe)\s+mitral\s+regurgitation"
        r"|mitral\s+regurgitation[^.]{0,40}(none|trace|trivial|mild|moderate|severe)"
        r"|mitral\s+valve[^.]{0,60}(none|trace|trivial|mild|moderate|severe)\s+(?:mitral\s+)?regurgitation",
        tl,
    )
    if mr_match:
        raw_word = next(g for g in mr_match.groups() if g)
        mr_grade = {"none": "None", "trace": "Trace", "trivial": "Trace",
                    "mild": "Mild", "moderate": "Moderate", "severe": "Severe"}.get(raw_word)

    return EchoExtractionRaw(
        LVEF=lvef, LVEDD=lvedd, IVSd=ivsd, LVPWd=lvpwd, WallMotion_Hypokinesia=wall_motion_hypokinesia,
        MitralRegurgitation_Grade=mr_grade, EA_Ratio=ea, PASP=pasp, LAVI=lavi,
        AorticValveArea=ava,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 — LLM extraction (Gemini) — zero-hallucination prompt
#  Uses EchoExtractionRaw (all nullable) so model cannot inject silent defaults.
# ═══════════════════════════════════════════════════════════════════════════

def parse_metrics_with_gemini(report_text: str, images_base64: list[str]) -> EchoExtractionRaw:
    """
    Gemini extraction with strict zero-hallucination guardrails.
    Full report text is in the prompt.  Output schema is nullable (EchoExtractionRaw).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(EchoExtractionRaw)

    directive = (
        "CRITICAL INSTRUCTIONS — READ BEFORE EXTRACTING:\n"
        "\n"
        "1. STRICT TEXTUAL FIDELITY: You MUST ONLY extract values that are explicitly "
        "written in the report text provided below.\n"
        "2. ZERO HALLUCINATION: Do NOT infer, guess, estimate, or calculate any metric. "
        "Do NOT use standard baseline, 'healthy', or 'normal' adult values as defaults "
        "under ANY circumstances.\n"
        "2b. WARNING: Do NOT confuse ultrasound machine configuration settings with clinical metrics. "
        "Values such as '2D XX%', 'C 50', 'FR XXHz', gain %, depth, power, or probe settings are hardware telemetry, NOT LVEF. "
        "Only extract LVEF if the percentage is explicitly labeled as Ejection Fraction, EF, or LVEF.\n"
        "3. HANDLING MISSING DATA: If a specific metric is not explicitly mentioned in "
        "the text, you MUST output null (None) for that field — NEVER a plausible number.\n"
        "4. NUMBER AND UNIT FOCUS: Pay close attention to units (%, mm, mL/m2, cm2, mmHg) "
        "to map the correct number to the correct field.\n"
        "\n"
        "=== FULL REPORT TEXT (authoritative — read every line) ===\n"
        f"{report_text}\n"
        "=== END OF REPORT TEXT ===\n"
        "\n"
        "Extract each metric from the text above:\n"
        "• LVEF                   — find 'LVEF: XX%' or 'EF XX%' or 'ejection fraction XX%'. Ignore machine settings like '2D 56%'. null if absent.\n"
        "• LVEDD                  — find 'LVEDD: XX mm' or 'LVIDd: X.XX cm'. Convert cm to mm. null if absent.\n"
        "• IVSd                   — find 'IVSd: X.XX cm' or 'IVSd: XX mm'. Convert cm to mm. null if absent.\n"
        "• LVPWd                  — find 'LVPWd: X.XX cm' or 'LVPWd: XX mm'. Convert cm to mm. null if absent.\n"
        "• WallMotion_Hypokinesia — true if hypokinesia/akinesia/wall-motion abnormality mentioned, "
        "false if explicitly normal, null if not mentioned at all.\n"
        "• MitralRegurgitation_Grade — exact string 'None','Trace','Mild','Moderate','Severe'. "
        "null if not mentioned.\n"
        "• EA_Ratio               — find 'E/A Ratio: X.X'. null if absent.\n"
        "• PASP                   — find 'PASP: XX mmHg'. null if absent.\n"
        "• LAVI                   — find 'LAVI: XX mL/m2'. null if absent.\n"
        "• AorticValveArea        — find 'Aortic Valve Area: X.X cm2'. null if absent.\n"
        "\n"
        "Return ONLY the JSON object. No markdown. No explanation."
    )

    content: list = [{"type": "text", "text": directive}]
    for img_b64 in images_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })

    return structured_llm.invoke([HumanMessage(content=content)])


def parse_metrics_from_graph_images_with_gemini(images_base64: list[str]) -> EchoExtractionRaw:
    """
    Vision-only extractor for 2D echo graphs/overlays.
    Use when textual extraction is sparse or missing.
    Strict rule: unreadable/uncertain fields must be null.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(EchoExtractionRaw)

    # Process images iteratively to avoid giant multimodal prompts causing all-null output.
    # Merge non-null fields across passes.
    merged = EchoExtractionRaw()

    directive = (
        "You are extracting cardiology measurements from ONE echocardiography image frame.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1) Use ONLY values explicitly visible in this image.\n"
        "2) If not clearly readable, return null for that field.\n"
        "3) Do not infer, estimate, or copy standard normal values.\n"
        "4) WARNING: Do NOT confuse machine settings with physiology. Values like '2D 56%', gain, FR, depth, C/P settings are hardware settings, not LVEF. "
        "Only extract LVEF if the value is explicitly labeled EF/LVEF/Ejection Fraction near the percentage.\n"
        "5) Respect units while mapping fields: %, mm, mmHg, mL/m2, cm2. Convert cm dimensions to mm for LVEDD, IVSd, and LVPWd.\n"
        "\n"
        "Return JSON in EchoExtractionRaw schema only."
    )

    max_images_for_vision = min(len(images_base64), 6)
    for idx in range(max_images_for_vision):
        img_b64 = images_base64[idx]
        content = [
            {"type": "text", "text": directive},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]
        try:
            partial = structured_llm.invoke([HumanMessage(content=content)])
        except Exception:
            continue

        merged = EchoExtractionRaw(
            LVEF=merged.LVEF if merged.LVEF is not None else partial.LVEF,
            LVEDD=merged.LVEDD if merged.LVEDD is not None else partial.LVEDD,
            IVSd=merged.IVSd if merged.IVSd is not None else partial.IVSd,
            LVPWd=merged.LVPWd if merged.LVPWd is not None else partial.LVPWd,
            WallMotion_Hypokinesia=(
                merged.WallMotion_Hypokinesia
                if merged.WallMotion_Hypokinesia is not None
                else partial.WallMotion_Hypokinesia
            ),
            MitralRegurgitation_Grade=(
                merged.MitralRegurgitation_Grade
                if merged.MitralRegurgitation_Grade is not None
                else partial.MitralRegurgitation_Grade
            ),
            EA_Ratio=merged.EA_Ratio if merged.EA_Ratio is not None else partial.EA_Ratio,
            PASP=merged.PASP if merged.PASP is not None else partial.PASP,
            LAVI=merged.LAVI if merged.LAVI is not None else partial.LAVI,
            AorticValveArea=(
                merged.AorticValveArea
                if merged.AorticValveArea is not None
                else partial.AorticValveArea
            ),
        )

        if _count_raw_fields(merged) >= 10:
            break

    return merged


def _count_raw_fields(raw: EchoExtractionRaw) -> int:
    values = [
        raw.LVEF,
        raw.LVEDD,
        raw.IVSd,
        raw.LVPWd,
        raw.WallMotion_Hypokinesia,
        raw.MitralRegurgitation_Grade,
        raw.EA_Ratio,
        raw.PASP,
        raw.LAVI,
        raw.AorticValveArea,
    ]
    return sum(v is not None for v in values)


def _looks_like_ecg_text(report_text: str) -> bool:
    """Heuristic detector for ECG/EKG-style documents."""
    t = (report_text or "").lower()
    ecg_markers = [
        "ecg", "ekg", "12-lead", "lead i", "lead ii", "lead iii",
        "avl", "avr", "avf", "v1", "v2", "v3", "v4", "v5", "v6",
        "mm/mv", "mm/s", "qrs", "st segment", "t wave",
    ]
    hits = sum(1 for marker in ecg_markers if marker in t)
    return hits >= 3


# ═══════════════════════════════════════════════════════════════════════════
#  Reconciler — EchoExtractionRaw → EchoMetrics + field provenance tracking
# ═══════════════════════════════════════════════════════════════════════════

def _reconcile(raw: EchoExtractionRaw) -> tuple[EchoMetrics, list[str], list[str]]:
    """
    Convert nullable EchoExtractionRaw into non-nullable EchoMetrics for the pipeline.
    Returns (EchoMetrics, extracted_fields, defaulted_fields).
    """
    extracted: list[str] = []
    defaulted: list[str] = []

    def _pick(field: str, raw_val, default_val):
        if raw_val is not None:
            extracted.append(field)
            return raw_val
        defaulted.append(field)
        return default_val

    if raw.WallMotion_Hypokinesia is not None:
        wm = 1 if raw.WallMotion_Hypokinesia else 0
        extracted.append("WallMotion")
    else:
        wm = _FIELD_DEFAULTS["WallMotion"]
        defaulted.append("WallMotion")

    if raw.MitralRegurgitation_Grade is not None:
        mr = _MR_GRADE_TO_INT.get(raw.MitralRegurgitation_Grade, 0)
        extracted.append("MitralRegurgitation")
    else:
        mr = _FIELD_DEFAULTS["MitralRegurgitation"]
        defaulted.append("MitralRegurgitation")

    return (
        EchoMetrics(
            LVEF=_pick("LVEF", raw.LVEF, _FIELD_DEFAULTS["LVEF"]),
            LVEDD=_pick("LVEDD", raw.LVEDD, _FIELD_DEFAULTS["LVEDD"]),
            IVSd=_pick("IVSd", raw.IVSd, _FIELD_DEFAULTS["IVSd"]),
            LVPWd=_pick("LVPWd", raw.LVPWd, _FIELD_DEFAULTS["LVPWd"]),
            WallMotion=wm,
            MitralRegurgitation=mr,
            EA_Ratio=_pick("EA_Ratio", raw.EA_Ratio, _FIELD_DEFAULTS["EA_Ratio"]),
            PASP=_pick("PASP", raw.PASP, _FIELD_DEFAULTS["PASP"]),
            LAVI=_pick("LAVI", raw.LAVI, _FIELD_DEFAULTS["LAVI"]),
            AorticValveArea=_pick("AorticValveArea", raw.AorticValveArea, _FIELD_DEFAULTS["AorticValveArea"]),
        ),
        extracted,
        defaulted,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator — combines both stages with per-field reconciliation
# ═══════════════════════════════════════════════════════════════════════════

def extract_echo_metrics(pdf_bytes: bytes) -> tuple[EchoMetrics, str, Optional[str], list[str], list[str]]:
    """
    Full extraction pipeline.
    Returns (EchoMetrics, status, warning, extracted_fields, defaulted_fields).

    Status:
      "llm_success"   — LLM ran; regex fills any fields LLM left null
      "regex_only"    — LLM unavailable; regex used for all found fields
      "defaults_only" — Neither method found data (blank/image-only PDF)
    extracted_fields: field names read from the actual report text.
    defaulted_fields: field names absent from report; clinical default applied.
    """
    report_text = extract_text_from_pdf_bytes(pdf_bytes)
    images_b64 = extract_images_from_pdf_bytes(pdf_bytes)
    log.info("[echo_agent] Extracted %d chars from PDF", len(report_text))

    # Stage 1: regex
    regex_raw = parse_metrics_with_regex(report_text)
    regex_found_count = _count_raw_fields(regex_raw)
    log.info("[echo_agent] Regex extracted %d fields", regex_found_count)

    # Stage 2: LLM
    llm_raw: Optional[EchoExtractionRaw] = None
    llm_warning: Optional[str] = None
    try:
        llm_raw = parse_metrics_with_gemini(report_text, images_b64)
        log.info("[echo_agent] LLM extraction succeeded")
    except Exception as exc:
        llm_warning = f"LLM extraction skipped ({type(exc).__name__}: {exc})"
        log.warning("[echo_agent] %s", llm_warning)

    # Stage 2b: vision-only graph extraction for sparse/no-text reports
    vision_raw: Optional[EchoExtractionRaw] = None
    vision_warning: Optional[str] = None
    need_graph_vision = regex_found_count < 8
    llm_found_count = _count_raw_fields(llm_raw) if llm_raw is not None else 0
    if need_graph_vision or llm_found_count <= 2:
        try:
            vision_raw = parse_metrics_from_graph_images_with_gemini(images_b64)
            log.info("[echo_agent] Vision-only graph extraction succeeded")
        except Exception as exc:
            vision_warning = f"Vision extraction skipped ({type(exc).__name__}: {exc})"
            log.warning("[echo_agent] %s", vision_warning)

    # Stage 3: merge — LLM non-null wins over regex non-null wins over None
    def _merge(llm_val, regex_val, vision_val):
        if llm_val is not None:
            return llm_val
        if regex_val is not None:
            return regex_val
        return vision_val

    if llm_raw is not None or vision_raw is not None:
        merged_raw = EchoExtractionRaw(
            LVEF=_merge(llm_raw.LVEF if llm_raw else None, regex_raw.LVEF, vision_raw.LVEF if vision_raw else None),
            LVEDD=_merge(llm_raw.LVEDD if llm_raw else None, regex_raw.LVEDD, vision_raw.LVEDD if vision_raw else None),
            IVSd=_merge(llm_raw.IVSd if llm_raw else None, regex_raw.IVSd, vision_raw.IVSd if vision_raw else None),
            LVPWd=_merge(llm_raw.LVPWd if llm_raw else None, regex_raw.LVPWd, vision_raw.LVPWd if vision_raw else None),
            WallMotion_Hypokinesia=_merge(
                llm_raw.WallMotion_Hypokinesia if llm_raw else None,
                regex_raw.WallMotion_Hypokinesia,
                vision_raw.WallMotion_Hypokinesia if vision_raw else None,
            ),
            MitralRegurgitation_Grade=_merge(
                llm_raw.MitralRegurgitation_Grade if llm_raw else None,
                regex_raw.MitralRegurgitation_Grade,
                vision_raw.MitralRegurgitation_Grade if vision_raw else None,
            ),
            EA_Ratio=_merge(llm_raw.EA_Ratio if llm_raw else None, regex_raw.EA_Ratio, vision_raw.EA_Ratio if vision_raw else None),
            PASP=_merge(llm_raw.PASP if llm_raw else None, regex_raw.PASP, vision_raw.PASP if vision_raw else None),
            LAVI=_merge(llm_raw.LAVI if llm_raw else None, regex_raw.LAVI, vision_raw.LAVI if vision_raw else None),
            AorticValveArea=_merge(
                llm_raw.AorticValveArea if llm_raw else None,
                regex_raw.AorticValveArea,
                vision_raw.AorticValveArea if vision_raw else None,
            ),
        )
        if _count_raw_fields(vision_raw) > 0 if vision_raw is not None else False:
            status = "llm_vision_graph"
        else:
            status = "llm_success"
    else:
        merged_raw = regex_raw
        status = "regex_only" if regex_found_count > 0 else "defaults_only"

    # Stage 4: reconcile → EchoMetrics + provenance lists
    echo_metrics, extracted_fields, defaulted_fields = _reconcile(merged_raw)

    # Stage 5: explicit incompatible modality status for ECG-like inputs.
    if len(extracted_fields) == 0 and _looks_like_ecg_text(report_text):
        status = "incompatible_modality_ecg"
        ecg_warning = (
            "Input appears to be ECG/EKG tracing, not echocardiography measurements. "
            "Echo metrics (LVEF, LVEDD, EA_Ratio, PASP, LAVI, AorticValveArea) cannot be read from ECG lead plots."
        )
        llm_warning = f"{llm_warning}; {ecg_warning}" if llm_warning else ecg_warning

    combined_warning = llm_warning
    if vision_warning:
        combined_warning = f"{combined_warning}; {vision_warning}" if combined_warning else vision_warning

    log.info("[echo_agent] extracted=%s | defaulted=%s | status=%s",
             extracted_fields, defaulted_fields, status)
    return echo_metrics, status, combined_warning, extracted_fields, defaulted_fields


# ═══════════════════════════════════════════════════════════════════════════
#  XGBoost disease risk predictor
# ═══════════════════════════════════════════════════════════════════════════

def predict_echo_risks(metrics: EchoMetrics, model_path: str = "models/echo_xgboost.pkl"):
    """Predict disease probabilities (HeartFailure, CAD, Cardiomyopathy)."""
    if not os.path.exists(model_path):
        return {"HeartFailure": 0.0, "CAD": 0.0, "Cardiomyopathy": 0.0}

    with open(model_path, "rb") as f:
        export = pickle.load(f)

    features = export["features"]
    models_dict = export["models"]
    metrics_dict = metrics.model_dump()
    X = np.array([metrics_dict.get(feat, 0.0) for feat in features]).reshape(1, -1)

    return {
        disease: float(mdl.predict_proba(X)[0, 1])
        for disease, mdl in models_dict.items()
    }

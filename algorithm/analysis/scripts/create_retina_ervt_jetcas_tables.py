#!/usr/bin/env python3
"""Create JETCAS-style error distribution XLSX files for Retina and ERVT."""

from __future__ import annotations

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "analysis" / "results"
METRICS = ("Mean", "Median", "P95", "P99")
MOTIONS = ("Fixation", "Smooth", "Saccade", "Blink")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=("Retina", "ERVT"), default=["Retina", "ERVT"])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--combined", action="store_true")
    return parser.parse_args()


def column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_ref(row_idx: int, col_idx: int) -> str:
    return f"{column_name(col_idx)}{row_idx + 1}"


def xml_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def table_from_distribution(dist_path: Path) -> list[list[object]]:
    dist = pd.read_csv(dist_path)
    rows: list[list[object]] = [
        ["Subject-wise Pixel Error Distribution"] + [""] * 17,
        ["Subject", "Split", "Fixation (Input)", "", "", "", "Smooth (Input)", "", "", "", "Saccade (Input)", "", "", "", "Blink (Input)", "", "", ""],
        ["", "", "Mean", "Median", "P95", "P99", "Mean", "Median", "P95", "P99", "Mean", "Median", "P95", "P99", "Mean", "Median", "P95", "P99"],
    ]

    subjects = sorted(int(x) for x in dist["Subject"].dropna().unique())
    for subject in subjects:
        row: list[object] = [subject, "Test"]
        subject_dist = dist[dist["Subject"].astype(int) == subject]
        for motion in MOTIONS:
            motion_rows = subject_dist[subject_dist["Motion"] == motion]
            if motion_rows.empty:
                row.extend([""] * len(METRICS))
                continue
            src = motion_rows.iloc[0]
            for metric in METRICS:
                value = pd.to_numeric(src.get(metric), errors="coerce")
                row.append("" if pd.isna(value) else f"{float(value):.6f}")
        rows.append(row)
    return rows


def sheet_xml(rows: list[list[object]]) -> str:
    xml_rows = []
    for r_idx, row in enumerate(rows):
        cells = []
        for c_idx, value in enumerate(row):
            ref = cell_ref(r_idx, c_idx)
            if value == "":
                cells.append(f'<c r="{ref}"/>')
            else:
                style = 1 if r_idx < 3 else 0
                cells.append(
                    f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{xml_text(value)}</t></is></c>'
                )
        xml_rows.append(f'<row r="{r_idx + 1}">{"".join(cells)}</row>')

    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
        for i, width in enumerate([10, 10] + [13] * 16)
    )
    merges = (
        '<mergeCells count="5">'
        '<mergeCell ref="A1:R1"/>'
        '<mergeCell ref="C2:F2"/>'
        '<mergeCell ref="G2:J2"/>'
        '<mergeCell ref="K2:N2"/>'
        '<mergeCell ref="O2:R2"/>'
        '</mergeCells>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<cols>{cols}</cols>"
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        f"{merges}"
        "</worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{xml_text(name)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
        for i, name in enumerate(sheet_names)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = [
        f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
        for i in range(sheet_count)
    ]
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rels)}'
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(sheet_count)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(overrides)}'
        "</Types>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment horizontal="center"/></xf></cellXfs>'
        "</styleSheet>"
    )


def write_xlsx(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>",
        )
        z.writestr("xl/workbook.xml", workbook_xml(list(sheets)))
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        z.writestr("xl/styles.xml", styles_xml())
        for i, rows in enumerate(sheets.values()):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", sheet_xml(rows))
        z.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:creator>HBTXR analysis</dc:creator>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
            "</cp:coreProperties>",
        )
        z.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>HBTXR analysis</Application>"
            "</Properties>",
        )


def main() -> int:
    args = parse_args()
    tables: dict[str, list[list[object]]] = {}
    for model in args.models:
        dist_path = args.results_root / model / f"{model}_subject37_48_error_distribution_by_subject_motion.csv"
        table = table_from_distribution(dist_path)
        tables[model] = table
        out_path = args.results_root / model / f"JETCAS_REPLY_TABLES (Error-Distributions)_{model}_subject37_48.xlsx"
        write_xlsx(out_path, {"Error Distribution": table})
        print(out_path)

    if args.combined:
        combined_path = args.results_root / "JETCAS_REPLY_TABLES (Error-Distributions)_Retina_ERVT_subject37_48.xlsx"
        write_xlsx(combined_path, tables)
        print(combined_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

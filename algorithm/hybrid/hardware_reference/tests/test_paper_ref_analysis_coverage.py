from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def load_checker():
    script = Path("scripts/external/check_paper_ref_analysis_coverage.py").resolve()
    spec = importlib.util.spec_from_file_location("check_paper_ref_analysis_coverage", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_analysis(path: Path, *, paper_ref: Path, pdf_name: str, slug: str, missing_text: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    local_text = "not present in `tmp_pdf_text/index.json`" if missing_text else "/tmp/text.txt"
    path.write_text(
        f'''---
title: "paper"
paper_slug: "{slug}"
local_pdf: "{paper_ref / pdf_name}"
local_text_file: "{local_text}"
category: "test"
hgtxr_relevance: "P1"
recommended_priority: "P1"
generated_from: "generator"
---

# Paper

## Source
source

## Task Framing
Phase/domain: P1 / Research Workflow.
Parent skill: `paper-idea-generator`.
Worker skill: `ablation-study-designer`.
XR-64-first.

| Metric | Gate |
|---|---:|
| center | `<16.468481131962367` |
| P10 | `>35.02295998845781` |
| P5 | `>12.133503770828247` |

## Paper Classification
`{pdf_name}`

## Existing-Method Problem
problem

## Proposed Method
method

## Algorithm Or Hardware Architecture
algorithm

## Experiments, Dataset, And Reported Results
evidence

## HGTXR Mapping
mapping

## Ablation Option
ablation

## HGTXR Experiment Option Matrix
| Axis | Direct action | Current status |
|---|---|---|
| Head | none | conditional |
| Loss | none | conditional |
| LR/Schedule | none | conditional |
| Data/Post-process | none | conditional |
| Teacher model | none | conditional |
| Self-supervised distillation | none | conditional |
| Optimizer | none | conditional |
| Hardware/export | none | conditional |

## Rejection Rule
reject

## Current Decision
decision
''',
        encoding="utf-8",
    )


def test_paper_ref_coverage_checker_accepts_complete_set(tmp_path: Path) -> None:
    module = load_checker()
    paper_ref = tmp_path / "paper_ref"
    text_root = paper_ref / "tmp_pdf_text"
    text_root.mkdir(parents=True)
    pdf_names = ["(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf", "i-FlatCam.pdf"]
    for pdf in pdf_names:
        (paper_ref / pdf).write_bytes(b"pdf")
    (text_root / "facet.txt").write_text("text", encoding="utf-8")
    (text_root / "index.json").write_text(
        json.dumps([{"pdf": pdf_names[0], "text_file": "facet.txt"}]),
        encoding="utf-8",
    )
    analysis_root = tmp_path / "analysis" / "papers"
    index_lines = ["# index"]
    for pdf in pdf_names:
        slug = module.slugify(pdf)
        write_analysis(
            analysis_root / slug / "analysis.md",
            paper_ref=paper_ref,
            pdf_name=pdf,
            slug=slug,
            missing_text=pdf == "i-FlatCam.pdf",
        )
        index_lines.append(f"`papers/{slug}/analysis.md`")
    (analysis_root / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    code, report = module.run_check(
        Namespace(project_root=str(tmp_path), paper_ref_root=str(paper_ref), analysis_root=str(analysis_root), format="summary")
    )

    assert code == 0
    assert report["ok"] is True
    assert report["pdf_count"] == 2
    assert report["analysis_count"] == 2
    assert report["missing_text_extractions"] == ["i-FlatCam.pdf"]
    summary = module.format_summary(report, code)
    assert "ok: true" in summary
    assert "missing_text_extractions: 1" in summary


def test_paper_ref_coverage_checker_rejects_missing_section(tmp_path: Path) -> None:
    module = load_checker()
    paper_ref = tmp_path / "paper_ref"
    text_root = paper_ref / "tmp_pdf_text"
    text_root.mkdir(parents=True)
    pdf = "paper.pdf"
    (paper_ref / pdf).write_bytes(b"pdf")
    (text_root / "paper.txt").write_text("text", encoding="utf-8")
    (text_root / "index.json").write_text(json.dumps([{"pdf": pdf, "text_file": "paper.txt"}]), encoding="utf-8")
    analysis_root = tmp_path / "analysis" / "papers"
    slug = module.slugify(pdf)
    write_analysis(analysis_root / slug / "analysis.md", paper_ref=paper_ref, pdf_name=pdf, slug=slug)
    body = (analysis_root / slug / "analysis.md").read_text(encoding="utf-8").replace("## Rejection Rule\n", "")
    (analysis_root / slug / "analysis.md").write_text(body, encoding="utf-8")
    (analysis_root / "index.md").write_text(f"`papers/{slug}/analysis.md`\n", encoding="utf-8")

    code, report = module.run_check(
        Namespace(project_root=str(tmp_path), paper_ref_root=str(paper_ref), analysis_root=str(analysis_root), format="json")
    )

    assert code == 1
    assert report["ok"] is False
    assert any("missing section" in error for error in report["errors"])

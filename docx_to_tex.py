import re
import os
from docx import Document

def clean_text_for_latex(text, is_heading=False):
    if not text:
        return ""

    if is_heading:
        text = text.replace("&", r"\&")
    
    # Clean specific raw unicode symbols that break standard text rendering
    text = text.replace("ℝ¹⁰²⁴", r"$\mathbb{R}^{1024}$")
    text = text.replace("ℝ⁵¹²", r"$\mathbb{R}^{512}$")
    text = text.replace("ℝ⁶", r"$\mathbb{R}^6$")
    text = text.replace("F₁", r"$F_1$")
    
    return text

def docx_to_latex(docx_path, tex_path):
    if not os.path.exists(docx_path):
        print(f"Error: Could not find '{docx_path}' in the workspace.")
        return False

    doc = Document(docx_path)
    latex_lines = []

    # Document Preamble and Settings for Two-Column AAAI/arXiv layout
    preamble = r"""\documentclass[10pt,twocolumn,letterpaper]{article}

% --- Core Packages ---
\usepackage[utf8]{inputenc}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

% --- arXiv-style Hyperref Setup ---
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    citecolor=green!70!black,  
    linkcolor=red!80!black,    
    urlcolor=magenta           
}

% --- Header Block ---
\title{\Large\bfseries Benchmarking Variational Quantum Classifiers Against Classical Machine Learning Baselines on Dimensionality-Reduced Bounding-Box Features}

\author{
    \textbf{Sharath Chandran} \\
    \textit{Independent Researcher, Pittsburgh, PA} \\
    \texttt{\small https://github.com/sharathchandran2001/qml\_yolo\_aaai\_project}
}

\date{}

\begin{document}

\maketitle
"""
    latex_lines.append(preamble)

    in_abstract = False

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect Sections & Headings
        if text.lower().startswith("abstract") or "1. Abstract" in text:
            latex_lines.append(r"\begin{abstract}")
            in_abstract = True
            continue
        elif in_abstract and (text.startswith("2.") or text.lower().startswith("introduction")):
            latex_lines.append(r"\end{abstract}")
            in_abstract = False
            cleaned_heading = clean_text_for_latex(text, is_heading=True)
            latex_lines.append(f"\n\\section{{{cleaned_heading}}}")
            continue

        if re.match(r"^[0-9]+\.\s+[A-Z]", text):
            cleaned_heading = clean_text_for_latex(text, is_heading=True)
            heading_title = re.sub(r"^[0-9]+\.\s*", "", cleaned_heading)
            latex_lines.append(f"\n\\section{{{heading_title}}}")
        elif in_abstract:
            latex_lines.append(clean_text_for_latex(text))
        else:
            cleaned_body = clean_text_for_latex(text)
            latex_lines.append(f"\n{cleaned_body}\n")

    if in_abstract:
        latex_lines.append(r"\end{abstract}")

    closing = r"""
\section*{References}
\begin{thebibliography}{10}
\bibitem{wierichs2022general}
David Wierichs et al.
\newblock General parameter-shift rules for quantum gradients.
\newblock {\em Quantum}, 6:677, 2022.

\bibitem{jocher2023ultralytics}
Glenn Jocher et al.
\newblock Ultralytics YOLOv8.
\newblock {\em GitHub Repository}, 2023.
\end{thebibliography}

\end{document}
"""
    latex_lines.append(closing)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    
    print(f"Successfully converted '{docx_path}' to '{tex_path}'!")
    return True

if __name__ == "__main__":
    # docx_file = "final_research_paper.docx"
    # tex_file = "final_research_paper.tex"
    docx_file = "temp one.docx"
    tex_file = "temp_one_final_research_paper.tex"
    docx_to_latex(docx_file, tex_file)
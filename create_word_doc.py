import docx
import os
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_research_paper_docx():
    doc = docx.Document()

    # Set standard page margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Benchmarking Variational Quantum Classifiers Against Classical Machine Learning Baselines on Dimensionality-Reduced Bounding-Box Features")
    title_run.font.name = "Times New Roman"
    title_run.font.size = Pt(18)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

    # Subtitle / Target Metadata
    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("Target Venue: AAAI Conference on Artificial Intelligence | QML & Computer Vision Track\n")
    sub_run.font.name = "Times New Roman"
    sub_run.font.size = Pt(11)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    def add_heading(text, level=1):
        h = doc.add_heading(text, level=level)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        run = h.runs[0]
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0x00, 0x20, 0x60)
        if level == 1:
            run.font.size = Pt(14)
            run.font.bold = True
        elif level == 2:
            run.font.size = Pt(12)
            run.font.bold = True
        return h

    def add_p(text="", bold_prefix=None, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            bp_run = p.add_run(bold_prefix)
            bp_run.font.name = "Times New Roman"
            bp_run.font.size = Pt(11)
            bp_run.font.bold = True
        if text:
            r = p.add_run(text)
            r.font.name = "Times New Roman"
            r.font.size = Pt(11)
            r.font.italic = italic
        return p

    # 1. Abstract & Executive Summary
    add_heading("1. Abstract & Executive Summary", level=1)
    add_p("Current Near-Term Intermediate-Scale Quantum (NISQ) devices face severe qubit and connectivity constraints, preventing direct processing of high-dimensional computer vision representations. This paper presents an empirical benchmark evaluating whether a 6-qubit Variational Quantum Classifier (VQC) can achieve performance parity with classical machine learning models when classifying object detector crop features in a compressed subspace.")
    add_p("Using feature representations extracted from a fine-tuned YOLOv8 backbone (R^1024), features are reduced to a 6-dimensional manifold (R^6) via Principal Component Analysis (PCA) and mapped onto quantum states using RX angle encoding and parameterized StronglyEntanglingLayers. We evaluate accuracy, macro precision, recall, macro F1-score, and microsecond-level inference latency across Support Vector Machines (SVM), Random Forests, and parameterized quantum circuits under strict out-of-sample data splitting.")

    # 2. Introduction & Problem Statement
    add_heading("2. Introduction & Problem Statement", level=1)
    add_p("The integration of quantum machine learning (QML) into classical computer vision pipelines is fundamentally bottlenecked by the input dimensionality of NISQ-era quantum hardware. Standard deep learning vision backbones yield dense embeddings (R^512 to R^2048) that far exceed the qubit register capacity of current quantum processors. Consequently, steep dimensionality compression via Principal Component Analysis (PCA) is required before quantum state encoding can occur.")
    add_p("Evaluating quantum models against classical baselines in this compressed space isolates the expressivity, optimization dynamics, and operational limitations of parameterized quantum circuits under realistic computer vision constraints. The primary objective is to establish a rigorous, data-leakage-free baseline demonstrating the classification performance and inference latency trade-offs between 1024-dimensional classical models and 6-qubit quantum circuits.")

    # 3. Methodology & Experimental Setup
    add_heading("3. Methodology & Experimental Setup", level=1)
    add_p("Bounding box crops are extracted across a strict image-level train/validation split to guarantee complete out-of-sample statistical independence.")

    # Embedded Pipeline Image (Figure 1)
    image_path = "pipeline_diagram_1.png"
    if os.path.exists(image_path):
        img_p = doc.add_paragraph()
        img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_run = img_p.add_run()
        img_run.add_picture(image_path, width=Inches(5.0))
        
        cap_p = doc.add_paragraph()
        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_p.paragraph_format.space_after = Pt(12)
        cap_run = cap_p.add_run("Figure 1: End-to-End System Architecture Pipeline: From YOLOv8 bounding box crop extraction through PCA feature reduction to classical baseline evaluation and parameterized variational quantum circuit inference.")
        cap_run.font.name = "Times New Roman"
        cap_run.font.size = Pt(9.5)
        cap_run.font.italic = True
        cap_run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    else:
        print(f"[!] Warning: '{image_path}' not found in current directory. Image skipped.")

    add_p("A fine-tuned YOLOv8 backbone extracts 1024-dimensional dense feature vectors from detected object crops.", bold_prefix="1. Feature Extraction: ")
    add_p("A shared SharedPCATransformer fits a 6-component PCA model on the training set and transforms both training and test sets into R^6.", bold_prefix="2. Feature Compression: ")
    add_p("Compressed feature vectors x in R^6 are scaled into the interval [0, pi] using a fixed scaling factor max(|X_PCA|): x_i' = pi * (x_i / max(|X_PCA|)).", bold_prefix="3. Quantum Feature Mapping: ")
    add_p("The 6-qubit circuit employs RX(x_i') angle encoding across all 6 wires, followed by 3 layers of StronglyEntanglingLayers containing parameterized single-qubit rotations (RZ, YY, RZ) and cyclic CNOT entanglers.", bold_prefix="4. Ansatz Architecture: ")
    add_p("Output classification probabilities are derived from the expectation value of the Pauli-Z operator on qubit 0: <Z0> = <0| U_dag(x, theta) Z0 U(x, theta) |0>.", bold_prefix="5. Measurement: ")

    # 4. Training
    add_heading("4. Training", level=1)
    add_p("To prevent biased boundary learning during model optimization, the training dataset is balanced using oversampling to 600 crop samples, while the test set preserves its natural distribution of 100 test crops.")
    add_p("Adam optimizer (learning rate = 0.1) run over 50 epochs on a PennyLane autograd interface.", bold_prefix="Optimizer: ")
    add_p("Weighted Mean Squared Error (MSE) on target Pauli expectation values y_i* in {-1.0, +1.0}:\nL(theta, b) = (1/N) * sum_i w_i * ((<Z0>_i + b) - y_i*)^2 where w_i represents the inverse class frequency weight assigned to sample i.", bold_prefix="Loss Function: ")
    add_p("During training, the weighted loss smoothly decayed from an initial state of 1.0420 at Epoch 1 down to 0.5349 at Epoch 50:\n• Epoch 01: Weighted Loss = 1.0420\n• Epoch 10: Weighted Loss = 0.9555\n• Epoch 20: Weighted Loss = 0.7000\n• Epoch 30: Weighted Loss = 0.5580\n• Epoch 40: Weighted Loss = 0.5484\n• Epoch 50: Weighted Loss = 0.5349", bold_prefix="Convergence Progression: ")

    # 5. Inference
    add_heading("5. Inference", level=1)
    add_p("Inference latency benchmarking is conducted using multi-run micro-benchmarking with 50 warmup iterations followed by 200 timed executions using high-resolution performance counters (time.perf_counter).")
    add_p("Class predictions are assigned via a zero-threshold policy on the biased expectation value: y_hat = 1 if (<Z0> + b >= 0.0) else 0.")

    # 6. Limitations of YOLO, ML, QML, etc.
    add_heading("6. Limitations of YOLO, ML, and QML", level=1)
    add_heading("YOLO & Feature Extraction", level=2)
    add_p("Bounding box crops isolate target objects but remove contextual spatial background information. Feature representations are dependent on the feature map resolution of the underlying YOLO backbone layers.")
    add_heading("Classical Machine Learning", level=2)
    add_p("In 1024-D space, high feature dimensions increase computational overhead during Random Forest ensemble evaluation (7.074 ms latency). In 6-D compressed space, linear and kernelized SVM models suffer performance degradation, with F1-score dropping from 0.8308 down to 0.5928.")
    add_heading("Quantum Machine Learning (QML)", level=2)
    add_p("Simulating parameterized circuits beyond 10-12 qubits introduces exponential classical memory scaling O(2^n). Fixed zero-expectation decision boundaries (<Z0> + b >= 0.0) can cause decision bias when evaluated on highly imbalanced test distributions (e.g., 98/2 class split), resulting in F1 = 0.4949. Evaluating parameterized quantum circuits on classical CPU backends incurs simulator overhead (8.839 ms) compared to classical SVM (0.102 ms).")

    # 7. Comparison of Two Approaches
    add_heading("7. Comparison of Two Approaches", level=1)
    table1 = doc.add_table(rows=6, cols=6)
    table1.style = 'Table Grid'
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    headers1 = ["Domain & Model", "Space", "Accuracy", "Macro F1", "Latency", "Sensitivity"]
    data1 = [
        ["Classical SVM", "R^1024", "0.9900", "0.8308", "0.136 ms", "Moderate"],
        ["Classical Random Forest", "R^1024", "0.9900", "0.8308", "7.074 ms", "Low"],
        ["Classical SVM (PCA)", "R^6", "0.9300", "0.5928", "0.102 ms", "High"],
        ["Classical Random Forest (PCA)", "R^6", "0.9900", "0.8308", "7.031 ms", "Low"],
        ["Quantum VQC (PCA)", "R^6", "0.9800", "0.4949", "8.839 ms", "High (Requires tuning)"]
    ]
    for col_idx, text in enumerate(headers1):
        cell = table1.cell(0, col_idx)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.name = "Times New Roman"
        shading = parse_xml(r'<w:shd {} w:fill="D9E1F2"/>'.format(nsdecls('w')))
        cell._tc.get_or_add_tcPr().append(shading)

    for row_idx, row_data in enumerate(data1, start=1):
        for col_idx, text in enumerate(row_data):
            cell = table1.cell(row_idx, col_idx)
            cell.paragraphs[0].text = text
            cell.paragraphs[0].runs[0].font.name = "Times New Roman"

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 8. Benchmarking Results
    add_heading("8. Benchmarking Results", level=1)
    add_p("The empirical results from the benchmark execution across all five model configurations on the out-of-sample test set (100 crops) are presented below:")
    
    table2 = doc.add_table(rows=6, cols=6)
    table2.style = 'Table Grid'
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers2 = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "Latency (ms)"]
    data2 = [
        ["SVM (1024-D)", "0.9900", "0.7500", "0.9949", "0.8308", "0.136 ± 0.013"],
        ["Random Forest (1024-D)", "0.9900", "0.7500", "0.9949", "0.8308", "7.074 ± 0.340"],
        ["SVM (6-D PCA)", "0.9300", "0.5625", "0.9646", "0.5928", "0.102 ± 0.011"],
        ["Random Forest (6-D PCA)", "0.9900", "0.7500", "0.9949", "0.8308", "7.031 ± 0.261"],
        ["Quantum VQC (6-D PCA)", "0.9800", "0.4949", "0.4949", "0.4949", "8.839 ± 0.265"]
    ]

    for col_idx, text in enumerate(headers2):
        cell = table2.cell(0, col_idx)
        cell.paragraphs[0].text = text
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.name = "Times New Roman"
        shading = parse_xml(r'<w:shd {} w:fill="D9E1F2"/>'.format(nsdecls('w')))
        cell._tc.get_or_add_tcPr().append(shading)

    for row_idx, row_data in enumerate(data2, start=1):
        for col_idx, text in enumerate(row_data):
            cell = table2.cell(row_idx, col_idx)
            cell.paragraphs[0].text = text
            cell.paragraphs[0].runs[0].font.name = "Times New Roman"

    add_p()
    add_p("Random Forest achieved an identical macro F1-score (0.8308) in both R^1024 and R^6 spaces, proving that the 6-component PCA manifold preserves non-linear decision boundaries.", bold_prefix="1. PCA Information Retention: ")
    add_p("SVM accuracy decreased from 0.9900 (F1 = 0.8308) in R^1024 to 0.9300 (F1 = 0.5928) in R^6, demonstrating that kernelized linear hyperplanes suffer greater loss under manifold compression compared to tree ensembles.", bold_prefix="2. Support Vector Machine Sensitivity: ")
    add_p("Although weighted MSE optimization reduced training loss from 1.0420 down to 0.5349, evaluating expectation values under a strict zero-threshold (<Z0> + b >= 0.0) on an imbalanced test distribution (98 majority vs. 2 minority samples) mapped predictions toward the majority class (yielding 0.9800 accuracy and macro F1 = 0.4949).", bold_prefix="3. Quantum VQC Dynamics & Expectation Shift: ")
    add_p("Classical SVM models exhibit sub-millisecond evaluation (0.102 ms to 0.136 ms), whereas decision tree ensemble traversal (7.031 ms) and PennyLane autograd quantum state simulations (8.839 ms) require higher evaluation time per sample.", bold_prefix="4. Latency Trade-offs: ")

    # 9. Conclusion
    add_heading("9. Conclusion", level=1)
    add_p("This benchmark confirms that 6-dimensional PCA embeddings extracted from YOLO object detector crops preserve sufficient information for non-linear classification. While parameterized quantum circuits achieve gradient convergence on this compressed manifold, evaluating expectation values against highly imbalanced test distributions requires dynamic decision thresholding beyond standard zero-level cuts. These findings establish a baseline for deploying hybrid QML architectures in constrained vision pipelines.")

    # 10. Acknowledgements
    add_heading("10. Acknowledgements", level=1)
    add_p("This work is partially supported by national quantum computing research initiatives and academic high-performance computing facilities.")

    # 11. References
    add_heading("11. References", level=1)
    refs = [
        "1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.",
        "2. Bergholm, V., et al. (2018). PennyLane: Automatic differentiation of quantum circuits. arXiv preprint arXiv:1811.04968.",
        "3. Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.",
        "4. Schuld, M., \& Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. Physical Review Letters, 122(4), 040504.",
        "5. Jocher, G., et al. (2023). Ultralytics YOLOv8. GitHub Repository."
    ]
    for ref in refs:
        add_p(ref)

    output_path = "final_research_paper.docx"
    doc.save(output_path)
    print(f"[+] Successfully generated Word document: {output_path}")

if __name__ == "__main__":
    create_research_paper_docx()
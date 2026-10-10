import os

latex_code = r"""\documentclass[10pt,twocolumn,letterpaper]{article}

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
    citecolor=green!70!black,  % Clickable green citation links [10]
    linkcolor=red!80!black,    % Clickable internal links (figures/tables)
    urlcolor=magenta           % Web and repository URLs
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

% --- 1. Abstract ---
\begin{abstract}
Current Near-Term Intermediate-Scale Quantum (NISQ) devices face severe qubit and connectivity constraints, preventing direct processing of high-dimensional computer vision representations. This paper presents an empirical benchmark evaluating whether a 6-qubit Variational Quantum Classifier (VQC) can achieve performance parity with classical machine learning models when classifying object detector crop features in a compressed subspace. Using feature representations extracted from a fine-tuned YOLOv8 backbone ($\mathbb{R}^{1024}$), features are reduced to a 6-dimensional manifold ($\mathbb{R}^6$) via Principal Component Analysis (PCA) and mapped onto quantum states using $RX$ angle encoding and parameterized \texttt{StronglyEntanglingLayers}. We evaluate accuracy, macro precision, recall, macro $F_1$-score, and microsecond-level inference latency across Support Vector Machines (SVM), Random Forests, and parameterized quantum circuits under strict out-of-sample data splitting.
\end{abstract}

% --- 2. Introduction ---
\section{Introduction \& Problem Statement}
Recent advancements in Noisy Intermediate-Scale Quantum (NISQ) technologies have driven significant interest in hybrid quantum-classical algorithms. Among these, variational quantum circuits (VQCs)---wherein parameter-dependent quantum gate sequences are optimized iteratively via classical loss functions---have emerged as a primary paradigm for practical quantum execution~\cite{schuld2019quantum, bergholm2018pennylane}. By delegating high-depth linear operations to classical compute while leveraging quantum state spaces for non-linear feature mappings, hybrid architectures provide a viable pathway for integrating quantum computing into domain-specific pipelines such as computer vision~\cite{havlicek2019supervised}.

However, embedding quantum subroutines within modern computer vision pipelines encounters a fundamental dimensionality bottleneck. High-performance deep convolutional backbones and object detection frameworks (e.g., YOLOv8~\cite{jocher2023ultralytics}) generate dense, high-dimensional visual feature embeddings ($\mathbb{R}^{512}$ to $\mathbb{R}^{2048}$) to capture complex spatial geometries. Direct state preparation of a 1024-dimensional feature vector onto a quantum processor requires either impractically wide qubit arrays or ultra-deep parameterized encoding circuits that exceed the coherence times and gate fidelity thresholds of NISQ devices. Consequently, steep feature space reduction---typically via Principal Component Analysis (PCA)---is necessary to compress the continuous image feature manifold into a low-dimensional register ($\mathbb{R}^6$) suitable for hardware execution.

While feature compression enables the execution of parameterized circuits on current hardware, projecting dense computer vision embeddings into a restricted subspace introduces critical trade-offs regarding information retention, decision boundary expressivity, and gradient optimization dynamics. Evaluating quantum classifiers against classical machine learning models within this compressed subspace isolates the expressivity and operational limitations of parameterized quantum circuits under realistic vision constraints.

To address these challenges, this study establishes a data-leakage-free benchmarking methodology evaluating hybrid quantum vision pipelines against classical baselines. The primary contributions of this work are fourfold:
\begin{enumerate}
    \item \textbf{Strict Out-of-Sample Isolation:} We enforce an image-level train/validation split prior to bounding-box crop extraction and feature normalization, ensuring complete statistical independence across PCA fit transformations and quantum model evaluations.
    \item \textbf{Dual-Path Subspace Benchmarking:} We quantify classification performance (accuracy, macro precision, recall, and macro $F_1$-score) across uncompressed ($\mathbb{R}^{1024}$) and PCA-projected ($\mathbb{R}^6$) feature spaces using Support Vector Machines (SVM), Random Forests, and a 6-qubit Variational Quantum Classifier.
    \item \textbf{Ansatz Optimization \& Expectation Dynamics:} We evaluate the optimization progression of a 3-layer \texttt{StronglyEntanglingLayers} circuit trained via weighted Mean Squared Error (MSE), analyzing the impact of Pauli-$Z$ expectation measurement thresholds ($\langle Z_0 \rangle + b \ge 0.0$) on skewed class distributions.
    \item \textbf{Microbenchmarked Operational Latency:} We measure and compare microsecond-level inference latencies across classical kernel executions, decision tree traversals, and PennyLane quantum state vector simulation backends under standardized execution counters.
\end{enumerate}

% --- 3. Methodology ---
\section{Methodology \& Experimental Setup}
Bounding box crops are extracted across a strict image-level train/validation split to guarantee complete out-of-sample statistical independence.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=0.8cm and 0.8cm,
    box/.style={draw=blue!80, fill=blue!8, rounded corners=4pt, align=center, font=\sffamily\small, inner sep=5pt, minimum width=3.1cm},
    purplebox/.style={draw=purple!80, fill=purple!8, rounded corners=4pt, align=center, font=\sffamily\small, inner sep=5pt, minimum width=3.1cm},
    greenbox/.style={draw=green!60!black, fill=green!8, rounded corners=4pt, align=center, font=\sffamily\small, inner sep=5pt, minimum width=3.1cm},
    arrow/.style={-{Stealth[length=2.5mm]}, thick, draw=gray!80}
]
    % Nodes
    \node (yolo) [box] {YOLOv8 Object Detector \\ \& Crop Extraction \\ \textbf{(1024-D Dense Vector)}};
    \node (pca) [purplebox, below=0.8cm of yolo, xshift=1.6cm] {PCA Compression \\ \textbf{(1024-D $\to$ 6-D)}};
    \node (vqc) [purplebox, below=0.7cm of pca] {6-Qubit VQC Circuit \\ \footnotesize($RX$ Angle Map + $L=3$ \\ \footnotesize Ansatz, 54 Params + \\ \footnotesize Pauli-$Z$ $\langle Z_0 \rangle$ Measurement)};
    \node (class) [box, below=2.0cm of yolo, xshift=-1.6cm] {Classical Classifiers \\ \textbf{(SVM \& Random Forest)}};
    \node (bench) [greenbox, below=0.8cm of vqc, xshift=-1.6cm] {Benchmarking Engine \\ \textbf{(Accuracy, F1, Latency)}};

    % Connectors
    \draw[arrow] (yolo.south) -- ++(0,-0.3) -| node[above, pos=0.25, font=\sffamily\tiny\bfseries] {Path A: Uncompressed} (class.north);
    \draw[arrow] (yolo.south) -- ++(0,-0.3) -| node[above, pos=0.25, font=\sffamily\tiny\bfseries] {Path B: 6-D Projection} (pca.north);
    \draw[arrow] (pca.south) -- (vqc.north);
    \draw[arrow] (vqc.south) |- (bench.east);
    \draw[arrow] (class.south) |- (bench.west);
\end{tikzpicture}
\caption{End-to-End System Architecture Pipeline: From YOLOv8 bounding box crop extraction through PCA feature reduction to classical baseline evaluation and parameterized variational quantum circuit inference.}
\label{fig:pipeline}
\end{figure}

\begin{enumerate}
    \item \textbf{Feature Extraction:} A fine-tuned YOLOv8 backbone~\cite{jocher2023ultralytics} extracts 1024-dimensional dense feature vectors from detected object crops.
    \item \textbf{Feature Compression:} A shared \texttt{SharedPCATransformer} fits a 6-component PCA model on the training set and transforms both training and test sets into $\mathbb{R}^6$.
    \item \textbf{Quantum Feature Mapping:} Compressed feature vectors $x \in \mathbb{R}^6$ are scaled into the interval $[0, \pi]$ using a fixed scaling factor $\max(|X_{\text{PCA}}|)$: $x_i' = \pi \cdot (x_i / \max(|X_{\text{PCA}}|))$.
    \item \textbf{Ansatz Architecture:} The 6-qubit circuit employs $RX(x_i')$ angle encoding across all 6 wires, followed by 3 layers of \texttt{StronglyEntanglingLayers} containing parameterized single-qubit rotations ($RZ$, $RY$, $RZ$) and cyclic CNOT entanglers.
    \item \textbf{Measurement:} Output classification probabilities are derived from the expectation value of the Pauli-$Z$ operator on qubit 0:
    \begin{equation}
        \langle Z_0 \rangle = \langle 0 | U^\dagger(x, \theta) Z_0 U(x, \theta) | 0 \rangle
    \end{equation}
\end{enumerate}

% --- 4. Training ---
\section{Training \& Optimization Dynamics}
To prevent biased boundary learning during model optimization, the training dataset is balanced using oversampling to 600 crop samples, while the test set preserves its natural distribution of 100 test crops.
\begin{itemize}
    \item \textbf{Optimizer:} Adam optimizer ($\text{learning rate} = 0.1$) run over 50 epochs on a PennyLane autograd interface~\cite{bergholm2018pennylane}.
    \item \textbf{Loss Function:} Weighted Mean Squared Error (MSE) on target Pauli expectation values $y_i^* \in \{-1.0, +1.0\}$ via parameter-shift differentiation~\cite{wierichs2022general}:
    \begin{equation}
        \mathcal{L}(\theta, b) = \frac{1}{N} \sum_{i=1}^N w_i \cdot \left( (\langle Z_0 \rangle_i + b) - y_i^* \right)^2
    \end{equation}
    where $w_i$ represents the inverse class frequency weight assigned to sample $i$.
    \item \textbf{Convergence Progression:} During training, the weighted loss smoothly decayed from an initial state of $1.0420$ at Epoch 1 down to $0.5349$ at Epoch 50:
    \begin{itemize}
        \item \textbf{Epoch 01:} Weighted Loss $= 1.0420$
        \item \textbf{Epoch 10:} Weighted Loss $= 0.9555$
        \item \textbf{Epoch 20:} Weighted Loss $= 0.7000$
        \item \textbf{Epoch 30:} Weighted Loss $= 0.5580$
        \item \textbf{Epoch 40:} Weighted Loss $= 0.5484$
        \item \textbf{Epoch 50:} Weighted Loss $= 0.5349$
    \end{itemize}
\end{itemize}

% --- 5. Inference ---
\section{Inference Benchmarking}
Inference latency benchmarking is conducted using multi-run micro-benchmarking with 50 warmup iterations followed by 200 timed executions using high-resolution performance counters (\texttt{time.perf\_counter}).

Class predictions are assigned via a zero-threshold policy on the biased expectation value:
\begin{equation}
    \hat{y} = \begin{cases} 1 & \text{if } (\langle Z_0 \rangle + b \ge 0.0) \\ 0 & \text{otherwise} \end{cases}
\end{equation}

% --- 6. Limitations ---
\section{Limitations of YOLO, ML, and QML}
\paragraph{YOLO \& Feature Extraction} Bounding box crops isolate target objects but remove contextual spatial background information. Feature representations are dependent on the feature map resolution of the underlying YOLO backbone layers~\cite{redmon2018yolo, jocher2023ultralytics}.

\paragraph{Classical Machine Learning} In 1024-D space, high feature dimensions increase computational overhead during Random Forest ensemble evaluation ($7.074\text{ ms}$ latency). In 6-D compressed space, linear and kernelized SVM models suffer performance degradation, with $F_1$-score dropping from $0.8308$ down to $0.5928$.

\paragraph{Quantum Machine Learning (QML)} Simulating parameterized circuits beyond 10--12 qubits introduces exponential classical memory scaling $\mathcal{O}(2^n)$. Fixed zero-expectation decision boundaries ($\langle Z_0 \rangle + b \ge 0.0$) can cause decision bias when evaluated on highly imbalanced test distributions (e.g., 98/2 class split), resulting in $F_1 = 0.4949$. Evaluating parameterized quantum circuits on classical CPU backends incurs simulator overhead ($8.839\text{ ms}$) compared to classical SVM ($0.102\text{ ms}$).

% --- 7. Comparison Table ---
\section{Comparison of Technical Approaches}
Table~\ref{tab:comparison} outlines the architectural and operational differences across model domains.

\begin{table}[htbp]
\centering
\caption{Comparative Assessment Across Model Spaces}
\label{tab:comparison}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Domain \& Model} & \textbf{Space} & \textbf{Accuracy} & \textbf{Macro $F_1$} & \textbf{Latency} & \textbf{Sensitivity} \\
\midrule
Classical SVM & $\mathbb{R}^{1024}$ & 0.9900 & 0.8308 & 0.136 ms & Moderate \\
Classical Random Forest & $\mathbb{R}^{1024}$ & 0.9900 & 0.8308 & 7.074 ms & Low \\
Classical SVM (PCA) & $\mathbb{R}^6$ & 0.9300 & 0.5928 & 0.102 ms & High \\
Classical Random Forest (PCA) & $\mathbb{R}^6$ & 0.9900 & 0.8308 & 7.031 ms & Low \\
Quantum VQC (PCA) & $\mathbb{R}^6$ & 0.9800 & 0.4949 & 8.839 ms & High \\
\bottomrule
\end{tabular}
\end{table}

% --- 8. Benchmarking Results ---
\section{Benchmarking Results}
The empirical results from the benchmark execution across all five model configurations on the out-of-sample test set (100 crops) are presented in Table~\ref{tab:results}.

\begin{table}[htbp]
\centering
\caption{Out-of-Sample Empirical Benchmarking Results}
\label{tab:results}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Model Architecture} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{$F_1$-Score} & \textbf{Latency (ms)} \\
\midrule
SVM (1024-D) & 0.9900 & 0.7500 & 0.9949 & 0.8308 & $0.136 \pm 0.013$ \\
Random Forest (1024-D) & 0.9900 & 0.7500 & 0.9949 & 0.8308 & $7.074 \pm 0.340$ \\
SVM (6-D PCA) & 0.9300 & 0.5625 & 0.9646 & 0.5928 & $0.102 \pm 0.011$ \\
Random Forest (6-D PCA) & 0.9900 & 0.7500 & 0.9949 & 0.8308 & $7.031 \pm 0.261$ \\
Quantum VQC (6-D PCA) & 0.9800 & 0.4949 & 0.4949 & 0.4949 & $8.839 \pm 0.265$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Key Analytical Takeaways}
\begin{enumerate}
    \item \textbf{PCA Information Retention:} Random Forest achieved an identical macro $F_1$-score ($0.8308$) in both $\mathbb{R}^{1024}$ and $\mathbb{R}^6$ spaces, proving that the 6-component PCA manifold preserves non-linear decision boundaries.
    \item \textbf{Support Vector Machine Sensitivity:} SVM accuracy decreased from $0.9900$ ($F_1 = 0.8308$) in $\mathbb{R}^{1024}$ to $0.9300$ ($F_1 = 0.5928$) in $\mathbb{R}^6$, demonstrating that kernelized linear hyperplanes suffer greater loss under manifold compression compared to tree ensembles.
    \item \textbf{Quantum VQC Dynamics \& Expectation Shift:} Although weighted MSE optimization reduced training loss from $1.0420$ down to $0.5349$, evaluating expectation values under a strict zero-threshold ($\langle Z_0 \rangle + b \ge 0.0$) on an imbalanced test distribution (98 majority vs. 2 minority samples) mapped predictions toward the majority class (yielding $0.9800$ accuracy and macro $F_1 = 0.4949$).
    \item \textbf{Latency Trade-offs:} Classical SVM models exhibit sub-millisecond evaluation ($0.102\text{ ms}$ to $0.136\text{ ms}$), whereas decision tree ensemble traversal ($7.031\text{ ms}$) and PennyLane autograd quantum state simulations ($8.839\text{ ms}$) require higher evaluation time per sample.
\end{enumerate}

% --- 9. Conclusion ---
\section{Conclusion}
This benchmark confirms that 6-dimensional PCA embeddings extracted from YOLO object detector crops preserve sufficient information for non-linear classification. While parameterized quantum circuits achieve gradient convergence on this compressed manifold, evaluating expectation values against highly imbalanced test distributions requires dynamic decision thresholding beyond standard zero-level cuts. These findings establish a baseline for deploying hybrid QML architectures in constrained vision pipelines.

% --- 10. Acknowledgements ---
\section*{Acknowledgements}
Code development, benchmark automation, and manuscript formatting for this work were supported by generative AI tools. The authors retain full responsibility for the experimental design, dataset integrity, result validation, and final conclusions.

% --- 11. References ---
\begin{thebibliography}{10}

\bibitem{wierichs2022general}
David Wierichs, Josh Izaac, Cody Wang, and Cedric Yen-Yu Lin.
\newblock General parameter-shift rules for quantum gradients.
\newblock {\em Quantum}, 6:677, 2022.

\bibitem{redmon2018yolo}
Joseph Redmon and Ali Farhadi.
\newblock YOLOv3: An incremental improvement.
\newblock {\em arXiv preprint arXiv:1804.02767}, 2018.

\bibitem{bergholm2018pennylane}
Ville Bergholm et~al.
\newblock PennyLane: Automatic differentiation of hybrid quantum-classical computations.
\newblock {\em arXiv preprint arXiv:1811.04968}, 2018.

\bibitem{havlicek2019supervised}
Vojt{\v{e}}ch Havl{\'\i}{\v{c}}ek et~al.
\newblock Supervised learning with quantum-enhanced feature spaces.
\newblock {\em Nature}, 567(7747):209--212, 2019.

\bibitem{schuld2019quantum}
Maria Schuld and Nathan Killoran.
\newblock Quantum machine learning in feature Hilbert spaces.
\newblock {\em Physical Review Letters}, 122(4):040504, 2019.

\bibitem{jocher2023ultralytics}
Glenn Jocher et~al.
\newblock Ultralytics YOLOv8.
\newblock {\em GitHub Repository}, https://github.com/ultralytics/yolov8, 2023.

\end{thebibliography}

\end{document}
"""

def main():
    filename = "final_research_paper.tex"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_code.strip())
    print(f"[SUCCESS] Generated complete LaTeX file: '{filename}'")

if __name__ == "__main__":
    main()
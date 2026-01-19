# Curriculum Analysis - TODO/PRD Document

**Generated:** January 19, 2026
**Purpose:** Comprehensive list of issues and changes needed across AIML-Learning and QC-Learning curricula
**Target Audience:** Smaller model or developer to execute fixes

---

## Table of Contents

1. [Critical Issues](#critical-issues)
2. [AIML-Learning Issues](#aiml-learning-issues)
3. [QC-Learning Issues](#qc-learning-issues)
4. [Priority Matrix](#priority-matrix)

---

## Critical Issues

These issues break core functionality and should be addressed first.

### CRITICAL-001: QC Missing Routes (QC-Learning)

**Severity:** CRITICAL
**Impact:** 11 pages are completely inaccessible via routing
**File:** `/QC-Learning/quantum-computing-platform/src/routes.ts`

**Problem:**
The routes.ts file is missing route definitions for three entire sections. The pages exist, they appear in the navigation sidebar, and they're listed in `lessonOrder`, but there are no actual route imports or route definitions.

**Missing Sections:**

1. **Variational Algorithms** (4 pages)
   - `/variational-algorithms/nisq-overview` → `NISQOverview.tsx`
   - `/variational-algorithms/vqe` → `VQE.tsx`
   - `/variational-algorithms/qaoa` → `QAOA.tsx`
   - `/variational-algorithms/parameter-shift` → `ParameterShiftRule.tsx`

2. **Quantum Machine Learning** (4 pages)
   - `/quantum-ml/data-encoding` → `DataEncoding.tsx`
   - `/quantum-ml/quantum-kernels` → `QuantumKernels.tsx`
   - `/quantum-ml/qnn` → `QNN.tsx`
   - `/quantum-ml/qcnn` → `QCNN.tsx`

3. **Advanced Topics** (3 pages)
   - `/advanced-topics/quantum-gans` → `QuantumGANs.tsx`
   - `/advanced-topics/quantum-nlp` → `QuantumNLP.tsx`
   - `/advanced-topics/capstone-projects` → `CapstoneProjects.tsx`

**Fix Required:**

Add these imports at the top of routes.ts (around line 40):

```typescript
// Variational Algorithms
const NISQOverview = lazy(() => import('./pages/variational-algorithms/NISQOverview'));
const VQE = lazy(() => import('./pages/variational-algorithms/VQE'));
const QAOA = lazy(() => import('./pages/variational-algorithms/QAOA'));
const ParameterShiftRule = lazy(() => import('./pages/variational-algorithms/ParameterShiftRule'));

// Quantum Machine Learning
const DataEncoding = lazy(() => import('./pages/quantum-ml/DataEncoding'));
const QuantumKernels = lazy(() => import('./pages/quantum-ml/QuantumKernels'));
const QNN = lazy(() => import('./pages/quantum-ml/QNN'));
const QCNN = lazy(() => import('./pages/quantum-ml/QCNN'));

// Advanced Topics
const QuantumGANs = lazy(() => import('./pages/advanced-topics/QuantumGANs'));
const QuantumNLP = lazy(() => import('./pages/advanced-topics/QuantumNLP'));
const CapstoneProjects = lazy(() => import('./pages/advanced-topics/CapstoneProjects'));
```

Add these route definitions in the routes array (after line 179):

```typescript
{
  path: '/variational-algorithms',
  children: [
    { path: '/nisq-overview', component: NISQOverview },
    { path: '/vqe', component: VQE },
    { path: '/qaoa', component: QAOA },
    { path: '/parameter-shift', component: ParameterShiftRule },
  ],
},
{
  path: '/quantum-ml',
  children: [
    { path: '/data-encoding', component: DataEncoding },
    { path: '/quantum-kernels', component: QuantumKernels },
    { path: '/qnn', component: QNN },
    { path: '/qcnn', component: QCNN },
  ],
},
{
  path: '/advanced-topics',
  children: [
    { path: '/quantum-gans', component: QuantumGANs },
    { path: '/quantum-nlp', component: QuantumNLP },
    { path: '/capstone-projects', component: CapstoneProjects },
  ],
},
```

---

### CRITICAL-002: Dataset Mislabeling (AIML-Learning)

**Severity:** CRITICAL
**Impact:** Learners get wrong dataset, causing confusion about features and domain context
**File:** `/AIML-Learning/frontend/src/utils/datasets.ts`
**Lines:** 10-27

**Problem:**
The dataset labeled "Boston Housing" actually loads the Diabetes dataset from sklearn. The metadata says "House prices with 13 features" but the code loads clinical diabetes data.

**Current Code (WRONG):**
```typescript
{
  id: 'boston-housing',
  name: 'Boston Housing',
  description: 'House prices with 13 features',
  category: 'regression',
  code: `import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes  // ← WRONG! Loads diabetes, not housing

# Load dataset
data = load_diabetes()
...`
}
```

**Fix Option A - Rename to Diabetes (Recommended - simpler):**
```typescript
{
  id: 'diabetes',
  name: 'Diabetes',
  description: 'Diabetes progression prediction with 10 clinical features',
  category: 'regression',
  code: `import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# Load dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(f"Shape: {df.shape}")
print(f"\\nTarget stats:\\n{df['target'].describe()}")`
}
```

**Fix Option B - Use California Housing:**

Note: Boston Housing was deprecated in sklearn due to ethical concerns. Use California Housing instead:

```typescript
{
  id: 'california-housing',
  name: 'California Housing',
  description: 'House prices with 8 features (median house value)',
  category: 'regression',
  code: `import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(f"Shape: {df.shape}")
print(f"\\nTarget stats (median house value in $100k):\\n{df['target'].describe()}")`
}
```

---

## AIML-Learning Issues

### AIML-001: F1 Score Formula Uses Abbreviated Notation

**Severity:** LOW
**Impact:** May confuse beginners unfamiliar with P/R abbreviations
**File:** `/AIML-Learning/frontend/src/pages/machine-learning/Classification.tsx`
**Line:** 23

**Current:**
```typescript
formula: '2 × (P × R) / (P + R)'
```

**Suggested Fix:**
```typescript
formula: '2 × (Precision × Recall) / (Precision + Recall)'
```

---

### AIML-002: Quiz Options Ordering Could Be Clearer

**Severity:** LOW
**Impact:** Pedagogically confusing option ordering
**File:** `/AIML-Learning/frontend/src/utils/quizzes.ts`
**Lines:** 45-48

**Problem:**
For precision and recall quizzes, the first option shown is the formula for the OTHER metric, which could confuse learners even though the correct answer is indexed properly.

**rc7 (Precision quiz) options:**
- Option 0: `'TP / (TP + FN)'` ← This is RECALL formula
- Option 1: `'TP / (TP + FP)'` ← This is PRECISION (correct: 1) ✓

**rc8 (Recall quiz) options:**
- Option 0: `'TP / (TP + FP)'` ← This is PRECISION formula
- Option 1: `'TP / (TP + FN)'` ← This is RECALL (correct: 1) ✓

**Suggested Fix:**
Reorder options so the confusing similar formula isn't the first option:

```typescript
// rc7 - Precision
{ id: 'rc7', question: 'What is precision?',
  options: ['TP / (TP + FP)', 'TP / (TP + FN)', 'TN / (TN + FP)', 'Accuracy'],
  correct: 0,  // Update index to 0
  explanation: 'Precision is the ratio of true positives to all predicted positives.' },

// rc8 - Recall
{ id: 'rc8', question: 'What is recall?',
  options: ['TP / (TP + FN)', 'TP / (TP + FP)', 'TN / Total', 'Precision'],
  correct: 0,  // Update index to 0
  explanation: 'Recall is the ratio of true positives to all actual positives.' },
```

---

## QC-Learning Issues

### QC-001: Y Gate Visualization Is Incorrect

**Severity:** HIGH
**Impact:** Teaches wrong quantum mechanics - fundamental error in gate behavior
**File:** `/QC-Learning/quantum-computing-platform/src/components/quantum-visualizations/GateOperationD3.tsx`
**Lines:** 81-84

**Problem:**
The Y gate visualization shows the wrong final state position on the Bloch sphere.

**Quantum Physics Background:**
- Y|0⟩ = i|1⟩ (applies i phase and flips to |1⟩)
- On the Bloch sphere, Y|0⟩ should point to the SOUTH POLE (|1⟩ state), not the equator
- The visualization shows it going to `(centerX + radius, centerY)` which is the |+⟩ state on the equator
- The label shows `|+i⟩` which is also incorrect for Y|0⟩

**Current Code (WRONG):**
```typescript
case 'Y':
  finalState = { x: centerX + radius, y: centerY };  // Wrong - this is equator
  finalLabel = '|+i⟩';  // Wrong label
  break;
```

**Fix:**
```typescript
case 'Y':
  finalState = { x: centerX, y: centerY + radius };  // South pole (|1⟩)
  finalLabel = 'i|1⟩';  // Correct: Y|0⟩ = i|1⟩
  break;
```

**Note:** The visualization is simplified (2D projection of Bloch sphere), but it should at least show the correct final state. The Y gate applied to |0⟩ gives i|1⟩, which is the |1⟩ state with a global phase of i. On the Bloch sphere, this is the south pole.

---

### QC-002: Z Gate Visualization Could Be More Pedagogical

**Severity:** LOW
**Impact:** Pedagogically weak but not technically wrong
**File:** `/QC-Learning/quantum-computing-platform/src/components/quantum-visualizations/GateOperationD3.tsx`
**Lines:** 85-88

**Current Behavior:**
Z|0⟩ = |0⟩ (no visible change), which is mathematically correct.

**Problem:**
The Z gate's effect is invisible when applied to |0⟩ because Z only adds phase to the |1⟩ component. This doesn't demonstrate why Z is useful.

**Suggestion (Optional Enhancement):**
Consider adding a comment or tooltip explaining: "Z|0⟩ = |0⟩ (no visible change because Z only affects the |1⟩ component). Try Z on the |+⟩ state to see the phase flip: Z|+⟩ = |−⟩"

Or alternatively, add a feature to apply gates to |+⟩ initial state to show phase effects.

---

### QC-003: H Gate and X Gate Labels Could Include Ket Notation

**Severity:** LOW
**Impact:** Minor - cosmetic improvement
**File:** `/QC-Learning/quantum-computing-platform/src/components/quantum-visualizations/GateOperationD3.tsx`

**Current:**
- H gate shows `|+⟩` (correct, but could show the formula)
- X gate shows `|1⟩` (correct)

**Suggestion:**
Add the transformation as a subtitle or tooltip: "H|0⟩ = (|0⟩ + |1⟩)/√2 = |+⟩"

---

## Priority Matrix

| ID | Issue | Severity | Effort | Priority Score |
|----|-------|----------|--------|----------------|
| CRITICAL-001 | QC Missing Routes | CRITICAL | Low (copy-paste) | **P0** |
| CRITICAL-002 | Dataset Mislabeling | CRITICAL | Low | **P0** |
| QC-001 | Y Gate Visualization Wrong | HIGH | Low | **P1** |
| AIML-001 | F1 Formula Notation | LOW | Trivial | **P3** |
| AIML-002 | Quiz Options Order | LOW | Low | **P3** |
| QC-002 | Z Gate Pedagogy | LOW | Medium | **P4** |
| QC-003 | Gate Labels Enhancement | LOW | Low | **P4** |

---

## Execution Checklist

### Phase 1: Critical Fixes (Do First)

- [ ] **CRITICAL-001:** Add missing route imports to QC routes.ts
- [ ] **CRITICAL-001:** Add missing route definitions to QC routes.ts
- [ ] **CRITICAL-002:** Fix Boston Housing dataset mislabeling in AIML datasets.ts

### Phase 2: High Priority Fixes

- [ ] **QC-001:** Fix Y gate visualization coordinates and label

### Phase 3: Low Priority Improvements (Optional)

- [ ] **AIML-001:** Expand F1 formula notation
- [ ] **AIML-002:** Reorder precision/recall quiz options
- [ ] **QC-002:** Add Z gate pedagogical explanation
- [ ] **QC-003:** Enhance gate labels with full transformations

---

## Verification Steps

After making changes, verify:

1. **For CRITICAL-001:** Navigate to each of the 11 newly-routed pages and confirm they load
2. **For CRITICAL-002:** Run the dataset code and verify it loads the correct data
3. **For QC-001:** Visual inspection of Y gate animation - state should move to south pole

---

## Files Modified Summary

| File Path | Issues |
|-----------|--------|
| `QC-Learning/quantum-computing-platform/src/routes.ts` | CRITICAL-001 |
| `AIML-Learning/frontend/src/utils/datasets.ts` | CRITICAL-002 |
| `QC-Learning/.../GateOperationD3.tsx` | QC-001, QC-002, QC-003 |
| `AIML-Learning/.../Classification.tsx` | AIML-001 |
| `AIML-Learning/frontend/src/utils/quizzes.ts` | AIML-002 |

---

## Recommended Additions - AIML-Learning

This section contains recommendations for new lessons, visualizations, and content to make the AIML curriculum more comprehensive.

### AIML New Lessons - Tier 1 (Critical Additions)

#### AIML-NEW-001: Exploratory Data Analysis & Data Quality

**Priority:** CRITICAL
**Justification:** Students cannot assess data quality or identify issues before modeling. All competitor curricula cover this.
**Suggested Location:** New lesson in Foundations section (after Data Engineering)

**Topics to Cover:**
- Statistical summaries and distributions
- Missing data patterns and imputation strategies
- Outlier analysis and handling
- Multicollinearity detection
- Skewness, kurtosis, and distribution fitting

**Recommended Visualizations:**
- Interactive histogram/box plot/violin plot builder
- Correlation matrix heatmap
- Missing data pattern visualizer
- Outlier detection comparison tool

---

#### AIML-NEW-002: Feature Engineering & Selection

**Priority:** CRITICAL
**Justification:** 80% of ML success comes from features. Current coverage is superficial.
**Suggested Location:** New lesson in Foundations or Machine Learning section

**Topics to Cover:**
- Advanced feature interactions and polynomial features
- Domain-specific feature creation strategies
- Feature selection methods (correlation, mutual information, chi-square, RFE)
- Time-based and geographic features
- When to use each selection method

**Recommended Visualizations:**
- Feature interaction effect plots
- Feature importance comparison across methods
- Correlation-based feature clustering

---

#### AIML-NEW-003: Dimensionality Reduction (PCA, t-SNE, UMAP)

**Priority:** CRITICAL
**Justification:** Essential for high-dimensional data; only mentioned in glossary currently.
**Suggested Location:** New lesson in Machine Learning section (after Clustering)

**Topics to Cover:**
- Principal Component Analysis (PCA) theory and application
- t-SNE for visualization
- UMAP for visualization and clustering
- Feature extraction vs selection
- When to apply dimensionality reduction

**Recommended Visualizations:**
- **PCA Scree Plot:** Interactive explained variance visualization
- **3D to 2D Projection Demo:** Show data transformation in real-time
- **t-SNE/UMAP Clustering:** Interactive 2D/3D projection with cluster coloring

---

#### AIML-NEW-004: Imbalanced Classification & Resampling

**Priority:** CRITICAL
**Justification:** 90% of real-world classification problems are imbalanced. Not currently addressed.
**Suggested Location:** Machine Learning section (after Classification)

**Topics to Cover:**
- Class imbalance detection and impact
- Resampling techniques (SMOTE, undersampling, oversampling)
- Cost-sensitive learning
- Threshold tuning for imbalanced data
- Appropriate metrics (F1, AUC-ROC, precision-recall curves)

**Recommended Visualizations:**
- **SMOTE Visualization:** Show synthetic sample generation in 2D feature space
- **Resampling Comparison:** Before/after class distributions
- **Threshold Optimization Tool:** Interactive precision-recall tradeoff

---

#### AIML-NEW-005: Model Interpretability & Explainability (SHAP, LIME)

**Priority:** CRITICAL
**Justification:** Required for regulated industries; increasingly standard in ML practice.
**Suggested Location:** New section or after Model Evaluation

**Topics to Cover:**
- Feature importance methods (SHAP, LIME, permutation importance)
- Partial dependence plots (PDP)
- Individual conditional expectation (ICE)
- Model-agnostic explanation techniques
- When to prioritize interpretability over accuracy

**Recommended Visualizations:**
- **SHAP Force Plots:** Individual prediction explanations
- **SHAP Summary Plots:** Global feature importance
- **LIME Local Explanations:** Simplified model approximations
- **Partial Dependence Plots:** Feature effect on predictions

---

### AIML New Lessons - Tier 2 (Important Additions)

#### AIML-NEW-006: Hyperparameter Tuning & Optimization

**Priority:** HIGH
**Suggested Location:** Machine Learning section (after Model Training)

**Topics to Cover:**
- Grid search vs random search vs Bayesian optimization
- Hyperparameter importance analysis
- Learning rate scheduling strategies
- Early stopping mechanisms
- Cross-validation for hyperparameter selection

**Recommended Visualizations:**
- **Hyperparameter Heatmap:** Performance across parameter combinations
- **Optimization Trajectory:** Show Bayesian optimization exploration

---

#### AIML-NEW-007: Ensemble Methods Deep Dive

**Priority:** HIGH
**Suggested Location:** Machine Learning section (new dedicated lesson)

**Topics to Cover:**
- Random Forests mechanism and hyperparameters
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Bagging vs Boosting comparison
- Stacking and blending techniques
- When to ensemble vs single models

**Recommended Visualizations:**
- **Bagging vs Boosting Flowchart:** Side-by-side comparison
- **Feature Importance Comparison:** RF vs XGBoost vs LightGBM
- **Performance Benchmark Tool:** Compare ensemble strategies

---

#### AIML-NEW-008: Diffusion Models for Generative AI

**Priority:** HIGH
**Justification:** State-of-the-art for image generation; rapidly becoming standard.
**Suggested Location:** Generative AI section

**Topics to Cover:**
- DDPM (Denoising Diffusion Probabilistic Models)
- DDIM and faster sampling
- Guided diffusion and classifier-free guidance
- Latent diffusion (Stable Diffusion architecture)
- Text-to-image generation pipeline

**Recommended Visualizations:**
- **Denoising Process Animation:** Step-by-step noise removal
- **Latent Space Visualization:** Show compression and generation

---

#### AIML-NEW-009: Advanced NLP - Pre-training & Fine-tuning

**Priority:** HIGH
**Suggested Location:** NLP section (expand existing)

**Topics to Cover:**
- Masked language modeling (BERT-style)
- Causal language modeling (GPT-style)
- Next sentence prediction
- Instruction fine-tuning and RLHF basics
- Parameter-efficient fine-tuning (LoRA, adapters)

---

#### AIML-NEW-010: Vision Transformers & Modern CV Architectures

**Priority:** HIGH
**Suggested Location:** Deep Learning or Computer Vision section

**Topics to Cover:**
- Vision Transformer (ViT) architecture
- Patch embedding and position encoding
- Comparison with CNNs
- Hybrid architectures
- Self-supervised learning in vision (SimCLR, MAE, CLIP)

---

### AIML New Lessons - Tier 3 (Specialized Additions)

#### AIML-NEW-011: Graph Neural Networks

**Priority:** MEDIUM
**Topics:** Graph convolutions, message passing, GAT, knowledge graphs, social network analysis

#### AIML-NEW-012: Advanced Time Series - Neural Methods

**Priority:** MEDIUM
**Topics:** Transformer-based forecasting, N-BEATS, TCN, multivariate forecasting

#### AIML-NEW-013: MLOps Production Patterns (Expand Existing)

**Priority:** MEDIUM
**Topics:** Feature stores, model serving patterns, A/B testing, monitoring dashboards, data drift detection

#### AIML-NEW-014: Causal Inference Basics

**Priority:** MEDIUM
**Topics:** Causal graphs (DAGs), treatment effect estimation, counterfactual analysis

#### AIML-NEW-015: Recommender Systems

**Priority:** MEDIUM
**Justification:** Completely missing; covered by Coursera/Stanford courses
**Topics:** Collaborative filtering, content-based filtering, matrix factorization, deep learning recommenders

---

### AIML New Visualizations (Without New Lessons)

| Visualization | Target Lesson | Description |
|---------------|---------------|-------------|
| PCA Explained Variance | Dimensionality Reduction | Scree plot + cumulative variance |
| t-SNE/UMAP Projections | Dimensionality Reduction | Interactive cluster visualization |
| SHAP Values | Model Interpretability | Feature importance breakdowns |
| Hyperparameter Heatmaps | Model Training | Performance across parameter ranges |
| Data Drift Detection | MLOps Pipeline | Distribution shift over time |
| Transformer Architecture Flow | Transformers | Query/key/value computation paths |
| Graph Neural Network Propagation | GNN (new) | Node embedding message passing |

---

### AIML New Datasets

| Dataset | Category | Purpose |
|---------|----------|---------|
| Credit Fraud Detection | Classification | Highly imbalanced dataset |
| Movie Reviews (Sentiment) | NLP | Sentiment analysis practice |
| CIFAR-10 | Computer Vision | Image classification |
| Electricity Consumption | Time Series | Forecasting practice |
| MovieLens | Recommender | Collaborative filtering |
| CoNLL NER | NLP | Named entity recognition |

---

### AIML New Interactive Demos

1. **EDA Explorer:** Upload CSV, get instant statistical summaries, correlations, missing data patterns
2. **Feature Engineering Playground:** Create and test features interactively, see model impact
3. **Hyperparameter Tuner:** Grid/random/Bayesian comparison with live performance updates
4. **Data Drift Monitor:** Visualize distribution shifts, statistical tests
5. **Interpretability Dashboard:** SHAP/LIME explanations for uploaded models

---

## Recommended Additions - QC-Learning

This section contains recommendations for new lessons, visualizations, and content to make the Quantum Computing curriculum more comprehensive.

### QC New Lessons - Tier 1 (Critical Foundations)

#### QC-NEW-001: Linear Algebra Foundations for Quantum Computing

**Priority:** CRITICAL
**Justification:** Quantum computing requires linear algebra; curriculum currently assumes this knowledge.
**Suggested Location:** NEW SECTION before Quantum Fundamentals (or first lessons in Fundamentals)

**Topics to Cover (3-4 lessons):**
1. **Complex Numbers for Quantum:** Why complex numbers are needed, Euler's formula, polar form
2. **Vectors and Inner Products:** State vectors, bra-ket inner products, orthonormality
3. **Matrix Operations:** Matrix multiplication, unitary matrices, Hermitian matrices
4. **Eigenvalues and Eigenvectors:** Why they matter for measurement and VQE
5. **Tensor Products:** How multi-qubit states combine (essential for entanglement)

**Recommended Visualizations:**
- **Complex Number Visualizer:** Argand diagram with phase rotation
- **Matrix Multiplication Animation:** Step-by-step computation
- **Eigenvalue/Eigenvector Demo:** Show how matrices stretch/rotate vectors

---

#### QC-NEW-002: Quantum Notation & Formalism

**Priority:** CRITICAL
**Justification:** Students encounter Dirac notation without systematic explanation.
**Suggested Location:** Quantum Fundamentals (between "What is Quantum?" and "Qubits")

**Topics to Cover:**
- Complete guide to Dirac notation (|⟩ kets, ⟨| bras, |⟩⟨| outer products)
- Basis states and superposition notation
- Quantum state normalization (|α|² + |β|² = 1)
- Tensor product notation for multi-qubit systems (|00⟩, |01⟩, etc.)
- Expectation values and measurement formalism

**Recommended Visualization:**
- **Notation Builder:** Interactive tool showing how |ψ⟩ = α|0⟩ + β|1⟩ translates to probabilities

---

#### QC-NEW-003: Complex Amplitudes & Phase

**Priority:** CRITICAL
**Justification:** Phase is fundamental but barely explained; students don't understand interference.
**Suggested Location:** Quantum Fundamentals (after Superposition)

**Topics to Cover:**
- Difference between amplitude and probability
- Relative phase vs global phase (and why global phase doesn't matter)
- Why |+⟩ ≠ |−⟩ despite same measurement probabilities
- Quantum interference: constructive vs destructive
- Phase gates and their geometric meaning on Bloch sphere

**Recommended Visualizations:**
- **Phase Interference Demo:** Show how different phases create different interference patterns
- **Amplitude Evolution Animation:** How gates change both magnitude and phase

---

#### QC-NEW-004: Phase Kickback & Quantum Oracles

**Priority:** CRITICAL
**Justification:** Phase kickback is fundamental to Deutsch-Jozsa, Grover, Shor; not currently explained.
**Suggested Location:** Quantum Algorithms (before or with Algorithm Principles)

**Topics to Cover:**
- How eigenvalues appear as phases
- Deutsch algorithm as warm-up (1-qubit version)
- Connection to phase estimation
- How oracles encode problem information in phase
- Why phase information enables quantum speedup

**Recommended Visualization:**
- **Phase Kickback Animation:** Show controlled-U applying phase to control qubit

---

#### QC-NEW-005: Amplitude Amplification

**Priority:** CRITICAL
**Justification:** Core technique behind Grover; mentioned but not taught in depth.
**Suggested Location:** Quantum Algorithms (between Deutsch-Jozsa and Grover)

**Topics to Cover:**
- Inversion about average operation (geometric intuition)
- Why Grover iterations work (2D plane rotation)
- Optimal number of iterations (π/4 × √N)
- Generalization beyond search
- Connection to reflection operators

**Recommended Visualization:**
- **Geometric Grover Animation:** Show 2D plane with state vector rotating toward target by θ each iteration

---

#### QC-NEW-006: Tensor Products for Multi-Qubit Systems

**Priority:** HIGH
**Justification:** Multi-qubit lessons jump to entanglement without explaining state composition.
**Suggested Location:** Quantum Fundamentals (before Entanglement)

**Topics to Cover:**
- Tensor product notation (|ψ₁⟩ ⊗ |ψ₂⟩)
- Separable vs entangled states
- Why some states can't be written as tensor products
- Computational basis for n qubits
- State vector dimension (2ⁿ)

**Recommended Visualization:**
- **Separability Checker:** Input 2-qubit state, determine if separable or entangled

---

### QC New Lessons - Tier 2 (Algorithm & Theory)

#### QC-NEW-007: Simon's Algorithm

**Priority:** HIGH
**Justification:** Key algorithm showing exponential speedup; natural bridge to Shor's.
**Suggested Location:** Quantum Algorithms (between Deutsch-Jozsa and QFT)

**Topics to Cover:**
- Hidden subgroup problem introduction
- Simon's problem definition
- Quantum circuit construction
- Why it achieves exponential speedup
- Connection to period-finding in Shor's

---

#### QC-NEW-008: Quantum Phase Estimation (QPE)

**Priority:** HIGH
**Justification:** Foundation for VQE, chemistry simulation; mentioned but not taught.
**Suggested Location:** Quantum Algorithms (after QFT, before Variational Algorithms)

**Topics to Cover:**
- Eigenvalue estimation problem
- QPE circuit construction
- Role of QFT inverse
- Precision requirements (number of ancilla qubits)
- Connection to chemistry and optimization

**Recommended Visualization:**
- **QPE Circuit Walkthrough:** Step-by-step eigenvalue extraction

---

#### QC-NEW-009: Barren Plateaus & Trainability

**Priority:** HIGH
**Justification:** BarrenPlateauD3 visualization exists but no conceptual lesson.
**Suggested Location:** Variational Algorithms section

**Topics to Cover:**
- Why random circuits have flat loss landscapes
- Cost function concentration phenomenon
- Mitigation strategies (parameter initialization, layer-by-layer training)
- Hardware-efficient ansätze design
- Connection to expressibility and entanglement

**Recommended Visualization Enhancement:**
- Link existing BarrenPlateauD3 to this lesson
- Add **2D Loss Landscape Visualizer** showing gradient magnitudes

---

#### QC-NEW-010: Quantum Advantage Reality Check (for QML)

**Priority:** HIGH
**Justification:** QML lessons exist but don't discuss when/if quantum actually helps.
**Suggested Location:** Quantum ML section (early)

**Topics to Cover:**
- Dequantization attacks (classical simulation of QML)
- When quantum encoding might help (feature space arguments)
- Barren plateaus in QML (often worse than chemistry)
- Data loading bottleneck
- Current state: no clear quantum advantage demonstrated yet
- Honest assessment of near-term prospects

---

### QC New Lessons - Tier 3 (Hardware & Advanced)

#### QC-NEW-011: Superconducting Qubits Deep Dive

**Priority:** MEDIUM
**Topics:** IBM/Google approach, transmon physics, T1/T2 times, gate implementation, connectivity constraints

#### QC-NEW-012: Trapped Ion Qubits Deep Dive

**Priority:** MEDIUM
**Topics:** IonQ/Quantinuum approach, all-to-all connectivity, gate fidelities, scaling challenges

#### QC-NEW-013: Hardware Constraints & Transpilation

**Priority:** MEDIUM
**Topics:** SWAP insertion, connectivity topologies, circuit compilation, T1/T2 depth budgets

#### QC-NEW-014: Optimization Strategies for Variational Algorithms

**Priority:** MEDIUM
**Topics:** Classical optimizers (COBYLA, SPSA, Adam), analytic vs numeric gradients, natural gradient descent

#### QC-NEW-015: Quantum Feature Spaces (Expand Quantum Kernels)

**Priority:** MEDIUM
**Topics:** What makes good feature maps, implicit feature spaces, connection to classical kernel methods

---

### QC New Visualizations

| Visualization | Target Lesson | Description |
|---------------|---------------|-------------|
| **Unitary Matrix Visualizer** | Gates/General | Show gate matrices with complex phase coloring |
| **Circuit-to-Math Converter** | Gates | Convert visual circuit to matrix multiplication |
| **Interference Pattern Demo** | Phase/Superposition | Show constructive/destructive interference |
| **Phase Kickback Animation** | Phase Kickback (new) | Controlled-U phase transfer to control qubit |
| **Geometric Grover** | Grover/Amplitude Amp | 2D plane rotation toward target state |
| **2D Parameter Landscape** | Barren Plateaus | Loss surface with gradient magnitudes |
| **Quantum State Tomography** | Measurement | Reconstruct state from multiple basis measurements |
| **Tensor Product Builder** | Multi-Qubit States | Combine single-qubit states into multi-qubit |
| **Eigenvalue Extraction** | QPE (new) | Show phase appearing in measurement |

---

### QC New Practical Content

#### Code Exercises (30-40 recommended)

**Beginner:**
- Create specific superposition states with target amplitudes
- Verify Deutsch-Jozsa for constant vs balanced functions
- Build Bell states using H and CNOT

**Intermediate:**
- Implement Grover for small databases
- Build QFT circuit from scratch
- Create parameterized ansatz for VQE

**Advanced:**
- Optimize VQE for H₂ molecule
- Implement error mitigation techniques
- Design custom feature map for classification

#### Circuit Challenges

| Challenge | Difficulty | Description |
|-----------|------------|-------------|
| Create Bell State | ⭐ | Build |Φ+⟩ = (|00⟩ + |11⟩)/√2 |
| GHZ State | ⭐⭐ | Extend to 3-qubit entanglement |
| Grover 2-qubit | ⭐⭐ | Search 4-element database |
| Teleportation | ⭐⭐ | Full protocol implementation |
| VQE H₂ | ⭐⭐⭐ | Find ground state energy |
| Custom Oracle | ⭐⭐⭐ | Design Grover oracle for specific function |

---

## Updated Priority Matrix (Including New Content)

### Bug Fixes & Issues

| ID | Issue | Severity | Status |
|----|-------|----------|--------|
| CRITICAL-001 | QC Missing Routes | CRITICAL | ✅ FIXED |
| CRITICAL-002 | Dataset Mislabeling | CRITICAL | Pending |
| QC-001 | Y Gate Visualization | HIGH | ✅ FIXED |
| QC-002 | Z Gate Pedagogy | LOW | ✅ FIXED |
| QC-003 | Gate Labels | LOW | ✅ FIXED |
| AIML-001 | F1 Formula | LOW | Pending |
| AIML-002 | Quiz Options | LOW | Pending |

### New Content Priority

| Priority | AIML-Learning | QC-Learning |
|----------|---------------|-------------|
| **P0 - Critical** | EDA & Data Quality, Feature Engineering, Dimensionality Reduction | Linear Algebra Foundations, Quantum Notation, Complex Amplitudes & Phase |
| **P1 - High** | Imbalanced Classification, Model Interpretability (SHAP/LIME), Hyperparameter Tuning | Phase Kickback, Amplitude Amplification, Tensor Products, Simon's Algorithm, QPE |
| **P2 - Medium** | Ensemble Methods, Diffusion Models, Advanced NLP Pre-training | Barren Plateaus Theory, QML Reality Check, Hardware Deep Dives |
| **P3 - Low** | GNNs, Causal Inference, Recommender Systems | Optimization Strategies, Feature Spaces |

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
- [x] Fix QC missing routes
- [x] Fix Y gate visualization
- [x] Add Z gate pedagogical explanation
- [x] Add gate transformation formulas
- [ ] Fix AIML dataset mislabeling
- [ ] Fix AIML quiz ordering

### Phase 2: Critical New Content (Weeks 2-4)

**AIML:**
- [ ] Create EDA & Data Quality lesson + visualizations
- [ ] Create Feature Engineering lesson + demos
- [ ] Create Dimensionality Reduction lesson + PCA/t-SNE visualizations

**QC:**
- [ ] Create Linear Algebra Foundations section (3-4 lessons)
- [ ] Create Quantum Notation lesson
- [ ] Create Complex Amplitudes & Phase lesson + interference visualization

### Phase 3: High Priority Content (Weeks 5-8)

**AIML:**
- [ ] Create Imbalanced Classification lesson + SMOTE visualization
- [ ] Create Model Interpretability lesson + SHAP visualizations
- [ ] Create Hyperparameter Tuning lesson

**QC:**
- [ ] Create Phase Kickback lesson + visualization
- [ ] Create Amplitude Amplification lesson + geometric Grover visualization
- [ ] Create Simon's Algorithm lesson
- [ ] Create Quantum Phase Estimation lesson

### Phase 4: Medium Priority Content (Weeks 9-12)

**AIML:**
- [ ] Expand Ensemble Methods coverage
- [ ] Add Diffusion Models lesson
- [ ] Expand NLP pre-training content

**QC:**
- [ ] Create Barren Plateaus theory lesson
- [ ] Create QML Reality Check lesson
- [ ] Add hardware-specific lessons (superconducting, trapped ion)

### Phase 5: Practical Content & Polish (Weeks 13-16)

**Both Curricula:**
- [ ] Add code exercises (30-40 per curriculum)
- [ ] Add circuit/ML challenges
- [ ] Add new datasets
- [ ] Create interactive demos
- [ ] Add difficulty tags and learning paths

---

## Files to Create Summary

### AIML-Learning New Files

| File Path | Content Type |
|-----------|--------------|
| `pages/foundations/EDA.tsx` | New lesson |
| `pages/foundations/FeatureEngineering.tsx` | New lesson |
| `pages/machine-learning/DimensionalityReduction.tsx` | New lesson |
| `pages/machine-learning/ImbalancedClassification.tsx` | New lesson |
| `pages/machine-learning/Interpretability.tsx` | New lesson |
| `pages/machine-learning/HyperparameterTuning.tsx` | New lesson |
| `pages/machine-learning/EnsembleMethods.tsx` | New lesson |
| `pages/generative-ai/DiffusionModels.tsx` | New lesson |
| `components/diagrams/PCAVisualizationD3.tsx` | New visualization |
| `components/diagrams/TSNEVisualizationD3.tsx` | New visualization |
| `components/diagrams/SHAPVisualizationD3.tsx` | New visualization |
| `components/diagrams/SMOTEVisualizationD3.tsx` | New visualization |
| `components/diagrams/HyperparameterHeatmapD3.tsx` | New visualization |

### QC-Learning New Files

| File Path | Content Type |
|-----------|--------------|
| `pages/math-foundations/ComplexNumbers.tsx` | New lesson |
| `pages/math-foundations/LinearAlgebra.tsx` | New lesson |
| `pages/math-foundations/TensorProducts.tsx` | New lesson |
| `pages/quantum-basics/QuantumNotation.tsx` | New lesson |
| `pages/quantum-basics/ComplexAmplitudes.tsx` | New lesson |
| `pages/quantum-algorithms/PhaseKickback.tsx` | New lesson |
| `pages/quantum-algorithms/AmplitudeAmplification.tsx` | New lesson |
| `pages/quantum-algorithms/SimonAlgorithm.tsx` | New lesson |
| `pages/quantum-algorithms/PhaseEstimation.tsx` | New lesson |
| `pages/variational-algorithms/BarrenPlateaus.tsx` | New lesson |
| `pages/quantum-ml/QuantumAdvantageReality.tsx` | New lesson |
| `components/quantum-visualizations/UnitaryMatrixD3.tsx` | New visualization |
| `components/quantum-visualizations/InterferencePatternD3.tsx` | New visualization |
| `components/quantum-visualizations/PhaseKickbackD3.tsx` | New visualization |
| `components/quantum-visualizations/GeometricGroverD3.tsx` | New visualization |
| `components/quantum-visualizations/TensorProductBuilderD3.tsx` | New visualization |

---

*End of PRD Document*

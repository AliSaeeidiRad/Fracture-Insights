# Integrating Pattern Clustering for Enhanced Tibial PCL Avulsion Fracture Analysis

This repository contains the code, datasets, and supplementary materials for the study _"Advancements in 3D Fracture Mapping: Integrating Pattern Clustering for Enhanced Tibial PCL Avulsion Fracture Analysis"_ by Ali Saeedi Rad et al. The study introduces a novel, open-source approach to analyzing tibial posterior cruciate ligament (PCL) avulsion fractures by combining 3D fracture mapping with advanced pattern clustering techniques.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [References](#references)

---

## Overview

This study leverages:

1. **3D Fracture Mapping**: Extracting fracture lines and mapping them onto a standard tibial model.
2. **Pattern Clustering**: Utilizing Fréchet distance and Procrustes analysis combined with hierarchical clustering to identify distinct fracture patterns.

The dataset comprises 58 tibial PCL avulsion fracture cases, processed to extract and cluster fracture curves into distinct morphological groups.

Key outcomes include:

- Four primary fracture pattern groups visualized using Kernel Density Estimation (KDE).
- An open-source framework implemented in Python to promote reproducibility.

---

## Project Structure

```
.
├── Dataset/                # Contains fracture data in .txt and .stl formats
│   ├── Curves/             # Extracted fracture curves
│   ├── Sample/             # Sample STL model
│   └── STL/                # Tibial and fracture segmentations
├── figures/                # Scripts for generating visualizations
paper
├── output/                 # Clustered outputs and similarity matrices
├── pc/                     # Python scripts for clustering and similarity computation
├── output_vtks/            # VTK outputs for 3D visualizations
├── poetry.lock             # Poetry dependency lockfile
├── pyproject.toml          # Poetry project configuration
└── script.py               # Main script to execute the workflow
```

---

## Key Features

1. **Similarity Computation**:

   - Fréchet Distance
   - Procrustes Analysis
   - Combined similarity metrics

2. **Clustering**:

   - Hierarchical clustering algorithm for grouping fracture patterns.

3. **Visualization**:

   - Kernel Density Estimation (KDE) for fracture density maps.
   - Cluster visualization using ParaView and Python.

4. **Open Source**:
   - Reproducible framework built with Python and SciPy.

---

## Dependencies

- `python = "^3.10"`
- `tqdm = "^4.67.1"`
- `numpy = "^2.2.1"`
- `scipy = "^1.14.1"`
- `similaritymeasures = "^1.2.0"`
- `scikit-image = "^0.25.0"`
- `numpy-stl = "^3.2.0"`
- `matplotlib = "^3.10.0"`
- `seaborn = "^0.13.2"`
- `vtk = "^9.4.1"`

---

## Installation

1. Clone this repository:

```bash
   git clone https://github.com/AliSaeeidiRad/Fracture-Insights
   cd Fracture-Insights
```

2. Install Python 3.10.6 using one of the following methods:

   **Option A: Using `asdf`**

   Ensure `asdf` is installed, then use the `.tool-versions` file in the repository:

```bash
   asdf install
   python -m venv .env
   source .env/bin/activate  # On Windows: .env\Scripts\activate
```

**Option B: Using `conda`**

Ensure `conda` is installed, then:

```bash
   conda create -n fracture-insights python=3.10.6
   conda activate fracture-insights
```

**Option C: Using `uv`**

Ensure `uv` is installed, then:

```bash
   uv venv
```

3. Install package and dependencies:

```bash
   pip install -e .
```

---

## Usage

1. **Run the Main Script**:
   Execute the data processing and clustering pipeline:

   ```bash
   python script.py --dir-curves Dataset/Curves --t 4 --alpha 1.0 --beta 1.0 --output output --sample-stl Dataset/Sample/Sample.stl
   ```

   Additional flags:

   - `--only-export`: Enable this if you just need VTK exports with results of the previous analysis.
   - `--only-plots`: Enable this if you just need plots from previous analysis.

2. **Explore the Outputs**:
   - Clustering results: `output/`
   - VTK files for 3D visualization: `output_vtks/`

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

## Citation

<!-- If you use this dataset in your research or projects, please cite:

```bibtex
@article{SaeediRad2024,
  title = {Fracture Morphology and Pattern Recognition in Tibial PCL Avulsion Injuries Using Machine Learning-Enhanced 3D Mapping},
  author = {Ali Saeidi Rad, Azadeh Ghouchani and Ehsan Vahedi},
  journal = {},
  year = {2025},
  volume = {},
  number = {},
  pages = {},
  doi = {},
  publisher = {},
}
``` -->

---

For further inquiries, feel free to [contact](mailto:alisaeeidirad78@gmail.com) or open an issue.

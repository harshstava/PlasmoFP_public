# PlasmoFP\_public

This repository contains all code used to reproduce results in:
***PlasmoFP: leveraging deep learning to predict protein function of uncharacterized proteins across the malaria parasite genus***

---
We provide step-by-step instructions to reproduce all results from the PlasmoFP paper. Each subsection below corresponds to a main finding. 

### Data and trained model availability

- **Datasets**: Raw and processed data used for training, validation, and testing are available at [INSERT ZENODO LINK] in the `processed_data_90_30` folder


- **Pre-trained Models**: Trained model weights for PlasmoFP are included in this repo in the `data` folder. SAR-CNN, CAFA-TM-Vec, and CAFA-CNN model weights are provided in their respective `trained_*` directories at [INSERT ZENODO LINK]

- **Supporting Files**: Additional resources including MultiLabelBinarizer files (`.mlb`) and GO term ancestor files are organized in the `data` folder. 

---

### 1. PlasmoFP models are trained to predict GO terms for each subontology

PlasmoFP models are trained to predict GO terms for each subontology.

1. Generate train/val/test splits:

   ```bash
   src/create_data_splits.ipynb
   ```
2. Tune architectures using subontology-specific notebooks:

   ```bash
   src/tune_architecture/
   ```
3. Train PlasmoFP models (deterministic + deep ensembles):

   ```bash
   src/train_plasmoFP_models.py
   ```

---

### 2. PlasmoFP models quantify uncertainty with deep ensembles  

1. Run uncertainty analysis:

   ```bash
   src/uncertainty_analysis.py
   ```
2. Generate effective scores (per-term FDR cutoffs):

   ```bash
   src/effective_scores/
   ```

---

### 3. Phylogenetic relevance and TM-Vec embeddings improve function prediction in SAR

1. Build databases:

   ```bash
   scripts/create_blast_databases.sh
   scripts/create_foldseek_databases.sh
   ```
2. Run searches:

   ```bash
   scripts/run_blast_searches.sh
   scripts/run_foldseek_searches.sh
   ```
3. Map SwissProt GO terms:

   ```bash
   src/baseline_models/map_go_terms_for_blast_foldseek.ipynb
   ```
4. Build baselines:

   ```bash
   src/build_blast_baseline.py
   src/build_foldseek_basline.py
   src/evaluate_blast_foldseek_baseline.py
   ```
5. Preprocess CAFA data:

   ```bash
   cafa_preprocessing.py
   ```
6. Train baseline models (CAFA-CNN, CAFA-TM-Vec, SAR-CNN):

   ```bash
   src/train_CAFA_CNN.py
   src/train_CAFA_TM_VEC.py
   src/train_SAR_CNN.py
   ```
7. Predict on test set:

   ```bash
   src/predict_baseline_models_on_test.py
   ```
8. Run inference with PlasmoFP models:

   ```bash
   src/predict_PlasmoFP_with_uncertainty.py
   src/predict_PlasmoFP_on_CAFA_sequences_with_uncertainty.py
   ```

---

### 4. PlasmoFP models outperform existing protein function prediction models for *Plasmodium* and sustain performance with intrinsically disordered proteins

1. Run inference with DeepGOPlus & ProteInfer:

   ```bash
   notebooks/Proteinfer_on_test.ipynb
   notebooks/DeepGOPlus_on_test.ipynb
   ```
2. Run inference with PlasmoFP on holdout sets:

   ```bash
   src/predict_PlasmoFP_with_uncertainty.py
   ```
3. Analyze disorder sequences:

   ```bash
   notebooks/PlasmoFP_disorder_analysis.ipynb
   ```

---

### 5. PlasmoFP models predict GO terms for partially annotated proteins

1. Build similarity matrix (SwissProt):

   ```bash
   src/co_occurance/build_cooccurance_matracies.py
   ```
2. Calculate imputation/efficiency metrics:

   ```bash
   src/co_occurance/calculate_mask_and_compare_imputation_with_uncertainty.py
   ```

---

### 6. New functional annotations for partially annotated and proteins of unknown function in *Plasmodium* 

1. Build gene dicts (19 Plasmodium species):

   ```bash
   notebooks/generate_gene_dicts_for_prediction.ipynb
   ```
2. Run inference:

   ```bash
   notebooks/PlasmoFP_run_inference.ipynb
   ```
3. Generate counts:

   ```bash
   notebooks/figure_3.ipynb
   ```
4. Cluster predictions:

   ```bash
   notebooks/PlasmoFP_cluster_predictions.ipynb
   ```
5. Calculate Resnik similarity matrices:

   ```bash
   src/calculate_resnik_similarity.py
   ```
6. Generate GO term clusters (up to 'Protease' section):

   ```bash
   notebooks/generate_GO_term_clusters.ipynb
   ```

---

### 7. New functional annotations for partially annotated and proteins of unknown function in *Plasmodium* 

Recreate results and tables:

```bash
notebooks/PlasmoFP_RNA_associated.ipynb
```

---

### 8. PlasmoFP predictions expand existing functional classes in human-infecting malaria parasites

* **Protease analysis:**

  ```bash
  src/query_go_terms.py
  notebooks/generate_GO_term_clusters.ipynb (Protease section)
  ```
* **Transporter analysis:**

  ```bash
  notebooks/generate_GO_term_clusters.ipynb (Transporter section)
  ```

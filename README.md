# PlasmoFP_public

This repository contains all code used to reproduce results in: "PlasmoFP: leveraging deep learning to predict protein function of uncharacterized proteins across the malaria parasite genus"

To reproduce findings associated with the paper below are subheadings and the associated files used to create the results: 

PlasmoFP models are trained to predict GO terms for each subontology

a. To first generate the train/val/test splits use the @src/create_data_splits.ipynb file. 
b. Then use the subontology-specific notebooks found in @src/tune_architecture 
c. Train PlasmoFP models using the @scr/train_plasmoFP_models.py script (this will produce trained models for both deterministic and deep ensembles)

PlasmoFP models quantify uncertainty with deep ensembles  
a. Using the trained deep ensemble models run the @src/uncertainty_analysis.py script 
b. To generate effective_scores (i.e. thresholds which correspond to per-term FDR cutoffs) run the subontology-specific notebooks found in @src/effective_scores 

Phylogenetic relevance and TM-Vec embeddings improve function prediction in SAR
a. Build BLAST and Foldseek databases using the @scripts/crate_blast_databases.sh/, @scripts/create_foldseek_databases.sh using the test split data 
b. Run BLAST and Foldseek search using @scripts/run_blast_searches.sh and @scripts/run_foldseek_searches.sh
c. Map SwissProt GO terms using the @src/baseline_models/map_go_terms_for_blast_foldseek.ipynb 
d. Run the @src/build_blast_baseline.py and @src/build_foldseek_basline.py scripts to generate dictionaries for use in evaluation with @src/evaluate_blast_foldseek_baseline.py 
e. Preprocess the CAFA data using @cafa_preprocessing.py
f. Train CAFA-CNN, CAFA-TM-Vec, and SAR-CNN baseline models using @src/train_CAFA_CNN.py, @src/train_CAFA_TM_VEC.py, @src/train_SAR_CNN.py files 
g. Predict on test set using @src/predict_baseline_models_on_test.py 
h. Run inference on PlasmoFP models for test set sequences using @src/predict_PlasmoFP_with_uncertainty.py
i. Run inference on PlasmoFP models for CAFA sequences using @src/predict_PlasmoFP_on_CAFA_sequences_with_uncertainty 

PlasmoFP models outperform existing protein function prediction models for Plasmodium and sustain performance with intrinsically disordered proteins
a. Run inference on DeepGOPlus and ProteInfer models for all subontology Plasmodium holdoutsets sets using @notebooks/Proteinfer_on_test.ipynb and @DeepGOPlus_on_test.ipynb
b. Run @src/predict_PlasmoFP_with_uncertainty.py using the Plasmodium holdout set 
c. Run @notebooks/PlasmoFP_disorder_analysis.ipynb to generate metrics on disordered sequences 

PlasmoFP models predict GO terms for partially annotated proteins
a. Build similarity matrix for SwissProt annotations using @src/co_occurance/build_cooccurance_matracies.py 
b. Calculate imputation/efficiency metrics using @src/co_occurance/calculate_mask_and_compare_imputation_with_uncertainty.py

New functional annotations for partially annotated and proteins of unknown function in Plasmodium 
a. Build gene-dicts for 19 Plasmodium species using @notebooks/generate_gene_dicts_for_prediction.ipynb
b. Run inference on gene-dicts using @notebooks/PlasmoFP_run_inference 
c. Generate counts using @notebooks/figure_3.ipynb 
d. Run the first block in @notebooks/PlasmoFP_cluster_predictions.ipynb to generate a text file containing combined and existing terms found across gene dicts. 
e. Calculate Resnik simlarlity matracies for all terms using @src/calculate_resnik_similarity.py and per subontology clustering .tsv files. 
f. Run @notebooks/generate_GO_term_clusters.ipynb up until the 'Protease' subheading. 

Experimental validation of expanded PlasmoFP RNA‐associated protein repertoire across the Plasmodium genus
a. Run @notebooks/PlasmoFP_RNA_associated.ipynb to recreate results and tables 

PlasmoFP predictions expand existing functional classes in human-infecting malaria parasites
a. For the protease analysis, run @src/query_go_terms.py to generate tsv files containing target terms and parent terms. Then run the 'Protease' subheading in @notebooks/generate_GO_term_clusters.ipynb
b. For the transporter analysis run the 'Transporter' subheading in @notebooks/generate_GO_term_clusters.ipynb
combined_existing_pfp_NEW_2


 
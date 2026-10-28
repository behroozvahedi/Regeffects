# Regeffects / EMPRES

EMPRES predicts gene expression, as log10(1 + TPM), from the DNA sequence around a gene's transcription start site (TSS) and transcription termination site (TTS). It is a two-branch 1D convolutional network: one branch reads a 5 kb TSS window, the other a 5 kb TTS window. There are five EMPRES types, which differ only in how the sequence is represented:

| Type | Input representation | Channels per position |
|---|---|---|
| EMPRES 0 | one-hot DNA (A, C, G, T; any other base = all zero), 5,000 positions | 4 |
| EMPRES 1 | PlantCaduceus embeddings | 384 |
| EMPRES 2 | PlantCaduceus embeddings + a2z accessibility predictions | 385 |
| EMPRES 3 | PlantCaduceus embeddings + a2z embeddings | 1,309 |
| EMPRES 4 | a2z embeddings | 925 |

The models are trained and evaluated by 5-fold cross-validation on the PhytoExpr data: 589,125 genes from 17 species, using PhytoExpr's `group_for_cross_validation` folds. They are then used to predict:
- **SIEVE**: the *Brachypodium distachyon* BD21-3 mutant population;
- **Test species**: 10 species not used in training.

## Repository layout

- `Data_processing/` builds the PlantCaduceus and a2z embeddings and makes the EMPRES 1–4 predictions for SIEVE and the test species from `.h5` embedding files.
- `Scripts/` trains and evaluates EMPRES, builds the one-hot inputs for EMPRES 0, and makes and validates the EMPRES 0 predictions for SIEVE and the test species. The training, testing and prediction scripts work for all five types.

## Conventions

- **Folds.** Fold *k* uses one CV group for validation and the next for testing; the other three are used for training:

  | Fold | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | `--val_group` | 1 | 2 | 3 | 4 | 5 |
  | `--test_group` | 2 | 3 | 4 | 5 | 1 |

- **Model folders.** Each type and fold has its own folder: `<out_dir>/val{V}_test{T}/EMPRES_{type}/`. It holds the checkpoints (`checkpoint_trial_*.pth`), `trial_results.csv` and `predictions.csv`.
- **Training inputs** are in one folder, `npy_files/`:
  - `gene.npy`, `species.npy`, `family.npy`, `TPM.npy`, `group_for_cross_validation.npy`;
  - per type: `tss_OHE.npy`/`tts_OHE.npy` (EMPRES 0), `tss_embeddings_PlantCad.npy`/`tts_embeddings_PlantCad.npy`, `tss_embeddings_a2z.npy`/`tts_embeddings_a2z.npy`, `tss_predictions_a2z.npy`/`tts_predictions_a2z.npy`;
  - per training split, for standardising EMPRES 1–4: `global_stats_train_<groups>.npz`.
- **Scale.** Model outputs and `observed_TPM` are in log10(1 + TPM) space.
- **Top models.** Use `--top_x 5` for testing and every prediction step. The default in `test_PC_a2z.py` and `generate_predictions.py` is 1, and `1_Validating_predictions.R` averages exactly five models.
- **How the scripts are run.** The `Data_processing/` scripts have no command-line options. Paths are set in `workdir`/`outdir` inside `main()`, and most of them loop over all species or all EMPRES variants. The `Scripts/` Python scripts take the options shown below.

Some inputs are prepared outside this repository and are not described here:
- the conversion of the PlantCaduceus and a2z training embeddings into `npy_files/` and the `global_stats_train_*.npz` files;
- the one-hot training files for EMPRES 0.

## Run order

### 1. Embeddings for training (`Data_processing/`)

1. `python prepare.sequences.py` — cuts the TSS and TTS sequences of every training species in `data.csv` into overlapping chunks sized for each model. Writes `caduceus.sequences.<species>.tsv` (512 bp chunks) and `a2z.sequences.<species>.tsv` (600 bp chunks).
2. `python sequence2embedding.caduceus.py` — computes PlantCaduceus (`PlantCaduceus_l20`) embeddings for those chunks on the GPU and writes them per species (`caduceus.training.data.<species>.bin`/`.tsv`).
3. `python sequence2embedding.a2z.py` — computes a2z (DanQ accessibility model) embeddings and predictions for the same genes and writes them per species.

The `npy_files/` used in step 2 are then built from these outputs (outside this repository).

### 2. Training and cross-validation (`Scripts/`)

Repeat steps 1–4 for every EMPRES type and every fold.

1. `python train_PC_a2z.py --data_dir <npy_files> --out_dir <models> --val_group 1 --test_group 2 --EMPRES_type 0 [--data_on_gpu | --in_ram] [--num_workers N --prefetch_factor 2]` — tunes and trains one EMPRES type on one fold with Optuna (200 trials, early stopping on validation loss), saving a checkpoint per trial and the study history. If an interrupted study is restarted, it continues up to 200 trials. `--data_on_gpu` and `--in_ram` only change how batches are delivered, not the results.
2. `python model_info_df.py --out_dir <models> --val_group 1 --test_group 2 --EMPRES_type 0` — reads every trial checkpoint of that fold and writes `trial_results.csv`, ranked by validation loss. It stops with an error if no checkpoint can be read.
3. `python test_PC_a2z.py --data_dir <npy_files> --out_dir <models> --val_group 1 --test_group 2 --top_x 5 --EMPRES_type 0` — evaluates the top-X models and their ensemble on the fold's test group and writes `test_results_EMPRES_{type}_fold_{n}.tsv`.
4. `python generate_predictions.py --data_dir <npy_files> --out_dir <models> --val_group 1 --test_group 2 --top_x 5 --EMPRES_type 0` — writes `predictions.csv` for the fold's test genes: each top model's prediction, their mean (`ensemble_pred`) and `observed_TPM`.
5. `python aggregate_predictions.py --out_dir <models> --EMPRES_type 0 1 2 3 4 --npy_dir <npy_files> [--phyto_B_file <B.csv> --phyto_C_file <C.csv>] [--use_saved]` — combines the five folds of each type into `predictions_all_folds_EMPRES_{type}.csv`. It reports MSE and Pearson r overall, per fold and per species, compares the types on the same genes, and writes all metrics to `cv_metrics_EMPRES_<types>[_with_PhytoExpr].tsv`.
   - With the PhytoExpr transformer B/C files, all models are compared on the genes they share, merged on (gene, species). `--phyto_pred_col` and `--phyto_pred_transform` describe those files.
   - `--use_saved` reads the existing `predictions_all_folds` files instead of the per-fold files.

`utils_PC_a2z.py` is the shared library for all `Scripts/` Python files: the datasets, the model classes, `EMPRES_CONFIG` (input files and model class per type) and the evaluation helpers.

### 3. SIEVE population (*B. distachyon* BD21-3 mutants)

**Sequences and EMPRES 1–4 predictions (`Data_processing/`)**
1. `python make.bd.data.py` — builds `data.bd.csv` from the BD21-3 genome (FASTA), the gene annotation (GFF3), the mutant-line SNP calls and `gene.id.translation.tsv`. It has one row per distinct TSS/TTS sequence (haplotype) of each transcript; heterozygous calls are kept as IUPAC codes, and the lines carrying the sequence are listed, separated by spaces, in `group_for_cross_validation`.
2. `python make.bdi.data.py` — writes `data.bdi.csv`, a copy of `data.bd.csv` in which the gene column holds the Bd21 ortholog ID and the species column is `Bdi`, so the rows look like PhytoExpr's Bdi training rows. It was used as input for the PhytoExpr predictions on the SIEVE population.
3. `python make.bd.sequences.py` — cuts the `data.bd.csv` sequences into model-sized chunks (`bd.sequences.caduceus.tsv`, `bd.sequences.a2z.tsv`).
4. `python make.bd.embedding.caduceus.py` — computes the PlantCaduceus embeddings of those chunks into `embeddings.bd.caduceus.h5`.
5. `python make.bd.embedding.a2z.py` — computes the a2z embeddings and predictions into `embeddings.bd.a2z.h5`.
6. `python generate_predictions_hd5.py` — predicts every SIEVE sequence with the top-5 EMPRES 1–4 models of the fold that held out the gene's Bd21 ortholog. Writes `predictions_{none,pred,emb,a2z}.tsv` (EMPRES 1, 2, 3, 4), with one row per line ID and transcript.

**EMPRES 0 predictions and validation (`Scripts/`)**

7. `python SIEVE_OHE_seqs_efficient.py --data_file data.bd.csv --out_dir <sieve_OHE> [--chunk_size 10000]` — cuts TSS = promoter[1000:6000] and TTS = terminator[4000:9000] from every `data.bd.csv` sequence and one-hot encodes them. Writes `tss_OHE.npy`/`tts_OHE.npy` (N × 4 × 5000, uint8), `tss_sequences.npy`/`tts_sequences.npy`, `sieve_metadata.csv` (linked to the `.npy` rows by `array_index`) and a length summary.
8. `python generate_predictions_SIEVE.py --data_dir <sieve_OHE> --out_dir <models> --save_dir <results> --phyto_data_csv <data.csv> --translation_file <gene.id.translation.tsv> --top_x 5 --EMPRES_type 0` — predicts every SIEVE sequence with the top-5 models of the fold that held out its Bd21 ortholog, using the same assignment as step 6. Writes `predictions_EMPRES_0.tsv` in the same format as step 6.
   - `--phyto_data_csv` must be the training `data.csv`, which supplies the CV groups.
   - EMPRES 1–4 would also run here (`--global_stats_dir <npy_files>`), but no SIEVE embeddings in `.npy` form are produced in this repository.
9. `Rscript 1_Validating_predictions.R` — compares the PhytoExpr and EMPRES SIEVE predictions with measured expression in the population: control lines, and mutant-minus-control differences adjusted for PEER factors and principal components. Input paths and model files are set at the top of the script; EMPRES 0 has to be added to `empres_files` and `model_names`.

### 4. Test species (10 species not used in training)

**EMPRES 1–4 predictions (`Data_processing/`)**
1. `python prepare.sequences.review.py` — cuts the TSS/TTS sequences in `data.review.csv` into model-sized chunks per species (`caduceus.sequences.<species>.tsv`, `a2z.sequences.<species>.tsv`).
2. `python make.review.embedding.caduceus.py` — computes the PlantCaduceus embeddings per species into `embeddings.<species>.caduceus.h5`.
3. `python make.review.embedding.a2z.py` — computes the a2z embeddings and predictions per species into `embeddings.<species>.a2z.h5`.
4. `python generate_review_predictions_hd5.py` — predicts every test-species sequence with the top-5 models of all five folds. Writes `predictions_<species>_{none,pred,emb,a2z}.tsv`, with 5 rows per sequence (one per fold) keyed by `gene-transcript`.
   - The key is stored in 40 bytes, so longer keys (Vigna unguiculata) are shortened to their first 40 bytes.

**EMPRES 0 predictions and validation (`Scripts/`)**

5. `python Test_species_OHE_seqs_efficient.py --data_file data.review.csv --out_dir <test_OHE> [--chunk_size 10000] [--expected_rows N]` — cuts and one-hot encodes the TSS/TTS windows exactly as in SIEVE step 7. Writes `tss_OHE.npy`/`tts_OHE.npy`, `tss_sequences.npy`/`tts_sequences.npy`, `clean_metadata_exact_10000_original_5000_extracted.csv` (sorted by `array_index`) and per-species reports.
6. `python generate_predictions_new_test_species.py --data_dir <test_OHE> --out_dir <models> --save_dir <results> --phyto_pred_file <reviews_pred_out_modelBC.csv> --top_x 5 --EMPRES_type 0 [--join_only]` — predicts every test-species sequence with the top-5 models of each fold. Writes one ensemble column per fold to `EMPRES_{type}_pred_new_test_species.csv`, then joins it onto the PhytoExpr test-species predictions on (gene, species, transcript) as `..._with_PhytoExpr.csv`.
   - `--join_only` redoes only the join.
   - `generate_predictions_new_test_species_from_Hai.py` takes the same options but writes only the standalone file (all metadata columns kept). It was written for last year's PhytoExpr-derived rows, which could not be joined safely.
7. `python validate_predictions_new_test_species.py --save_dir <results> --EMPRES_type 0 [--joined_file F] [--out_file F]` — scores PhytoExpr B, PhytoExpr C and EMPRES on the joined file (MSE and Pearson r against log10(1 + median_TPM)). It does this per fold and for the mean of the five folds, overall and per species, and writes `validation_test_species_EMPRES_{type}.csv`.

## Notes

- The `Scripts/` training, testing and prediction scripts load models from the `.pth` checkpoints. The TorchScript `.pt` files written during training are not used.
- PlantCaduceus embeddings need a CUDA GPU. EMPRES training and prediction are run on one GPU; the scripts fall back to CPU, which is practical only for small tests.
- On clusters, the prediction scripts run much faster when the input `.npy` files are first copied to the node's local disk: they read the test rows one at a time, once per top model.

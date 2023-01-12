# few-shot-leaf-segmentation

Supporting information for "Few-Shot Learning Enables Population-Scale Analysis of Leaf Traits in *Populus trichocarpa*"

https://user-images.githubusercontent.com/37625677/212132074-6a5a1ffc-51a7-4a86-9eb4-2213631e4783.mov

**Manuscript:**

[DOI link when available]

**Description:**

In this work, we use few-shot learning to segment the body and vein architecture of *P. trichocarpa* leaves from high-resolution scans obtained in the UC Davis common garden. Leaf and vein segmentation are formulated as separate tasks, in which convolutional neural networks (CNNs) are used to iteratively expand partial segmentations until reaching stopping criteria. Our leaf and vein segmentation approaches use just 50 and 8 manually traced images for training, respectively, and are applied to a set of 2,634 top and bottom leaf scans. We show that both methods achieve high segmentation accuracy, in some cases exceeding even human-level segmentation. The leaf and vein segmentations are subsequently used to extract 68 morphological traits using traditional open-source image processing tools, which are validated using real-world physical measurements. For a biological perspective, we perform a genome-wide association study using the "vein density" trait to discover novel genetic architectures associated with multiple physiological processes relating to leaf development and function. In addition to sharing all of the few-shot learning code, we are releasing all images, manual segmentations, model predictions, 68 extracted leaf phenotypes, and a new set of SNPs called against the v4 *P. trichocarpa* genome for 1,419 genotypes.

Directories:

    Few-shot learning for p. trichocarpa leaf traits
    ├── data
    │   ├── genomes
    │   │   ├── Ptri_V4_Nisq1.1419.VQSRrecal9_tranche99_allchr.biallelicSNP.geno015.hwe1e50.maf05.mind015.bed
    │   │   ├── Ptri_V4_Nisq1.1419.VQSRrecal9_tranche99_allchr.biallelicSNP.geno015.hwe1e50.maf05.mind015.bim
    │   │   ├── Ptri_V4_Nisq1.1419.VQSRrecal9_tranche99_allchr.biallelicSNP.geno015.hwe1e50.maf05.mind015.fam
    │   │   └── README.txt
    │   ├── images
    │   │   └── *.jpeg
    │   ├── leaf_masks
    │   │   └── *.png
    │   ├── leaf_preds
    │   │   └── *.png
    │   ├── results
    │   │   ├── digital_traits.tsv
    │   │   ├── gwas_results.csv
    │   │   ├── manual_traits.tsv
    │   │   ├── vein_density_blups.tsv
    │   │   └── vein_density_tps_adj.tsv
    │   ├── vein_masks
    │   │   └── *.png
    │   ├── vein_preds
    │   │   └── *.png
    │   └── vein_probs
    │      └── *.png
    ├── figures
    │   └── *.png
    ├── logs
    │   ├── leaf_tracer_256.txt
    │   └── vein_grower_128.txt
    ├── models
    │   ├── BuildCNN.py
    │   ├── LeafTracer.py
    │   └── VeinGrower.py
    ├── notebooks
    │   ├── Figures.ipynb
    │   ├── GrowerInference.ipynb
    │   ├── GrowerTraining.ipynb
    │   ├── TracerInference.ipynb
    │   └── TracerTraining.ipynb
    ├── utils
    │   ├── GetLowestGPU.py
    │   ├── ImageLoader.py
    │   ├── LeafGenerator.py
    │   ├── ModelWrapperGenerator.py
    │   ├── TimeRemaining.py
    │   ├── TraceInitializer.py
    │   └── VeinGenerator.py
    └── weights
    ├── leaf_tracer_256_best_val_model.save
    └── vein_grower_128_best_val_model.save

**Data:**

The `data` folder includes all images, ground truth segmentations, predicted segmentations, and extracted leaf traits. All images encode the sample ID in the file name by indicating the treatment, block, row, position, and leaf side, respectively. For example, the file, `C_1_1_2_bot.jpeg`, indicates the control treatment, block 1, row 1, position 2, and the bottom side of the leaf. Tabulated results include position IDs as well as the corresponding genotype IDs.

- The `images` folder includes the 2,906 high-resolution leaf scans taken in the field. 
- The `leaf_masks` folder includes 50 ground truth segmentations used for training the leaf tracing algorithm. 
- The `leaf_preds` folder includes the 2,906 predicted segmentations from the leaf tracing algorithm. 
- The `vein_masks` folder includes 8 ground truth segmentations used for training the vein growing algorithm.
- The `vein_preds` folder includes the 1,453 predicted segmentations from the vein growing algorithm.
- The `vein_probs` folder includes the 1,453 predicted probability maps from the vein growing algorithm before thresholding.
- The `genomes` folder includes the set of SNPs called against the v4 *P. trichocarpa* genome for 1,419 genotypes with a README file detailing the steps taken. 
- The `results` folder includes 
  - Raw values of the 68 predicted leaf traits in `digital_traits.tsv`
  - Manually measured values of petiole length and width in `manual_traits.tsv`
  - Thin plate spline (TPS) adjusted values of the vein density trait in `vein_density_tps_adj.tsv`
  - Best linear unbiased prediction (BLUP) adjusted values of the vein density trait in `vein_density_blups.tsv`
  - GWAS results for the vein density trait, including chromosome positions and corresponding P values, in `gwas_results.csv`

**Figures:**

The `figures` folder includes all figures and videos used in the manuscript. See `notebooks/Figures.ipynb` for the methods used to generate these figures.

**Logs:**

The `logs` folder includes logs of CNN convergence for the training and validation sets during model training for the leaf tracing CNN and vein growing CNN. The file names include the size of the input window for each method (e.g., 128 for the vein growing CNN).

**Models:**

The `models` folder includes the CNN implementations in PyTorch as well as the leaf tracing and vein growing algorithms at inference time.

- `BuildCNN.py` defines the CNN architecture for leaf tracing or vein growing, with user-specified input shape, output shape, layers, and output activation functions.
- `LeafTracer.py` defines the leaf tracing algorithm at inference time.
- `VeinGrower.py` defines the vein growing algorithm at inference time.

**Notebooks:**

The `notebooks` folder includes jupyter notebooks used for model training, model inference, and figure generation.

- `TracerTraining.ipynb` is used to train the leaf tracing CNN.
- `TracerInference.ipynb` is used to apply the leaf tracing algorithm to the 2,906 leaf top and bottom images.
- `GrowerTraining.ipynb` is used to train the vein growing CNN.
- `GrowerInference.ipynb` is used to apply the vein growing algorithm to the 1,453 leaf bottom images.
- `Figures.ipynb` is used to generate all of the manuscript figures.

**Utils:**

The `utils` folder includes utility scripts implemented in Python that assist in model training and inference. 

- `ImageLoader.py` loads image/mask pairs for sampling training/validation tiles.
- `LeafGenerator.py` generates inputs/outputs for the leaf tracing CNN.
- `VeinGenerator.py` generates inputs/outputs for the vein growing CNN.
- `GetLowestGPU.py` identifies available GPUs using the `nvidia-smi` command and selects the one with lowest memory usage, if none available the device is set to CPU.
- `ModelWrapperGenerator.py` wraps the PyTorch CNN and data loaders with similar functionality to the Keras Model class in TensorFlow (e.g., model.fit(...)).
- `TimeRemaining.py` is used by the model wrapper to estimate remaining time left per epoch.
- `TraceInitializer.py` is used by the tracing algorithm at inference time to initialize the leaf trace using automatic thresholding.

**Weights:**

The `weights` folder includes the CNN parameters from the epoch resulting in the best validation error. The file names include the size of the input window for each method (e.g., 128 for the vein growing CNN). The weights are loaded into the CNN models for inference.

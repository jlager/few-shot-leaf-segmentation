# Few-shot learning for leaf and vein segmentation

**Manuscript:**

https://doi.org/10.48550/arXiv.2301.10351

**Leaf segmentation**

https://user-images.githubusercontent.com/37625677/212694490-45f31320-7c2f-4720-8e4b-dd868200a0a4.mov

**Vein segmentation**

https://user-images.githubusercontent.com/37625677/212132074-6a5a1ffc-51a7-4a86-9eb4-2213631e4783.mov 

**Description**

In this work, we use few-shot learning to segment the body and vein architecture of *P. trichocarpa* leaves from high-resolution scans obtained in the UC Davis common garden. Leaf and vein segmentation are formulated as separate tasks, in which convolutional neural networks (CNNs) are used to iteratively expand partial segmentations until reaching stopping criteria. Our leaf and vein segmentation approaches use just 50 and 8 manually traced images for training, respectively, and are applied to a set of 2,634 top and bottom leaf scans. We show that both methods achieve high segmentation accuracy and retain biologically realistic features. The leaf and vein segmentations are compared against a U-Net baseline model and subsequently used to extract 68 morphological traits using traditional open-source image processing tools, which are validated using real-world physical measurements. For a biological perspective, we perform a genome-wide association study using the "vein density" trait to discover novel genetic architectures associated with multiple physiological processes relating to leaf development and function. In addition to sharing all of the few-shot learning code, we are releasing all images, manual segmentations, model predictions, 68 extracted leaf phenotypes, and a new set of SNPs called against the v4 *P. trichocarpa* genome for 1,419 genotypes.

**Directories:**

    Few-shot learning for p. trichocarpa leaf traits
    ├── data
    │   ├── images
    │   │   └── *.jpeg
    │   ├── leaf_masks
    │   │   └── *.png
    │   ├── leaf_preds
    │   │   └── *.png
    │   ├── leaf_unet_preds
    │   │   └── *.png
    │   ├── results
    │   │   ├── digital_traits.tsv
    │   │   ├── gwas_results.csv
    │   │   ├── manual_traits.tsv
    │   │   ├── vein_density_blups.tsv
    │   │   └── vein_density_tps_adj.tsv
    │   ├── vein_bce_preds
    │   │   └── *.png
    │   ├── vein_bce_probs
    │   │   └── *.png
    │   ├── vein_fl_preds
    │   │   └── *.png
    │   ├── vein_fl_probs
    │   │   └── *.png
    │   ├── vein_masks
    │   │   └── *.png
    │   ├── vein_unet_bce_preds
    │   │   └── *.png
    │   ├── vein_unet_bce_probs
    │   │   └── *.png
    │   ├── vein_unet_fl_preds
    │   │   └── *.png
    │   ├── vein_unet_fl_probs
    │   │   └── *.png
    ├── figures
    │   └── *.png
    ├── logs
    │   ├── leaf_tracer_256.txt
    │   ├── leaf_unet_256.txt
    │   ├── vein_grower_bce_128.txt
    │   ├── vein_grower_fl_128.txt
    │   ├── vein_unet_bce_128.txt
    │   └── vein_unet_fl_128.txt
    ├── models
    │   ├── BuildCNN.py
    │   ├── BuildUNet.py
    │   ├── LeafTracer.py
    │   └── VeinGrower.py
    ├── notebooks
    │   ├── Figures.ipynb
    │   ├── GrowerInference.ipynb
    │   ├── GrowerTraining.ipynb
    │   ├── TracerInference.ipynb
    │   ├── TracerTraining.ipynb
    │   ├── UNetLeafSegmentation.ipynb
    │   └── UNetVeinSegmentation.ipynb
    ├── utils
    │   ├── GetLowestGPU.py
    │   ├── ImageLoader.py
    │   ├── LeafGenerator.py
    │   ├── ModelWrapperGenerator.py
    │   ├── TimeRemaining.py
    │   ├── TraceInitializer.py
    │   ├── UNetTileGenerator.py
    │   └── VeinGenerator.py
    └── weights
        ├── leaf_tracer_256_best_val_model.save
        ├── leaf_unet_256_best_val_model.save
        ├── vein_grower_bce_128_best_val_model.save
        ├── vein_grower_fl_128_best_val_model.save
        ├── vein_unet_bce_128_best_val_model.save
        └── vein_unet_fl_128_best_val_model.save

**Data:**

The `data` folder includes all images, ground truth segmentations, predicted segmentations, and extracted leaf traits. All images encode the sample ID in the file name by indicating the treatment, block, row, position, and leaf side, respectively. For example, the file, `C_1_1_2_bot.jpeg`, indicates the control treatment, block 1, row 1, position 2, and the bottom side of the leaf. Tabulated results include position IDs as well as the corresponding genotype IDs.

- The `images` folder includes the 2,906 high-resolution leaf scans taken in the field. 
- The `leaf_masks` folder includes 50 ground truth segmentations used for training the leaf tracing algorithm. 
- The `leaf_preds` folder includes the 2,906 predicted segmentations from the leaf tracing algorithm. 
- The `leaf_unet_preds` folder includes the 2,906 predicted segmentations from the U-Net model for leaf segmentation. 
- The `vein_masks` folder includes 8 ground truth segmentations used for training the vein growing algorithm.
- The `vein_*_preds` folder includes the 1,453 predicted segmentations from the vein growing algorithm, where * specifies the loss function (bce: binary cross-entropy, fl: focal loss).
- The `vein_*_probs` folder includes the 1,453 predicted probability maps from the vein growing algorithm before thresholding, where * specifies the loss function (bce: binary cross-entropy, fl: focal loss).
- The `vein_unet_*_preds` folder includes the 1,453 predicted segmentations from the U-Net model for vein segmentation, where * specifies the loss function (bce: binary cross-entropy, fl: focal loss).
- The `vein_unet_*_probs` folder includes the 1,453 predicted probability maps from the U-Net model for vein segmentation before thresholding, where * specifies the loss function (bce: binary cross-entropy, fl: focal loss).
- The `genomes` folder includes the set of SNPs called against the v4 *P. trichocarpa* genome for 1,419 genotypes with a README file detailing the steps taken. Note, this folder is not included in this repository, but can be accessed using the Oak Ridge National Laboratory (ORNL) [Constellation Portal](https://doi.org/10.13139/ORNLNCCS/1908723).
- The `results` folder includes 
  - Raw values of the 68 predicted leaf traits in `digital_traits.tsv`
  - Manually measured values of petiole length and width in `manual_traits.tsv`
  - Thin plate spline (TPS) adjusted values of the vein density trait in `vein_density_tps_adj.tsv`
  - Best linear unbiased prediction (BLUP) adjusted values of the vein density trait in `vein_density_blups.tsv`
  - GWAS results for the vein density trait, including chromosome positions and corresponding P values, in `gwas_results.csv`

**Figures:**

The `figures` folder includes all figures and videos used in the manuscript. See `notebooks/Figures.ipynb` for the methods used to generate these figures.

**Logs:**

The `logs` folder includes logs of CNN convergence for the training and validation sets during model training for the leaf tracing CNN vein growing CNN, and U-Net models. The file names include the model, loss function (bce: binary cross-entropy, fl: focal loss), and size of the input window for each method (e.g., 128 for the vein growing CNN).

**Models:**

The `models` folder includes the CNN implementations in PyTorch as well as the leaf tracing and vein growing algorithms at inference time.

- `BuildCNN.py` defines the CNN architecture for leaf tracing or vein growing, with user-specified input shape, output shape, layers, and output activation functions.
- `BuildUNet.py` defines the U-Net architecture for leaf and vein segmentation, with user-specified input/output shape, layers, and output activation functions.
- `LeafTracer.py` defines the leaf tracing algorithm at inference time.
- `VeinGrower.py` defines the vein growing algorithm at inference time.

**Notebooks:**

The `notebooks` folder includes jupyter notebooks used for model training, model inference, and figure generation.

- `Figures.ipynb` is used to generate all of the manuscript figures.
- `GrowerTraining.ipynb` is used to train the vein growing CNN.
- `GrowerInference.ipynb` is used to apply the vein growing algorithm to the 1,453 leaf bottom images.
- `TracerTraining.ipynb` is used to train the leaf tracing CNN.
- `TracerInference.ipynb` is used to apply the leaf tracing algorithm to the 2,906 leaf top and bottom images.
- `UNetLeafSegmentation.ipynb` is used to train and apply U-Net for leaf segmentation.
- `UNetVeinSegmentation.ipynb` is used to train and apply U-Net for vein segmentation.

**Utils:**

The `utils` folder includes utility scripts implemented in Python that assist in model training and inference. 

- `ImageLoader.py` loads image/mask pairs for sampling training/validation tiles.
- `LeafGenerator.py` generates inputs/outputs for the leaf tracing CNN.
- `VeinGenerator.py` generates inputs/outputs for the vein growing CNN.
- `UNetTileGenerator.py` generates inputs/outputs for the U-Net model.
- `GetLowestGPU.py` identifies available GPUs using the `nvidia-smi` command and selects the one with lowest memory usage, if none available the device is set to CPU.
- `ModelWrapperGenerator.py` wraps the PyTorch CNN and data loaders with similar functionality to the Keras Model class in TensorFlow (e.g., model.fit(...)).
- `TimeRemaining.py` is used by the model wrapper to estimate remaining time left per epoch.
- `TraceInitializer.py` is used by the tracing algorithm at inference time to initialize the leaf trace using automatic thresholding.

**Weights:**

The `weights` folder includes the CNN parameters from the epoch resulting in the best validation error. The file names include the model, loss function (bce: binary cross-entropy, fl: focal loss), and size of the input window for each method (e.g., 128 for the vein growing CNN). The weights are loaded into the CNN models for inference.

**Citation:**

    @misc{https://doi.org/10.48550/arxiv.2301.10351,
      doi = {10.48550/ARXIV.2301.10351},
      url = {https://arxiv.org/abs/2301.10351},
      author = {Lagergren, John and Pavicic, Mirko and Chhetri, Hari B. and York, Larry M. and Hyatt, P. Doug and Kainer, David and Rutter, Erica M. and Flores, Kevin and Bailey-Bale, Jack and Klein, Marie and Taylor, Gail and Jacobson, Daniel and Streich, Jared},
      keywords = {Few-shot learning, Image-based plant phenotyping, Genomic analysis},
      title = {Few-Shot Learning Enables Population-Scale Analysis of Leaf Traits in Populus trichocarpa},
      publisher = {arXiv},
      year = {2023},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }

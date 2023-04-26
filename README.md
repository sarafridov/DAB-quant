# DAB-quant

This repository contains the DAB-quant code for use in quantifying DAB immunohistochemical (IHC) staining in tissue samples. In a nutshell, DAB-quant measures the fraction of tissue area that is stained brown, when all tissue is also stained with hematoxylin (light blue). It does so using a threshold determined by control slides, which contain only blue staining (but no brown). DAB-quant can process slides in full or by sampling random regions, to measure staining heterogeneity and allow excluding any problematic portions of a sample. Here we provide usage instructions.

This repository is accompanied by the following paper:
https://pubmed.ncbi.nlm.nih.gov/35857792/

## Preliminaries
(skip these if you already have git and conda)
1. Install git (downloads are available from https://git-scm.com/downloads)
2. Install miniconda (downloads are available from https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

## One-time setup
1. Open a command line (on Mac this is called Terminal; on Windows it’s called Command Prompt)
2. Navigate to the directory where you want to set up IHC slide quantification. On Mac/Linux you can navigate by using `ls` to list the contents of the current directory, `cd directory_name` to enter a so-named directory, and `cd ..` to go up one level in the file structure. On Windows you can navigate the same way, except with the command `ls` replaced by `dir`. On either system, you can use the tab key to autocomplete a directory or file name.
3. Once you are in the desired location, run `git clone https://github.com/sarafridov/DAB-quant.git`. This will create a folder called DAB-quant, with the quantification code inside it. Enter this folder with the command `cd DAB-quant`.
4. Run the command `conda create -n py38dabquant python=3.8 -y` to create a python environment. You can name the environment whatever you want; here we use py38dabquant.
5. Once the environment is created, activate it by running the command `conda activate py38dabquant` (or replace py38dabquant with your own environment name). You should see (py38dabquant) appear at the beginning of the command prompt to indicate the environment is active.
6. Install the necessary packages by running the command `pip install -r requirements.txt` from inside the DAB-quant directory.
7. Finally, we need to install the OpenSlide package to read IHC slides. Here, the instructions are slightly different on Mac vs. Windows.
  * Mac: If you don’t have Homebrew already, install it (follow the instructions at https://brew.sh/). Then, run the command `brew install openslide` (while the py38dabquant environment is active).
  * Windows: Download OpenSlide for Windows from https://openslide.org/download/ (slide quantification has been tested with the 64-bit Windows binary dated 2017-11-22). Unzip the downloaded folder, move it into the DAB-quant directory, and rename it (the unzipped folder) openslide.

## Example slides
You can download example control and test slides from
https://drive.google.com/drive/folders/1tBItwTxydE3DReZvWReEUr9VWT1iRnEw

## Each-time usage
1. Open Terminal (for Mac) or Command Prompt (for Windows) and navigate to the DAB-quant directory.
2. Activate the python environment with `conda activate py38dabquant` (or the name of your python environment).
3. Place all the slides you wish to quantify in a directory test_slides (or whatever name you like) inside DAB-quant.
4. Place the control slides (with only background blue staining, no brown staining) in a separate directory control_slides (or whatever name you like) inside DAB-quant (or skip this step if you prefer to set quantification thresholds manually).
5. Quantify the brown staining with the command `python quantify.py --folder=test_slides --controls_folder=control_slides [any other options you want]` where the folder names denote where you placed the slides you wish to quantify and the control slides, respectively. This command will process all the slides in the test_slides directory using all the control images in the control_slides directory. 

### Options
Brackets denote a value to be specified; if an option is not passed then the default setting is used.  
`--reprocess_controls`: re-process all control slides (default is to only process new control slides)  
`--reprocess`: re-process all test slides (default is to only process new test slides). Note that if you want to reprocess only a specific slide, you can just delete its results folder and re-run the original command without this option.  
`--rmb_precision=[number]`: quantization precision for normalized red minus blue values (default=0.01)  
`--gray_precision=[number]`: quantization precision for grayscale brightness (default=1, brightness ranges from 0 to 255)  
`--num_regions=[number]`: number of slide regions to sample (default=10)  
`--region_size=[number]`: select square regions with this many pixels on a side (default=500)  
`--include_full`: process the full slide (excluding whitespace) in addition to sample regions  
`--show_processed_images`: visualize the sample regions in pseudocolor to visualize stain classification (pink) and background classification (green)  
`--stain_threshold=[number]`: how high the normalized red minus blue value must be for a pixel to be considered stained (default=0.2, but if controls_folder is provided, stain_threshold is ignored and overridden by the threshold computed from the control slides)  
`--include_background`: include near-white pixels that are intermixed with cells in the sample regions (whitespace is excluded by default)  
`--background_threshold=[number]`: average brightness must be at least this value (out of 255) to be considered background whitespace (default=220, but if controls_folder is provided, background_threshold is ignored and overridden by the threshold computed from the control slides)  
`--error_tolerance=[number]`: what fraction of control pixels to tolerate being misclassified as stained (default=0.001, or one tenth of one percent). Used in conjunction with control slides to automatically choose stain_threshold.  
`--background_fraction=[number]`: what fraction of background (whitespace) to tolerate in a sample region (default=0.5)  
`--seed=[number]`: random seed (default=0); choose different numbers to randomize sampled regions, or leave fixed for reproducibility  
`--ext=[extension]`: filename extension for your slides, including the dot (default=.ndpi). Note that slides must be compatible with OpenSlide (see https://openslide.org/formats/), and each file must contain a single slide. 

### Excluding slide artifacts
Sometimes, a slide contains regions that should not be included in the quantification results (for example, if a piece of tissue is ripped or folded). After a slide is processed, a thumbnail image of the entire slide is produced, showing the locations and numeric IDs of each of the sampled regions. If any regions should be excluded, simply add their IDs to the initially empty spreadsheet exclude_regions.csv that is produced in the test slide folder (this can be edited in Excel or any text editor). Then re-process the slide(s) and these regions will be avoided.

### Output
The test slide folder will contain a file stained_fractions.txt, which summarizes the fraction of pixels classified as stained in each region of each slide (and the full slide, if `--include_full`).  

For each slide_name.ndpi (or other extension) file, a folder slide_name is created (also inside the test images folder) with the following files:  
* thumbnail.png: A low-resolution image of the entire slide, overlaid with boxes showing the location of each numbered region.  
* stained_fractions.txt: A two-column table where the first column is the region ID (with 0 denoting the entire image, included only if `--include_full`) and the second column is the fraction of pixels (excluding whitespace unless `--include_whitespace`) whose normalized red - blue color is above stain_threshold.  
* histogram.png: A histogram of normalized red - blue color for each region, all regions combined, and the full image (if `--include_full`)  
* all_regions.txt: The raw histogram data showing the proportion of pixels in all regions combined (excluding whitespace unless --include_whitespace) with each level of normalized red - blue color (from -3 to 3, in bins of width 0.01).  
* full.txt: The raw histogram data showing the proportion of pixels in the entire image (excluding whitespace) with each level of normalized red - blue color (from -3 to 3, in bins of width 0.01). Only produced if `--include_full` 

For each numbered region:
* [region_ID].png: An image of the region.  
* [region_ID]_rmb_normalized.png: A processed image of the region (produced only if `--show_processed_images`) in which grayscale brightness denotes level of staining, whitespace is tinted green, and stained pixels (above stain_threshold) are tinted pink.  
* [region_ID].txt: The raw histogram data showing the proportion of pixels in the region (excluding whitespace unless `--include_whitespace`) with each level of normalized red - blue color (from -3 to 3, in bins of width 0.01).  


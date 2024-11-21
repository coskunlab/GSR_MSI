# GSR_MSI
This folder contains the scripts to reproduce the result in the paper "Super Resolved Single-Cell Spatial Metabolomics from Multimodal Mass Spectrometry Imaging guided by Imaging Mass Cytometry".

Mass spectrometry imaging (MSI) is a powerful technique for spatially resolved analysis of metabolites and other biomolecules within biological tissues. However, the inherent low spatial resolution of MSI often limits its ability to obtain sub-cellular-level information. To address this limitation, we propose a guided super-resolution (GSR) approach that leverages Imaging Mass Cytometry (IMC) images to enhance the spatial granularity of MSI data. By using the detailed IMC images as guides, we achieve a five-fold increase in resolution, creating super-resolved 112 distinct metabolite maps of 85,762 single cells and 24 cell phenotypes across various colorectal cancer tumor samples. This enhancement facilitates more precise analysis of cellular structures and tissue architectures, providing deeper insights into tissue heterogeneity and cellular interactions within the tumor microenvironment through high-definition spatial metabolomics in individual cells.

![Alt text](figures/Picture1.png)

Workflow for GSR in high-resolution IMC and MALDI MSI for colorectal cancer analysis. 
(a) Acquisition of low-resolution MSI data via MALDI and high-resolution IMC images. 
(b) Automated selection of metabolite channels from MSI data, with selected channels (green) showing significant information and non-selected channels (red) being excluded. 
(c) The GSR process integrates spatial coordinates and pixel intensities from the IMC images and produces super-resolution MSI images.
(d) Examples of two GSR outputs are phosphatidylserine (PS 36:1, 788.54 m/z) and phosphatidylinositol (PI 34:2, 833.51 m/z). The top row shows super-resolution and low-resolution images for larger regions, while the bottom row presents the corresponding zoomed-in regions of interest.


## Notebooks 
"notebooks" folder contains jupyter notebook and python scripts used.

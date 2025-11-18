# TMBench Spatial Genomic Simulation Data Software

## Introduction

TMBench is a software tool for generating spatial genomic simulation data. It can create tissue matrices of different shapes and simulate single nucleotide variants (SNVs) and copy number variants (CNVs) to generate genomic data with spatial characteristics.

## Features

- Supports multiple tissue shapes: circle, square or import from image
- Supports multiple mutation patterns: boundary, gradient and nested
- Visualizes tissue matrix and mutation distribution
- Generates simulated sequencing data with spatial barcodes
- Supports simulation of SNV and CNV mutations
- **NEW**: Gene expression matrix integration with CNV-based expression adjustment
- **NEW**: Reference data support for realistic gene expression simulation

## Installation Dependencies

```bash
pip install numpy opencv-python matplotlib pandas
```

## Usage

### Basic Usage

```bash
python TMBench.py [parameters]
```

### Main Parameters

#### Tissue Shape Parameters

- `--rows`: Number of rows in tissue matrix, default is 50
- `--cols`: Number of columns in tissue matrix, default is 50
- `--shape`: Tissue shape, options are 'Circle', 'Square' or 'image' (import from image), default is 'image'
- `--image_path`: When shape is 'image', specify image path, default is './data/img.png'

#### Output Parameters

- `--out_path`: Output path for simulated data, defaults to creating folder with current timestamp
- `--barcode_file`: Spatial barcode file path, default is 'spatial_barcodes_new.txt'

#### Gene Expression Parameters

- `--expression_matrix`: Gene expression matrix file path (CSV format), containing reference gene expression data
- `--gene_annotation`: Gene annotation file path (TSV format), containing gene chromosome positions for CNV integration
- `--noise_level`: Noise level for gene expression adjustment (0-1), default is 0
- `--noise_dispersion`: Noise dispersion parameter, default is 1
- `--spatial_corr`: Spatial correlation strength (-1 to 1), default is 0

#### Mutation Parameters

- `--TMBspatial_snv`: SNV mutation parameters, format as 'name,chromosome,position,reference_base,variant_base,mutation_frequency', e.g.: 'onesnv,chr12,25398284,C,T,0.001'
- `--TMBspatial_cnv`: CNV mutation parameters, multiple CNVs separated by colons
- `--TMBspatial_mutation_mode`: CNV mutation pattern, options are 'boundary', 'gradient' or 'nested', default is 'gradient'
- `--gradient_direction`: Gradient direction, options are 'horizontal', 'vertical' or 'radial', default is 'radial'
- `--snv_mutation_mode`: SNV-specific mutation mode (overrides TMBspatial_mutation_mode)
- `--snv_gradient_direction`: SNV-specific gradient direction (overrides gradient_direction)
- `--cnv_mutation_mode`: CNV-specific mutation mode (overrides TMBspatial_mutation_mode)
- `--cnv_gradient_direction`: CNV-specific gradient direction (overrides gradient_direction)

### Mutation Pattern Description

#### Boundary Mutation Pattern

Set different mutation states in different regions of tissue, separated by boundaries. Supported boundary types:
- Diagonal boundary
- Horizontal boundary
- Vertical boundary

#### Gradient Mutation Pattern

Mutation frequency or copy number gradually changes along specific direction. Supported gradient directions:
- Horizontal gradient: left to right
- Vertical gradient: top to bottom
- Radial gradient: center to outward

#### Nested Mutation Pattern

Set different mutation states in different regions (core region, intermediate region and peripheral region).

## Examples

### Generate circular tissue with radial gradient SNV mutation

```bash
python TMBench.py --shape Circle --rows 50 --cols 50 --TMBspatial_snv "onesnv,chr12,25398284,C,T,0.8" --TMBspatial_mutation_mode gradient --gradient_direction radial
```

### Generate square tissue with horizontal gradient CNV mutation

```bash
python TMBench.py --shape Square --rows 60 --cols 60 --TMBspatial_cnv "cnv1,chr1,100000,200000,6" --TMBspatial_mutation_mode gradient --gradient_direction horizontal
```

### Import tissue shape from image with nested mutation pattern

```bash
python TMBench.py --shape image --image_path "./data/custom_shape.png" --TMBspatial_cnv "cnv1,chr1,100000,200000,4" --TMBspatial_mutation_mode nested
```

### Generate tissue with gene expression matrix and CNV-based expression adjustment

```bash
python TMBench.py --shape Circle --rows 50 --cols 50 --expression_matrix "./data/expression_matrix.csv" --gene_annotation "./data/gene_annotation.txt" --TMBspatial_cnv "cnv1,chr1,100000,200000,6" --noise_level 0.1 --spatial_corr 0.2
```

## Output Files

After running, the program will generate following files in specified output directory (or timestamp-named directory):

- `cell.csv`: CSV file containing all cell information
- `expression_matrix_adjusted.csv`: Gene expression matrix with CNV-based adjustments (if expression matrix provided)
- `original_matrix.png`: Visualization image of original tissue morphology
- `snv_matrix.png`: Visualization image of SNV mutation distribution (if SNV mutation specified)
- `cnv_matrix{i}.png`: Visualization image of CNV mutation distribution (if CNV mutation specified, i is CNV index)
- Simulated sequencing data folders for each cell
- `merge/`: Merged FASTQ files

## Notes

- Barcode file format should be tab-delimited text file, each line contains: barcode, row coordinate, column coordinate
- For SNV mutations, mutation frequency should be between 0-1
- For CNV mutations, copy number can be any positive number, 2 indicates normal copy number
- Expression matrix should be CSV format with genes as columns and cell coordinates as rows (format: "{row}x{col}")
- Gene annotation file should be TSV format with columns: gene_name, chromosome, start_position, end_position
- CNV-based expression adjustment uses copy number ratio (copy_num/2) to scale gene expression levels
- Noise parameters control biological variability: noise_level (0-1), noise_dispersion (>0), spatial_corr (-1 to 1)

## Recent Updates

### Gene Expression Matrix Integration (Latest)
- Added support for reference gene expression matrix input
- Implemented CNV-based gene expression adjustment functionality
- Added noise modeling and spatial correlation for realistic expression patterns
- New output file: `expression_matrix_adjusted.csv` containing CNV-adjusted expression values

## Citation

If you use TMBench in your research, please cite this software.

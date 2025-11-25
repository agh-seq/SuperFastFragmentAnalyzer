# SuperFast Fragment Analyzer

A fast Python package for processing BED files and computing overlap statistics with GTF annotation files using Polars.

## Features

- **Fast BED Processing**: Convert 3-column BED files to Parquet format using Polars
- **Fragment Length Analysis**: Automatically calculates fragment length and classifies into size categories:
  - sub-nucleosome: 0-100 bp
  - mono-nucleosome: 101-250 bp
  - di-nucleosome: 251-420 bp
  - tri-nucleosome: 421+ bp
- **GTF Feature Extraction**: Extract exons, introns, and promoters (±2kb from TSS) from GTF files
- **Overlap Statistics**: Compute overlap statistics between reads and genomic features with fragment length metrics
- **Multi-Genome Support**: Automatically handles human (`chr*`) and pig (`NC_*`) genomes
- **Efficient Processing**: Uses Polars for high-performance data processing

## Installation

```bash
pip install -e .
```

## Usage

### Command-Line Interface

#### Convert BED to Parquet

```bash
superfast-fragment-analyzer bed-to-parquet input.bed -o output.parquet
```

#### Extract Features from GTF

```bash
superfast-fragment-analyzer extract-features annotation.gtf -o features.parquet
```

#### Compute Overlap Statistics

```bash
superfast-fragment-analyzer compute-overlaps reads.bed annotation.gtf -o results/
```

This will generate:
- `overlap_stats_human_summary.tsv` - Human genome statistics with median fragment lengths
- `overlap_stats_pig_summary.tsv` - Pig genome statistics with median fragment lengths
- `overlap_stats_combined_summary.tsv` - Combined statistics with median fragment lengths
- `overlap_stats_*_fragment_stats.tsv` - Fragment length statistics by feature type and size category
- `overlap_stats_*_overlaps.parquet` - Detailed overlap data including fragment_length and size_category

### Python API

```python
from superfast_fragment_analyzer import BedProcessor, GtfProcessor, OverlapStats

# Process BED file
bed_processor = BedProcessor("reads.bed")
bed_df = bed_processor.read_bed()
bed_processor.to_parquet("reads.parquet")

# Process GTF file
gtf_processor = GtfProcessor("annotation.gtf")
features_df = gtf_processor.extract_features()

# Compute overlap statistics
overlap_stats = OverlapStats(bed_df, features_df)
all_stats = overlap_stats.compute_all_statistics()

# Access statistics
human_stats = all_stats["human"]["summary"]
pig_stats = all_stats["pig"]["summary"]
combined_stats = all_stats["combined"]["summary"]
```

## Input Formats

### BED File Format

3-column BED files (tab-separated):
- Column 1: Chromosome name
- Column 2: Start position (0-based)
- Column 3: End position (0-based, exclusive)

Each row represents a single read/fragment.

### GTF File Format

Standard GTF format with the following required attributes:
- `gene_id`: Gene identifier
- `transcript_id`: Transcript identifier (optional)
- `gene_name`: Gene name (optional)

## Output

### Summary Statistics

TSV files with columns:
- `feature_type`: Type of feature (exon, intron, promoter)
- `overlapping_reads`: Number of reads overlapping this feature type
- `total_reads`: Total number of reads
- `percentage`: Percentage of reads overlapping
- `median_fragment_length`: Median fragment length (bp) for overlapping reads

### Fragment Length Statistics

TSV files with columns:
- `feature_type`: Type of feature (exon, intron, promoter)
- `size_category`: Fragment size category (sub-nucleosome, mono-nucleosome, di-nucleosome, tri-nucleosome)
- `count`: Number of overlapping reads in this category
- `median_fragment_length`: Median fragment length (bp) for reads in this category

### Detailed Overlaps

Parquet files with detailed overlap information including:
- Read coordinates (read_start, read_end)
- Fragment length and size category
- Feature coordinates (feature_start, feature_end)
- Feature type and gene information (gene_id, gene_name)

## Genome Classification

The package automatically classifies reads and features by genome:
- **Human**: Chromosomes starting with `chr` (e.g., `chr1`, `chr2`)
- **Pig**: Chromosomes starting with `NC_` (e.g., `NC_010443.5`)

## Requirements

- Python >= 3.8
- polars >= 0.19.0
- pyarrow >= 10.0.0
- click >= 8.0.0

## License

MIT


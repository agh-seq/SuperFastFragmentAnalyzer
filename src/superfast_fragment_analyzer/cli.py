"""Command-line interface for SuperFast Fragment Analyzer."""

import sys
from pathlib import Path
from typing import Optional

import click
import polars as pl

from superfast_fragment_analyzer.bed_processor import BedProcessor
from superfast_fragment_analyzer.gtf_processor import GtfProcessor
from superfast_fragment_analyzer.overlap_stats import OverlapStats


@click.group()
@click.version_option(version="0.1.0")
def main():
    """SuperFast Fragment Analyzer - Fast BED file processing and GTF overlap statistics."""
    pass


@main.command()
@click.argument("bed_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output Parquet file path (default: bed_file.parquet)",
)
@click.option(
    "--compression",
    "-c",
    default="zstd",
    type=click.Choice(["zstd", "snappy", "gzip", "lz4", "brotli"]),
    help="Compression algorithm (default: zstd)",
)
@click.option(
    "--compression-level",
    "-l",
    default=3,
    type=int,
    help="Compression level (default: 3)",
)
def bed_to_parquet(
    bed_file: Path,
    output: Optional[Path],
    compression: str,
    compression_level: int,
):
    """Convert BED file to Parquet format."""
    try:
        processor = BedProcessor(bed_file)
        output_path = processor.to_parquet(
            output_path=output,
            compression=compression,
            compression_level=compression_level,
        )
        click.echo(f"Successfully converted {bed_file} to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("gtf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output Parquet file path (default: gtf_file_features.parquet)",
)
@click.option(
    "--biotype",
    "-b",
    type=str,
    default=None,
    help="Filter by biotype (e.g., 'protein_coding'). If not specified, includes all biotypes.",
)
def extract_features(gtf_file: Path, output: Optional[Path], biotype: Optional[str]):
    """Extract exons, introns, and promoters from GTF file."""
    try:
        processor = GtfProcessor(gtf_file)
        features_df = processor.extract_features(biotype_filter=biotype)
        
        if output is None:
            output = gtf_file.parent / f"{gtf_file.stem}_features.parquet"
        
        features_df.write_parquet(output)
        click.echo(f"Successfully extracted features to {output}")
        click.echo(f"Total features: {len(features_df)}")
        click.echo(f"Feature types: {features_df['feature_type'].value_counts()}")
        if biotype:
            click.echo(f"Filtered by biotype: {biotype}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("bed_file", type=click.Path(exists=True, path_type=Path))
@click.argument("gtf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Output directory for statistics files (default: current directory)",
)
@click.option(
    "--prefix",
    "-p",
    default="overlap_stats",
    help="Prefix for output files (default: overlap_stats)",
)
@click.option(
    "--keep-parquet",
    is_flag=True,
    help="Keep intermediate BED and GTF Parquet files",
)
@click.option(
    "--biotype",
    "-b",
    type=str,
    default=None,
    help="Filter GTF features by biotype (e.g., 'protein_coding'). If not specified, includes all biotypes.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output to diagnose chromosome matching issues",
)
def compute_overlaps(
    bed_file: Path,
    gtf_file: Path,
    output_dir: Path,
    prefix: str,
    keep_parquet: bool,
    biotype: Optional[str],
    debug: bool,
):
    """Compute overlap statistics between BED reads and GTF features."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Convert BED to Parquet
        click.echo(f"Converting BED file to Parquet: {bed_file}")
        bed_parquet = output_dir / f"{prefix}_bed.parquet"
        bed_processor = BedProcessor(bed_file)
        bed_processor.to_parquet(
            output_path=bed_parquet,
            compression="zstd",
            compression_level=3,
        )
        click.echo(f"  Saved to: {bed_parquet}")
        
        # Step 2: Extract GTF features and save to Parquet
        click.echo(f"Processing GTF file and extracting features: {gtf_file}")
        if biotype:
            click.echo(f"  Filtering by biotype: {biotype}")
        gtf_parquet = output_dir / f"{prefix}_gtf_features.parquet"
        gtf_processor = GtfProcessor(gtf_file)
        features_df = gtf_processor.extract_features(biotype_filter=biotype)
        features_df.write_parquet(gtf_parquet, compression="zstd", compression_level=3)
        click.echo(f"  Saved to: {gtf_parquet}")
        click.echo(f"  Extracted {len(features_df)} features")
        
        # Step 3: Compute overlaps from Parquet files using SQL
        click.echo("Computing overlap statistics from Parquet files...")
        overlap_stats = OverlapStats(bed_parquet, gtf_parquet)
        output_files = overlap_stats.save_statistics_with_counts(output_dir, prefix, debug=debug)
        
        click.echo(f"\nStatistics saved to {output_dir}:")
        for stat_type, file_path in output_files.items():
            click.echo(f"  - {stat_type}: {file_path}")
        
        # Clean up intermediate files if not keeping them
        if not keep_parquet:
            bed_parquet.unlink()
            gtf_parquet.unlink()
            click.echo(f"\nCleaned up intermediate Parquet files")
        else:
            click.echo(f"\nIntermediate Parquet files kept:")
            click.echo(f"  - BED: {bed_parquet}")
            click.echo(f"  - GTF features: {gtf_parquet}")
        
        # Print summary
        all_stats = overlap_stats.compute_all_statistics(debug=debug)
        click.echo("\n=== Summary Statistics ===")
        for genome_type in ["human", "pig", "combined"]:
            click.echo(f"\n{genome_type.upper()}:")
            summary = all_stats[genome_type]["summary"]
            for row in summary.iter_rows(named=True):
                median_str = f", median length: {row['median_fragment_length']:.1f} bp" if row['median_fragment_length'] is not None else ""
                click.echo(
                    f"  {row['feature_type']}: "
                    f"{row['overlapping_reads']}/{row['total_reads']} "
                    f"({row['percentage']:.2f}%){median_str}"
                )
            
            # Print fragment length statistics by size category
            fragment_stats = all_stats[genome_type]["fragment_stats"]
            if not fragment_stats.is_empty():
                click.echo(f"\n  Fragment Length Statistics by Size Category:")
                for row in fragment_stats.iter_rows(named=True):
                    median_str = f", median: {row['median_fragment_length']:.1f} bp" if row['median_fragment_length'] is not None else ""
                    click.echo(
                        f"    {row['feature_type']} - {row['size_category']}: "
                        f"{row['count']} reads{median_str}"
                    )
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("overlaps_parquet", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--feature-type",
    "-f",
    type=click.Choice(["exon", "intron", "promoter"], case_sensitive=False),
    default=None,
    help="Feature type to generate counts for (exon, intron, or promoter). If not specified, generates counts for all features.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for count tables (default: same directory as input file)",
)
@click.option(
    "--prefix",
    "-p",
    default=None,
    help="Prefix for output files (default: inferred from input filename)",
)
def generate_counts(
    overlaps_parquet: Path,
    feature_type: Optional[str],
    output_dir: Optional[Path],
    prefix: Optional[str],
):
    """Generate gene count tables from existing overlaps parquet file.
    
    Can generate counts for a specific feature type (exon, intron, promoter)
    or for all features combined.
    """
    try:
        overlaps_parquet = Path(overlaps_parquet)
        
        if not overlaps_parquet.exists():
            click.echo(f"Error: File not found: {overlaps_parquet}", err=True)
            sys.exit(1)
        
        # Determine output directory
        if output_dir is None:
            output_dir = overlaps_parquet.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine prefix from filename if not provided
        if prefix is None:
            # Extract prefix from filename (e.g., "overlap_stats_pig_overlaps.parquet" -> "overlap_stats_pig")
            stem = overlaps_parquet.stem
            if "_overlaps" in stem:
                prefix = stem.replace("_overlaps", "")
            else:
                prefix = stem
        
        click.echo(f"Loading overlaps from: {overlaps_parquet}")
        overlaps_df = pl.read_parquet(overlaps_parquet)
        click.echo(f"  Loaded {len(overlaps_df)} overlap records")
        
        # Generate counts for specified feature type or all features
        if feature_type is None:
            click.echo("Generating gene count table (all features)...")
            counts = OverlapStats.generate_feature_counts(overlaps_df, feature_type=None)
            counts_path = output_dir / f"{prefix}_gene_counts.tsv"
            feature_label = "all features"
        else:
            feature_type_lower = feature_type.lower()
            click.echo(f"Generating {feature_type_lower} count table...")
            counts = OverlapStats.generate_feature_counts(overlaps_df, feature_type=feature_type_lower)
            counts_path = output_dir / f"{prefix}_{feature_type_lower}_counts.tsv"
            feature_label = feature_type_lower
        
        counts.write_csv(counts_path, separator="\t")
        click.echo(f"  Saved to: {counts_path}")
        click.echo(f"  Total genes with {feature_label} overlaps: {len(counts)}")
        if len(counts) > 0:
            click.echo(f"  Top gene: {counts[0, 'gene_name']} ({counts[0, 'read_count']} reads)")
        
        click.echo(f"\nCount table saved to {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("gtf_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--sample-lines",
    "-n",
    default=20,
    type=int,
    help="Number of sample lines to show (default: 20)",
)
def inspect_gtf(gtf_file: Path, sample_lines: int):
    """Inspect GTF file to diagnose chromosome naming and genome classification issues."""
    try:
        gtf_file = Path(gtf_file)
        click.echo(f"Inspecting GTF file: {gtf_file}")
        
        # Read GTF file
        gtf_processor = GtfProcessor(gtf_file)
        gtf_df = gtf_processor.read_gtf()
        
        click.echo(f"\nTotal GTF records: {len(gtf_df)}")
        
        # Check chromosome names (seqname)
        click.echo("\n=== Chromosome Names (seqname) ===")
        seqname_counts = gtf_df["seqname"].value_counts().sort("count", descending=True)
        click.echo(f"Unique chromosomes: {len(seqname_counts)}")
        click.echo("\nTop 20 chromosomes:")
        for row in seqname_counts.head(20).iter_rows(named=True):
            click.echo(f"  {row['seqname']}: {row['count']} records")
        
        # Check genome classification
        click.echo("\n=== Genome Classification ===")
        if "genome" in gtf_df.columns:
            genome_counts = gtf_df["genome"].value_counts()
            click.echo("Genome classification counts:")
            for row in genome_counts.iter_rows(named=True):
                click.echo(f"  {row['genome']}: {row['count']} records")
            
            # Show chromosome names by genome
            click.echo("\nChromosomes by genome type:")
            for genome_type in ["human", "pig", "unknown"]:
                genome_chroms = (
                    gtf_df
                    .filter(pl.col("genome") == genome_type)
                    .select("seqname")
                    .unique()
                    .sort("seqname")
                )
                if len(genome_chroms) > 0:
                    click.echo(f"\n  {genome_type.upper()} ({len(genome_chroms)} chromosomes):")
                    chrom_list = genome_chroms["seqname"].to_list()
                    # Show first 20
                    for chrom in chrom_list[:20]:
                        click.echo(f"    {chrom}")
                    if len(chrom_list) > 20:
                        click.echo(f"    ... and {len(chrom_list) - 20} more")
        else:
            click.echo("  Genome column not found in GTF")
        
        # Check feature types
        click.echo("\n=== Feature Types ===")
        if "feature" in gtf_df.columns:
            feature_counts = gtf_df["feature"].value_counts()
            click.echo("Feature type counts:")
            for row in feature_counts.iter_rows(named=True):
                click.echo(f"  {row['feature']}: {row['count']} records")
        
        # Sample data
        click.echo(f"\n=== Sample Data (first {sample_lines} lines) ===")
        sample_df = gtf_df.head(sample_lines)
        for row in sample_df.iter_rows(named=True):
            seqname = row.get("seqname", "N/A")
            feature = row.get("feature", "N/A")
            start = row.get("start", "N/A")
            end = row.get("end", "N/A")
            genome = row.get("genome", "N/A")
            gene_id = row.get("gene_id", "N/A")
            click.echo(f"  {seqname}\t{feature}\t{start}-{end}\tgenome={genome}\tgene_id={gene_id}")
        
        # Check extracted features
        click.echo("\n=== Extracted Features ===")
        try:
            features_df = gtf_processor.extract_features()
            click.echo(f"Total extracted features: {len(features_df)}")
            if len(features_df) > 0:
                feature_type_counts = features_df["feature_type"].value_counts()
                click.echo("\nFeature type counts in extracted features:")
                for row in feature_type_counts.iter_rows(named=True):
                    click.echo(f"  {row['feature_type']}: {row['count']} features")
                
                if "genome" in features_df.columns:
                    genome_counts = features_df["genome"].value_counts()
                    click.echo("\nGenome counts in extracted features:")
                    for row in genome_counts.iter_rows(named=True):
                        click.echo(f"  {row['genome']}: {row['count']} features")
        except Exception as e:
            click.echo(f"  Error extracting features: {e}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


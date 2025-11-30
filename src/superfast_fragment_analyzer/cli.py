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
    "--genome-filter",
    "-g",
    type=click.Choice(["human", "pig"], case_sensitive=False),
    default=None,
    help="Filter to only process reads from specified genome (human or pig). This significantly speeds up processing for large files. If not specified, processes all reads.",
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
    genome_filter: Optional[str],
    debug: bool,
):
    """Compute overlap statistics between BED reads and GTF features."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Convert BED to Parquet (with genome filter if specified)
        click.echo(f"Converting BED file to Parquet: {bed_file}")
        if genome_filter:
            click.echo(f"  Filtering to {genome_filter} reads only (this will speed up processing!)")
        bed_parquet = output_dir / f"{prefix}_bed.parquet"
        bed_processor = BedProcessor(bed_file)
        bed_processor.to_parquet(
            output_path=bed_parquet,
            compression="zstd",
            compression_level=3,
            genome_filter=genome_filter,
        )
        click.echo(f"  Saved to: {bed_parquet}")
        
        # Step 2: Extract GTF features and save to Parquet
        click.echo(f"Processing GTF file and extracting features: {gtf_file}")
        if biotype:
            click.echo(f"  Filtering by biotype: {biotype}")
        gtf_parquet = output_dir / f"{prefix}_gtf_features.parquet"
        gtf_processor = GtfProcessor(gtf_file)
        features_df = gtf_processor.extract_features(biotype_filter=biotype)
        
        # Also filter GTF features by genome if specified
        if genome_filter:
            features_df = features_df.filter(pl.col("genome") == genome_filter)
            click.echo(f"  Filtered GTF features to {genome_filter} only")
        
        features_df.write_parquet(gtf_parquet, compression="zstd", compression_level=3)
        click.echo(f"  Saved to: {gtf_parquet}")
        click.echo(f"  Extracted {len(features_df)} features")
        
        # Step 3: Compute overlaps from Parquet files
        click.echo("Computing overlap statistics from Parquet files...")
        overlap_stats = OverlapStats(bed_parquet, gtf_parquet)
        
        # If genome_filter is specified, only compute statistics for that genome
        if genome_filter:
            output_files = {}
            all_stats = {}
            stats = overlap_stats.compute_statistics(genome_filter=genome_filter, debug=debug)
            all_stats[genome_filter] = stats
            
            # Save statistics for the filtered genome
            summary_path = output_dir / f"{prefix}_{genome_filter}_summary.tsv"
            stats["summary"].write_csv(summary_path, separator="\t")
            output_files[f"{genome_filter}_summary"] = summary_path
            
            fragment_stats_path = output_dir / f"{prefix}_{genome_filter}_fragment_stats.tsv"
            stats["fragment_stats"].write_csv(fragment_stats_path, separator="\t")
            output_files[f"{genome_filter}_fragment_stats"] = fragment_stats_path
            
            overlaps_path = output_dir / f"{prefix}_{genome_filter}_overlaps.parquet"
            stats["overlaps"].write_parquet(overlaps_path)
            output_files[f"{genome_filter}_overlaps"] = overlaps_path
            
            # Generate and save gene counts
            gene_counts = OverlapStats.generate_gene_counts(stats["overlaps"])
            gene_counts_path = output_dir / f"{prefix}_{genome_filter}_gene_counts.tsv"
            gene_counts.write_csv(gene_counts_path, separator="\t")
            output_files[f"{genome_filter}_gene_counts"] = gene_counts_path
            
            # Generate and save exon counts
            exon_counts = OverlapStats.generate_exon_counts(stats["overlaps"])
            exon_counts_path = output_dir / f"{prefix}_{genome_filter}_exon_counts.tsv"
            exon_counts.write_csv(exon_counts_path, separator="\t")
            output_files[f"{genome_filter}_exon_counts"] = exon_counts_path
        else:
            # Original behavior: compute all statistics
            output_files, all_stats = overlap_stats.save_statistics_with_counts(output_dir, prefix, debug=debug)
        
        click.echo(f"\nStatistics saved to {output_dir}:")
        for stat_type, file_path in output_files.items():
            click.echo(f"  - {stat_type}: {file_path}")
        
        # Print summary using already-computed statistics (no recomputation needed)
        click.echo("\n=== Summary Statistics ===")
        genome_types = [genome_filter] if genome_filter else ["human", "pig", "combined"]
        for genome_type in genome_types:
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
        
        # Clean up intermediate files if not keeping them (after summary is printed)
        if not keep_parquet:
            bed_parquet.unlink()
            gtf_parquet.unlink()
            click.echo(f"\nCleaned up intermediate Parquet files")
        else:
            click.echo(f"\nIntermediate Parquet files kept:")
            click.echo(f"  - BED: {bed_parquet}")
            click.echo(f"  - GTF features: {gtf_parquet}")
        
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
    type=str,
    multiple=True,
    default=None,
    help="Feature type(s) to generate counts for (gene, exon, intron, or promoter). Can be specified multiple times or as comma-separated list. 'gene' counts across all features. If not specified, generates counts for all features. Example: -f gene,exon,intron,promoter or -f gene -f exon",
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
    feature_type: tuple,
    output_dir: Optional[Path],
    prefix: Optional[str],
):
    """Generate gene count tables from existing overlaps parquet file.
    
    Can generate counts for a specific feature type (gene, exon, intron, promoter).
    Each feature type is calculated separately for computational efficiency.
    - 'gene': Counts reads per gene across ALL feature types (exon, intron, promoter)
    - 'exon', 'intron', 'promoter': Counts reads per gene for that specific feature type only
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
        
        # Parse feature types - handle both comma-separated strings and multiple flags
        # Click's multiple=True returns a tuple of strings
        valid_types = {"gene", "exon", "intron", "promoter"}
        feature_types = []
        
        if not feature_type or len(feature_type) == 0:
            feature_types = ["gene"]  # Default to gene if nothing specified
        else:
            # Process each item in the tuple (could be comma-separated or individual)
            for ft_str in feature_type:
                if not ft_str:
                    continue
                # Split on comma and process each
                for ft in ft_str.split(","):
                    ft_clean = ft.strip().lower()
                    if ft_clean in valid_types:
                        if ft_clean not in feature_types:  # Avoid duplicates
                            feature_types.append(ft_clean)
                    else:
                        click.echo(f"Warning: '{ft.strip()}' is not a valid feature type. Valid types: {', '.join(valid_types)}", err=True)
        
        if not feature_types:
            click.echo("Error: No valid feature types specified", err=True)
            sys.exit(1)
        
        click.echo(f"\nGenerating count tables for {len(feature_types)} feature type(s): {', '.join(feature_types)}")
        
        # Generate counts for each feature type separately
        # Each calculation is done independently for computational efficiency
        generated_files = []
        for i, feature_type_lower in enumerate(feature_types, 1):
            try:
                if feature_type_lower == "gene":
                    # Gene counts: count across ALL feature types
                    click.echo(f"\n  [{i}/{len(feature_types)}] Processing 'gene' (across all feature types)...")
                    counts = OverlapStats.generate_gene_counts(overlaps_df)
                    counts_path = output_dir / f"{prefix}_gene_counts.tsv"
                    feature_label = "all features (gene-level)"
                else:
                    # Feature-specific counts: filter by feature type
                    click.echo(f"\n  [{i}/{len(feature_types)}] Processing '{feature_type_lower}' (filtered by {feature_type_lower} only)...")
                    counts = OverlapStats.generate_feature_counts(overlaps_df, feature_type=feature_type_lower)
                    counts_path = output_dir / f"{prefix}_{feature_type_lower}_counts.tsv"
                    feature_label = feature_type_lower
                
                counts.write_csv(counts_path, separator="\t")
                generated_files.append(counts_path)
                click.echo(f"    ✓ Saved to: {counts_path}")
                click.echo(f"    Total genes with {feature_label} overlaps: {len(counts)}")
                if len(counts) > 0:
                    click.echo(f"    Top gene: {counts[0, 'gene_name']} ({counts[0, 'read_count']} reads)")
            except Exception as e:
                click.echo(f"    ✗ Error processing '{feature_type_lower}': {e}", err=True)
                import traceback
                traceback.print_exc()
                continue  # Continue with next feature type
        
        click.echo(f"\n✓ Successfully generated {len(generated_files)} count table(s):")
        for file_path in generated_files:
            click.echo(f"  - {file_path}")
        
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


@main.command()
@click.argument("parquet_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--type",
    "-t",
    type=click.Choice(["bed", "gtf"], case_sensitive=False),
    required=True,
    help="Type of parquet file: 'bed' for BED reads or 'gtf' for GTF features",
)
def inspect_parquet(parquet_file: Path, type: str):
    """Inspect parquet file to check chromosome names and genome classification."""
    try:
        parquet_file = Path(parquet_file)
        click.echo(f"Inspecting {type.upper()} parquet file: {parquet_file}")
        
        df = pl.read_parquet(parquet_file)
        click.echo(f"\nTotal records: {len(df)}")
        click.echo(f"Columns: {', '.join(df.columns)}")
        
        if type.lower() == "bed":
            # BED file inspection
            if "chromosome" not in df.columns:
                click.echo("Error: 'chromosome' column not found", err=True)
                sys.exit(1)
            
            click.echo("\n=== Chromosome Names ===")
            chrom_counts = df["chromosome"].value_counts().sort("count", descending=True)
            click.echo(f"Unique chromosomes: {len(chrom_counts)}")
            click.echo("\nTop 30 chromosomes:")
            for row in chrom_counts.head(30).iter_rows(named=True):
                click.echo(f"  {row['chromosome']}: {row['count']} records")
            
            if "genome" in df.columns:
                click.echo("\n=== Genome Classification ===")
                genome_counts = df["genome"].value_counts()
                for row in genome_counts.iter_rows(named=True):
                    click.echo(f"  {row['genome']}: {row['count']} records")
                
                # Show chromosomes by genome
                for genome_type in ["human", "pig", "unknown"]:
                    genome_chroms = (
                        df
                        .filter(pl.col("genome") == genome_type)
                        .select("chromosome")
                        .unique()
                        .sort("chromosome")
                    )
                    if len(genome_chroms) > 0:
                        click.echo(f"\n  {genome_type.upper()} chromosomes ({len(genome_chroms)}):")
                        chrom_list = genome_chroms["chromosome"].to_list()
                        for chrom in chrom_list[:30]:
                            click.echo(f"    {chrom}")
                        if len(chrom_list) > 30:
                            click.echo(f"    ... and {len(chrom_list) - 30} more")
        
        elif type.lower() == "gtf":
            # GTF features file inspection
            if "seqname" not in df.columns:
                click.echo("Error: 'seqname' column not found", err=True)
                sys.exit(1)
            
            click.echo("\n=== Chromosome Names (seqname) ===")
            chrom_counts = df["seqname"].value_counts().sort("count", descending=True)
            click.echo(f"Unique chromosomes: {len(chrom_counts)}")
            click.echo("\nTop 30 chromosomes:")
            for row in chrom_counts.head(30).iter_rows(named=True):
                click.echo(f"  {row['seqname']}: {row['count']} records")
            
            if "genome" in df.columns:
                click.echo("\n=== Genome Classification ===")
                genome_counts = df["genome"].value_counts()
                for row in genome_counts.iter_rows(named=True):
                    click.echo(f"  {row['genome']}: {row['count']} records")
                
                # Show chromosomes by genome
                for genome_type in ["human", "pig", "unknown"]:
                    genome_chroms = (
                        df
                        .filter(pl.col("genome") == genome_type)
                        .select("seqname")
                        .unique()
                        .sort("seqname")
                    )
                    if len(genome_chroms) > 0:
                        click.echo(f"\n  {genome_type.upper()} chromosomes ({len(genome_chroms)}):")
                        chrom_list = genome_chroms["seqname"].to_list()
                        for chrom in chrom_list[:30]:
                            click.echo(f"    {chrom}")
                        if len(chrom_list) > 30:
                            click.echo(f"    ... and {len(chrom_list) - 30} more")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


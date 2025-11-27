"""Command-line interface for SuperFast Fragment Analyzer."""

import sys
from pathlib import Path
from typing import Optional

import click

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
def compute_overlaps(
    bed_file: Path,
    gtf_file: Path,
    output_dir: Path,
    prefix: str,
    keep_parquet: bool,
    biotype: Optional[str],
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
        output_files = overlap_stats.save_statistics(output_dir, prefix)
        
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
        all_stats = overlap_stats.compute_all_statistics()
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


if __name__ == "__main__":
    main()


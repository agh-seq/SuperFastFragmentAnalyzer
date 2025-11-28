"""Overlap statistics computation between BED reads and GTF features."""

from pathlib import Path
from typing import Dict, Optional, Union

import polars as pl


class OverlapStats:
    """Compute overlap statistics between BED reads and GTF annotations."""
    
    def __init__(
        self,
        bed_source: Union[pl.DataFrame, Path],
        gtf_features_source: Union[pl.DataFrame, Path],
    ):
        """
        Initialize overlap statistics calculator.
        
        Args:
            bed_source: BED reads DataFrame or path to Parquet file
            gtf_features_source: GTF features DataFrame or path to Parquet file
        """
        self.bed_source = bed_source
        self.gtf_features_source = gtf_features_source
        self._bed_df = None
        self._gtf_features_df = None
    
    @property
    def bed_df(self) -> pl.LazyFrame:
        """Lazy load BED DataFrame from Parquet if needed."""
        if self._bed_df is None:
            if isinstance(self.bed_source, Path):
                self._bed_df = pl.scan_parquet(self.bed_source)
            else:
                self._bed_df = self.bed_source.lazy() if isinstance(self.bed_source, pl.DataFrame) else self.bed_source
        return self._bed_df
    
    @property
    def gtf_features_df(self) -> pl.LazyFrame:
        """Lazy load GTF features DataFrame from Parquet if needed."""
        if self._gtf_features_df is None:
            if isinstance(self.gtf_features_source, Path):
                self._gtf_features_df = pl.scan_parquet(self.gtf_features_source)
            else:
                self._gtf_features_df = self.gtf_features_source.lazy() if isinstance(self.gtf_features_source, pl.DataFrame) else self.gtf_features_source
        return self._gtf_features_df
    
    def _compute_overlaps(
        self,
        reads_df: pl.LazyFrame,
        features_df: pl.LazyFrame,
        debug: bool = False,
    ) -> pl.LazyFrame:
        """
        Compute overlaps between reads and features using interval joins.
        
        Args:
            reads_df: Reads LazyFrame
            features_df: Features LazyFrame
            debug: If True, print diagnostic information about chromosome matching
            
        Returns:
            LazyFrame with overlapping reads and their feature types
        """
        # Get chromosomes efficiently without materializing
        read_chromosomes = set(reads_df.select("chromosome").unique().collect()["chromosome"].to_list())
        feature_chromosomes = set(features_df.select("seqname").unique().collect()["seqname"].to_list())
        chromosomes = sorted(read_chromosomes & feature_chromosomes)
        
        if debug:
            import sys
            print(f"DEBUG: Read chromosomes ({len(read_chromosomes)}): {sorted(list(read_chromosomes))[:10]}...", file=sys.stderr)
            print(f"DEBUG: Feature chromosomes ({len(feature_chromosomes)}): {sorted(list(feature_chromosomes))[:10]}...", file=sys.stderr)
            print(f"DEBUG: Matching chromosomes ({len(chromosomes)}): {chromosomes[:10]}...", file=sys.stderr)
            if not chromosomes:
                print(f"DEBUG: No matching chromosomes found!", file=sys.stderr)
                print(f"DEBUG: Read chromosomes sample: {sorted(list(read_chromosomes))[:20]}", file=sys.stderr)
                print(f"DEBUG: Feature chromosomes sample: {sorted(list(feature_chromosomes))[:20]}", file=sys.stderr)
        
        if not chromosomes:
            return pl.LazyFrame(
                schema={
                    "chromosome": pl.Utf8,
                    "read_start": pl.Int64,
                    "read_end": pl.Int64,
                    "fragment_length": pl.Int64,
                    "size_category": pl.Utf8,
                    "genome": pl.Utf8,
                    "feature_type": pl.Utf8,
                    "gene_id": pl.Utf8,
                    "gene_name": pl.Utf8,
                    "feature_start": pl.Int64,
                    "feature_end": pl.Int64,
                }
            )
        
        # Process chromosome by chromosome to limit memory
        # Build lazy frame operations for each chromosome and union them
        overlap_frames = []
        
        for chrom in chromosomes:
            reads_chr = (
                reads_df
                .filter(pl.col("chromosome") == chrom)
                .with_columns([
                    pl.col("start").alias("read_start"),
                    pl.col("end").alias("read_end"),
                ])
            )
            
            features_chr = (
                features_df
                .filter(pl.col("seqname") == chrom)
                .with_columns([
                    pl.col("start").alias("feature_start"),
                    pl.col("end").alias("feature_end"),
                ])
                .select([
                    "seqname",
                    "feature_start",
                    "feature_end",
                    "feature_type",
                    "gene_id",
                    "gene_name",
                    "genome",
                ])
            )
            
            # Use cross join with filtering - more efficient than Cartesian on full dataset
            # Polars will optimize this better when done per-chromosome
            chrom_overlaps = (
                reads_chr.join(
                    features_chr,
                    how="cross",
                    suffix="_right"
                )
                .filter(
                    (pl.col("read_start") < pl.col("feature_end"))
                    & (pl.col("read_end") > pl.col("feature_start"))
                )
                .select([
                    "chromosome",
                    "read_start",
                    "read_end",
                    pl.coalesce([pl.col("fragment_length"), (pl.col("read_end") - pl.col("read_start"))]).alias("fragment_length"),
                    pl.coalesce([pl.col("size_category"), pl.lit(None).cast(pl.Utf8)]).alias("size_category"),
                    pl.coalesce([pl.col("genome"), pl.col("genome_right")]).alias("genome"),
                    "feature_type",
                    "gene_id",
                    "gene_name",
                    "feature_start",
                    "feature_end",
                ])
                .drop("genome_right", strict=False)
            )
            
            overlap_frames.append(chrom_overlaps)
        
        if not overlap_frames:
            return pl.LazyFrame(
                schema={
                    "chromosome": pl.Utf8,
                    "read_start": pl.Int64,
                    "read_end": pl.Int64,
                    "fragment_length": pl.Int64,
                    "size_category": pl.Utf8,
                    "genome": pl.Utf8,
                    "feature_type": pl.Utf8,
                    "gene_id": pl.Utf8,
                    "gene_name": pl.Utf8,
                    "feature_start": pl.Int64,
                    "feature_end": pl.Int64,
                }
            )
        
        # Union all chromosome overlaps
        result = overlap_frames[0]
        for frame in overlap_frames[1:]:
            result = pl.concat([result, frame])
        
        # Add size_category if missing
        result_schema = result.collect_schema()
        if "size_category" not in result_schema.names() or result.select(pl.col("size_category").is_null().sum()).collect().item() > 0:
            from superfast_fragment_analyzer.bed_processor import BedProcessor
            result = result.with_columns(
                pl.when(pl.col("size_category").is_null())
                .then(
                    pl.col("fragment_length")
                    .map_elements(BedProcessor._classify_fragment_size, return_dtype=pl.Utf8)
                )
                .otherwise(pl.col("size_category"))
                .alias("size_category")
            )
        
        return result
    
    def compute_statistics(
        self,
        genome_filter: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, pl.DataFrame]:
        """
        Compute overlap statistics.
        
        Args:
            genome_filter: Filter by genome ('human', 'pig', or None for both)
            
        Returns:
            Dictionary with statistics DataFrames
        """
        # Filter by genome if specified
        reads_df = self.bed_df
        features_df = self.gtf_features_df
        
        if genome_filter:
            reads_df = reads_df.filter(pl.col("genome") == genome_filter)
            features_df = features_df.filter(pl.col("genome") == genome_filter)
        
        # Compute overlaps (returns LazyFrame)
        overlaps_lazy = self._compute_overlaps(reads_df, features_df, debug=debug)
        
        # Materialize overlaps only when needed for statistics
        # First check if empty efficiently
        overlaps_count = overlaps_lazy.select(pl.len()).collect().item()
        if overlaps_count == 0:
            # Return empty statistics - get total reads efficiently
            total_reads = reads_df.select(pl.len()).collect().item()
            return {
                "overlaps": pl.DataFrame(
                    schema={
                        "chromosome": pl.Utf8,
                        "read_start": pl.Int64,
                        "read_end": pl.Int64,
                        "fragment_length": pl.Int64,
                        "size_category": pl.Utf8,
                        "genome": pl.Utf8,
                        "feature_type": pl.Utf8,
                        "gene_id": pl.Utf8,
                        "gene_name": pl.Utf8,
                        "feature_start": pl.Int64,
                        "feature_end": pl.Int64,
                    }
                ),
                "summary": pl.DataFrame({
                    "feature_type": ["exon", "intron", "promoter"],
                    "overlapping_reads": [0, 0, 0],
                    "total_reads": [total_reads, total_reads, total_reads],
                    "percentage": [0.0, 0.0, 0.0],
                    "median_fragment_length": [None, None, None],
                }),
                "fragment_stats": pl.DataFrame(
                    schema={
                        "feature_type": pl.Utf8,
                        "size_category": pl.Utf8,
                        "count": pl.Int64,
                        "median_fragment_length": pl.Float64,
                    }
                ),
            }
        
        # Get total reads efficiently
        total_reads = reads_df.select(pl.len()).collect().item()
        
        # Compute statistics using lazy evaluation
        # Get unique reads with overlaps
        unique_reads_with_overlaps = (
            overlaps_lazy
            .select(["read_start", "read_end"])
            .unique()
            .select(pl.len())
            .collect()
            .item()
        )
        
        # Compute summary statistics using lazy evaluation
        summary_data = []
        for feature_type in ["exon", "intron", "promoter"]:
            feature_overlaps_lazy = overlaps_lazy.filter(pl.col("feature_type") == feature_type)
            
            # Count unique reads efficiently
            overlapping_reads = (
                feature_overlaps_lazy
                .select(["read_start", "read_end"])
                .unique()
                .select(pl.len())
                .collect()
                .item()
            )
            
            # Calculate median fragment length efficiently
            if overlapping_reads > 0:
                median_length = (
                    feature_overlaps_lazy
                    .select(pl.col("fragment_length").median())
                    .collect()
                    .item()
                )
            else:
                median_length = None
            
            percentage = (overlapping_reads / total_reads * 100) if total_reads > 0 else 0.0
            
            summary_data.append({
                "feature_type": feature_type,
                "overlapping_reads": overlapping_reads,
                "total_reads": total_reads,
                "percentage": percentage,
                "median_fragment_length": median_length,
            })
        
        summary_df = pl.DataFrame(summary_data)
        
        # Create fragment length statistics by feature type and size category
        fragment_stats_data = []
        for feature_type in ["exon", "intron", "promoter"]:
            feature_overlaps_lazy = overlaps_lazy.filter(pl.col("feature_type") == feature_type)
            
            for size_cat in ["sub-nucleosome", "mono-nucleosome", "di-nucleosome", "tri-nucleosome"]:
                size_overlaps_lazy = feature_overlaps_lazy.filter(pl.col("size_category") == size_cat)
                
                # Count unique reads efficiently
                count = (
                    size_overlaps_lazy
                    .select(["read_start", "read_end"])
                    .unique()
                    .select(pl.len())
                    .collect()
                    .item()
                )
                
                if count > 0:
                    median_length = (
                        size_overlaps_lazy
                        .select(pl.col("fragment_length").median())
                        .collect()
                        .item()
                    )
                    fragment_stats_data.append({
                        "feature_type": feature_type,
                        "size_category": size_cat,
                        "count": count,
                        "median_fragment_length": median_length,
                    })
        
        fragment_stats_df = pl.DataFrame(fragment_stats_data) if fragment_stats_data else pl.DataFrame(
            schema={
                "feature_type": pl.Utf8,
                "size_category": pl.Utf8,
                "count": pl.Int64,
                "median_fragment_length": pl.Float64,
            }
        )
        
        # Materialize overlaps DataFrame only at the end, and write directly to disk if possible
        # For now, collect it but this could be optimized further to write streaming
        overlaps_df = overlaps_lazy.collect()
        
        return {
            "overlaps": overlaps_df,
            "summary": summary_df,
            "fragment_stats": fragment_stats_df,
        }
    
    def compute_all_statistics(self, debug: bool = False) -> Dict[str, Dict[str, pl.DataFrame]]:
        """
        Compute statistics for human, pig, and combined genomes.
        
        Args:
            debug: If True, print diagnostic information
        
        Returns:
            Dictionary with 'human', 'pig', and 'combined' statistics
        """
        results = {}
        
        # Human statistics
        if debug:
            import sys
            print("DEBUG: Computing human statistics...", file=sys.stderr)
        results["human"] = self.compute_statistics(genome_filter="human", debug=debug)
        
        # Pig statistics
        if debug:
            import sys
            print("DEBUG: Computing pig statistics...", file=sys.stderr)
        results["pig"] = self.compute_statistics(genome_filter="pig", debug=debug)
        
        # Combined statistics
        if debug:
            import sys
            print("DEBUG: Computing combined statistics...", file=sys.stderr)
        results["combined"] = self.compute_statistics(genome_filter=None, debug=debug)
        
        return results
    
    def save_statistics(
        self,
        output_dir: Path,
        prefix: str = "overlap_stats",
    ) -> Dict[str, Path]:
        """
        Save statistics to files.
        
        Args:
            output_dir: Directory to save statistics files
            prefix: Prefix for output files
            
        Returns:
            Dictionary mapping statistic type to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_stats = self.compute_all_statistics()
        output_files = {}
        
        for genome_type, stats in all_stats.items():
            # Save summary
            summary_path = output_dir / f"{prefix}_{genome_type}_summary.tsv"
            stats["summary"].write_csv(summary_path, separator="\t")
            output_files[f"{genome_type}_summary"] = summary_path
            
            # Save fragment length statistics
            fragment_stats_path = output_dir / f"{prefix}_{genome_type}_fragment_stats.tsv"
            stats["fragment_stats"].write_csv(fragment_stats_path, separator="\t")
            output_files[f"{genome_type}_fragment_stats"] = fragment_stats_path
            
            # Save detailed overlaps
            overlaps_path = output_dir / f"{prefix}_{genome_type}_overlaps.parquet"
            stats["overlaps"].write_parquet(overlaps_path)
            output_files[f"{genome_type}_overlaps"] = overlaps_path
        
        return output_files
    
    @staticmethod
    def generate_feature_counts(
        overlaps_df: pl.DataFrame,
        feature_type: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Generate gene count table filtered by feature type.
        
        Counts unique reads per gene for a specific feature type (exon, intron, promoter)
        or across all feature types if None.
        
        Args:
            overlaps_df: Overlaps DataFrame with columns: gene_id, gene_name, feature_type, read_start, read_end
            feature_type: Feature type to filter by ('exon', 'intron', 'promoter', or None for all)
            
        Returns:
            DataFrame with columns: gene_id, gene_name, read_count
        """
        if overlaps_df.is_empty():
            return pl.DataFrame(
                schema={
                    "gene_id": pl.Utf8,
                    "gene_name": pl.Utf8,
                    "read_count": pl.Int64,
                }
            )
        
        # Filter by feature type if specified
        filtered_df = overlaps_df
        if feature_type is not None:
            if feature_type not in ["exon", "intron", "promoter"]:
                raise ValueError(f"feature_type must be 'exon', 'intron', 'promoter', or None, got: {feature_type}")
            filtered_df = overlaps_df.filter(pl.col("feature_type") == feature_type)
        
        # Generate counts
        counts = (
            filtered_df
            .select(["gene_id", "gene_name", "read_start", "read_end"])
            .unique(subset=["gene_id", "read_start", "read_end"])  # Count each read only once per gene
            .group_by(["gene_id", "gene_name"])
            .agg(pl.count().alias("read_count"))
            .sort("read_count", descending=True)
        )
        
        return counts
    
    @staticmethod
    def generate_gene_counts(overlaps_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate gene count table from overlaps DataFrame.
        
        Counts unique reads per gene across all feature types.
        
        Args:
            overlaps_df: Overlaps DataFrame with columns: gene_id, gene_name, read_start, read_end
            
        Returns:
            DataFrame with columns: gene_id, gene_name, read_count
        """
        return OverlapStats.generate_feature_counts(overlaps_df, feature_type=None)
    
    @staticmethod
    def generate_exon_counts(overlaps_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate exon count table with per-gene aggregate counts over exons.
        
        Counts unique reads per gene that overlap exons.
        
        Args:
            overlaps_df: Overlaps DataFrame with columns: gene_id, gene_name, feature_type, read_start, read_end
            
        Returns:
            DataFrame with columns: gene_id, gene_name, read_count
        """
        return OverlapStats.generate_feature_counts(overlaps_df, feature_type="exon")
    
    def save_statistics_with_counts(
        self,
        output_dir: Path,
        prefix: str = "overlap_stats",
        debug: bool = False,
    ) -> Dict[str, Path]:
        """
        Save statistics to files including gene and exon count tables.
        
        Args:
            output_dir: Directory to save statistics files
            prefix: Prefix for output files
            
        Returns:
            Dictionary mapping statistic type to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_stats = self.compute_all_statistics()
        output_files = {}
        
        for genome_type, stats in all_stats.items():
            # Save summary
            summary_path = output_dir / f"{prefix}_{genome_type}_summary.tsv"
            stats["summary"].write_csv(summary_path, separator="\t")
            output_files[f"{genome_type}_summary"] = summary_path
            
            # Save fragment length statistics
            fragment_stats_path = output_dir / f"{prefix}_{genome_type}_fragment_stats.tsv"
            stats["fragment_stats"].write_csv(fragment_stats_path, separator="\t")
            output_files[f"{genome_type}_fragment_stats"] = fragment_stats_path
            
            # Save detailed overlaps
            overlaps_path = output_dir / f"{prefix}_{genome_type}_overlaps.parquet"
            stats["overlaps"].write_parquet(overlaps_path)
            output_files[f"{genome_type}_overlaps"] = overlaps_path
            
            # Generate and save gene counts
            gene_counts = self.generate_gene_counts(stats["overlaps"])
            gene_counts_path = output_dir / f"{prefix}_{genome_type}_gene_counts.tsv"
            gene_counts.write_csv(gene_counts_path, separator="\t")
            output_files[f"{genome_type}_gene_counts"] = gene_counts_path
            
            # Generate and save exon counts
            exon_counts = self.generate_exon_counts(stats["overlaps"])
            exon_counts_path = output_dir / f"{prefix}_{genome_type}_exon_counts.tsv"
            exon_counts.write_csv(exon_counts_path, separator="\t")
            output_files[f"{genome_type}_exon_counts"] = exon_counts_path
        
        return output_files


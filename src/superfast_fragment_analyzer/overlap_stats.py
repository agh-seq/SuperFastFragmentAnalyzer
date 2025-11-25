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
    def bed_df(self) -> pl.DataFrame:
        """Lazy load BED DataFrame from Parquet if needed."""
        if self._bed_df is None:
            if isinstance(self.bed_source, Path):
                self._bed_df = pl.read_parquet(self.bed_source)
            else:
                self._bed_df = self.bed_source
        return self._bed_df
    
    @property
    def gtf_features_df(self) -> pl.DataFrame:
        """Lazy load GTF features DataFrame from Parquet if needed."""
        if self._gtf_features_df is None:
            if isinstance(self.gtf_features_source, Path):
                self._gtf_features_df = pl.read_parquet(self.gtf_features_source)
            else:
                self._gtf_features_df = self.gtf_features_source
        return self._gtf_features_df
    
    def _compute_overlaps(
        self,
        reads_df: pl.DataFrame,
        features_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute overlaps between reads and features using interval joins.
        
        Args:
            reads_df: Reads DataFrame
            features_df: Features DataFrame
            
        Returns:
            DataFrame with overlapping reads and their feature types
        """
        read_chromosomes = set(reads_df["chromosome"].unique().to_list())
        feature_chromosomes = set(features_df["seqname"].unique().to_list())
        chromosomes = sorted(read_chromosomes & feature_chromosomes)
        
        if not chromosomes:
            return pl.DataFrame(
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
        
        overlaps = []
        READ_CHUNK_SIZE = 10000  # Process reads in small chunks to limit memory
        FEATURE_CHUNK_SIZE = 50000  # Process features in chunks too
        
        for chrom in chromosomes:
            reads_chr = reads_df.filter(pl.col("chromosome") == chrom)
            features_chr = features_df.filter(pl.col("seqname") == chrom)
            
            if reads_chr.is_empty() or features_chr.is_empty():
                continue
            
            # Process both reads and features in chunks to avoid memory explosion
            num_reads = len(reads_chr)
            num_features = len(features_chr)
            chrom_overlaps = []
            
            # Double chunking: process reads in chunks, and for each read chunk,
            # process features in chunks too
            for read_start in range(0, num_reads, READ_CHUNK_SIZE):
                reads_chunk = reads_chr.slice(read_start, READ_CHUNK_SIZE)
                
                # Process features in chunks for this read chunk
                for feat_start in range(0, num_features, FEATURE_CHUNK_SIZE):
                    features_chunk = features_chr.slice(feat_start, FEATURE_CHUNK_SIZE)
                    
                    # Prepare frames for join
                    reads_join = reads_chunk.rename({"chromosome": "seqname"}).with_columns(pl.lit(1).alias("_key"))
                    features_join = features_chunk.with_columns(pl.lit(1).alias("_key"))
                    
                    overlaps_chunk = (
                        reads_join.join(features_join, on="_key", how="inner")
                        .filter(
                            (pl.col("start") < pl.col("end_right"))
                            & (pl.col("end") > pl.col("start_right"))
                        )
                        .rename(
                            {
                                "seqname": "chromosome",
                                "start": "read_start",
                                "end": "read_end",
                                "start_right": "feature_start",
                                "end_right": "feature_end",
                            }
                        )
                        .drop("_key")
                    )
                    
                    if "fragment_length_right" in overlaps_chunk.columns:
                        overlaps_chunk = overlaps_chunk.rename({"fragment_length_right": "fragment_length"})
                    if "size_category_right" in overlaps_chunk.columns:
                        overlaps_chunk = overlaps_chunk.rename({"size_category_right": "size_category"})
                    if "genome_right" in overlaps_chunk.columns:
                        overlaps_chunk = overlaps_chunk.rename({"genome_right": "genome"})
                    
                    if "fragment_length" not in overlaps_chunk.columns:
                        overlaps_chunk = overlaps_chunk.with_columns(
                            (pl.col("read_end") - pl.col("read_start")).alias("fragment_length")
                        )
                    if "size_category" not in overlaps_chunk.columns:
                        from superfast_fragment_analyzer.bed_processor import BedProcessor
                        overlaps_chunk = overlaps_chunk.with_columns(
                            pl.col("fragment_length")
                            .map_elements(BedProcessor._classify_fragment_size, return_dtype=pl.Utf8)
                            .alias("size_category")
                        )
                    
                    if not overlaps_chunk.is_empty():
                        chrom_overlaps.append(overlaps_chunk)
            
            # Concatenate all chunks for this chromosome
            if chrom_overlaps:
                overlaps.append(pl.concat(chrom_overlaps))
        
        if not overlaps:
            return pl.DataFrame(
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
        
        return pl.concat(overlaps)
    
    def compute_statistics(
        self,
        genome_filter: Optional[str] = None,
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
        
        # Compute overlaps
        overlaps_df = self._compute_overlaps(reads_df, features_df)
        
        if overlaps_df.is_empty():
            # Return empty statistics
            return {
                "overlaps": overlaps_df,
                "summary": pl.DataFrame({
                    "feature_type": ["exon", "intron", "promoter"],
                    "overlapping_reads": [0, 0, 0],
                    "total_reads": [len(reads_df), len(reads_df), len(reads_df)],
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
        
        # Count overlaps by feature type
        total_reads = len(reads_df)
        unique_reads_with_overlaps = len(
            overlaps_df.select(["read_start", "read_end"]).unique()
        )
        
        summary_data = []
        for feature_type in ["exon", "intron", "promoter"]:
            feature_overlaps = overlaps_df.filter(pl.col("feature_type") == feature_type)
            if not feature_overlaps.is_empty():
                overlapping_reads = len(
                    feature_overlaps.select(["read_start", "read_end"]).unique()
                )
                # Calculate median fragment length for this feature type
                median_length = feature_overlaps["fragment_length"].median()
            else:
                overlapping_reads = 0
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
            feature_overlaps = overlaps_df.filter(pl.col("feature_type") == feature_type)
            if not feature_overlaps.is_empty():
                for size_cat in ["sub-nucleosome", "mono-nucleosome", "di-nucleosome", "tri-nucleosome"]:
                    size_overlaps = feature_overlaps.filter(pl.col("size_category") == size_cat)
                    if not size_overlaps.is_empty():
                        count = len(size_overlaps.select(["read_start", "read_end"]).unique())
                        median_length = size_overlaps["fragment_length"].median()
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
        
        return {
            "overlaps": overlaps_df,
            "summary": summary_df,
            "fragment_stats": fragment_stats_df,
        }
    
    def compute_all_statistics(self) -> Dict[str, Dict[str, pl.DataFrame]]:
        """
        Compute statistics for human, pig, and combined genomes.
        
        Returns:
            Dictionary with 'human', 'pig', and 'combined' statistics
        """
        results = {}
        
        # Human statistics
        results["human"] = self.compute_statistics(genome_filter="human")
        
        # Pig statistics
        results["pig"] = self.compute_statistics(genome_filter="pig")
        
        # Combined statistics
        results["combined"] = self.compute_statistics(genome_filter=None)
        
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


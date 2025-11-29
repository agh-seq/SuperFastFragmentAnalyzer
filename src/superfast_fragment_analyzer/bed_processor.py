"""BED file processing and Parquet conversion using Polars."""

from pathlib import Path
from typing import Optional, Tuple

import polars as pl


class BedProcessor:
    """Process 3-column BED files and convert to Parquet format."""
    
    def __init__(self, bed_file: Path):
        """
        Initialize BED processor.
        
        Args:
            bed_file: Path to 3-column BED file (chromosome, start, end)
        """
        self.bed_file = Path(bed_file)
        if not self.bed_file.exists():
            raise FileNotFoundError(f"BED file not found: {bed_file}")
    
    def _determine_genome(self, chromosome: str) -> str:
        """
        Determine genome type based on chromosome name.
        
        Args:
            chromosome: Chromosome name
            
        Returns:
            'human' if starts with 'chr', 'pig' if starts with 'NC_', else 'unknown'
        """
        if chromosome.startswith("chr"):
            return "human"
        elif chromosome.startswith("NC_"):
            return "pig"
        else:
            return "unknown"
    
    @staticmethod
    def _classify_fragment_size(length: int) -> str:
        """
        Classify fragment into size categories.
        
        Args:
            length: Fragment length in base pairs
            
        Returns:
            Size category: 'sub-nucleosome', 'mono-nucleosome', 'di-nucleosome', or 'tri-nucleosome'
        """
        if length <= 100:
            return "sub-nucleosome"
        elif length <= 250:
            return "mono-nucleosome"
        elif length <= 420:
            return "di-nucleosome"
        else:
            return "tri-nucleosome"
    
    def read_bed(self) -> pl.DataFrame:
        """
        Read 3-column BED file into Polars DataFrame.
        
        Returns:
            DataFrame with columns: chromosome, start, end, fragment_length, size_category, genome
        """
        # Read BED file using lazy evaluation for better memory efficiency
        df_lazy = pl.scan_csv(
            self.bed_file,
            separator="\t",
            has_header=False,
            new_columns=["chromosome", "start", "end"],
            schema={"chromosome": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
        )
        
        # Filter out invalid records where end <= start
        # This handles cases where end position is 0 or less than start
        df_lazy = df_lazy.filter(pl.col("end") > pl.col("start"))
        
        # Calculate fragment length (end - start)
        df_lazy = df_lazy.with_columns(
            (pl.col("end") - pl.col("start")).alias("fragment_length")
        )
        
        # Classify fragment size using when/then instead of map_elements for better performance
        df_lazy = df_lazy.with_columns(
            pl.when(pl.col("fragment_length") <= 100)
            .then(pl.lit("sub-nucleosome"))
            .when(pl.col("fragment_length") <= 250)
            .then(pl.lit("mono-nucleosome"))
            .when(pl.col("fragment_length") <= 420)
            .then(pl.lit("di-nucleosome"))
            .otherwise(pl.lit("tri-nucleosome"))
            .alias("size_category")
        )
        
        # Add genome classification using when/then
        df_lazy = df_lazy.with_columns(
            pl.when(pl.col("chromosome").str.starts_with("chr"))
            .then(pl.lit("human"))
            .when(pl.col("chromosome").str.starts_with("NC_"))
            .then(pl.lit("pig"))
            .otherwise(pl.lit("unknown"))
            .alias("genome")
        )
        
        # Collect the lazy frame
        return df_lazy.collect()
    
    def to_parquet(
        self,
        output_path: Optional[Path] = None,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> Path:
        """
        Convert BED file to Parquet format.
        
        Args:
            output_path: Output Parquet file path. If None, uses bed_file with .parquet extension
            compression: Compression algorithm (default: zstd)
            compression_level: Compression level (default: 3)
            
        Returns:
            Path to created Parquet file
        """
        if output_path is None:
            output_path = self.bed_file.with_suffix(".parquet")
        else:
            output_path = Path(output_path)
        
        df = self.read_bed()
        
        # Write to Parquet with compression
        df.write_parquet(
            output_path,
            compression=compression,
            compression_level=compression_level,
        )
        
        return output_path
    
    def split_by_genome(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split BED reads by genome type.
        
        Returns:
            Tuple of (human_df, pig_df) DataFrames
        """
        df = self.read_bed()
        human_df = df.filter(pl.col("genome") == "human")
        pig_df = df.filter(pl.col("genome") == "pig")
        
        return human_df, pig_df


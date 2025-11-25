"""GTF annotation file processing and feature extraction."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl


class GtfProcessor:
    """Process GTF files and extract exons, introns, and promoters."""
    
    PROMOTER_UPSTREAM = 2000  # 2kb upstream
    PROMOTER_DOWNSTREAM = 2000  # 2kb downstream
    
    def __init__(self, gtf_file: Path):
        """
        Initialize GTF processor.
        
        Args:
            gtf_file: Path to GTF annotation file
        """
        self.gtf_file = Path(gtf_file)
        if not self.gtf_file.exists():
            raise FileNotFoundError(f"GTF file not found: {gtf_file}")
    
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
    
    def _parse_attributes(self, attributes_str: str) -> Dict[str, str]:
        """
        Parse GTF attributes column into dictionary.
        
        Args:
            attributes_str: Attributes string from GTF file
            
        Returns:
            Dictionary of attribute key-value pairs
        """
        attrs = {}
        # Remove trailing semicolon and split
        for item in attributes_str.rstrip(";").split(";"):
            item = item.strip()
            if not item:
                continue
            # Split on first space (key "value")
            if " " in item:
                key, value = item.split(" ", 1)
                # Remove quotes from value
                value = value.strip('"')
                attrs[key] = value
        return attrs
    
    def read_gtf(self) -> pl.DataFrame:
        """
        Read GTF file into Polars DataFrame.
        
        Returns:
            DataFrame with parsed GTF columns and attributes
        """
        # Read GTF file (standard 9 columns)
        df = pl.read_csv(
            self.gtf_file,
            separator="\t",
            has_header=False,
            comment_prefix="#",
            new_columns=[
                "seqname",
                "source",
                "feature",
                "start",
                "end",
                "score",
                "strand",
                "frame",
                "attributes",
            ],
            schema={
                "seqname": pl.Utf8,
                "source": pl.Utf8,
                "feature": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "score": pl.Utf8,  # Can be "." or number
                "strand": pl.Utf8,
                "frame": pl.Utf8,
                "attributes": pl.Utf8,
            },
        )
        
        # Parse attributes
        parsed_attrs = [
            self._parse_attributes(attr)
            for attr in df["attributes"].to_list()
        ]
        
        # Create DataFrame with all possible attribute keys
        all_keys = set()
        for attrs in parsed_attrs:
            all_keys.update(attrs.keys())
        
        # Build columns for each attribute
        attr_columns = {}
        for key in all_keys:
            attr_columns[key] = [
                attrs.get(key, None) for attrs in parsed_attrs
            ]
        
        # Add parsed columns
        if "gene_id" in attr_columns:
            df = df.with_columns(pl.Series("gene_id", attr_columns["gene_id"]))
        else:
            raise ValueError("GTF file must contain 'gene_id' attribute")
        
        if "transcript_id" in attr_columns:
            df = df.with_columns(pl.Series("transcript_id", attr_columns["transcript_id"]))
        
        if "gene_name" in attr_columns:
            df = df.with_columns(pl.Series("gene_name", attr_columns["gene_name"]))
        
        # Add genome classification
        df = df.with_columns(
            pl.col("seqname")
            .map_elements(self._determine_genome, return_dtype=pl.Utf8)
            .alias("genome")
        )
        
        # Convert start/end to proper types (GTF is 1-based, convert to 0-based for intervals)
        df = df.with_columns([
            (pl.col("start") - 1).alias("start_0based"),
            pl.col("end").alias("end_0based"),
        ])
        
        return df
    
    def _get_transcript_start(self, transcript_df: pl.DataFrame) -> int:
        """
        Get transcript start position (TSS) based on strand.
        
        Args:
            transcript_df: DataFrame for a single transcript
            
        Returns:
            TSS position (0-based)
        """
        strand = transcript_df["strand"][0]
        if strand == "+":
            return transcript_df["start_0based"].min()
        else:  # strand == "-"
            return transcript_df["end_0based"].max()
    
    def extract_features(self) -> pl.DataFrame:
        """
        Extract exons, introns, and promoters from GTF.
        
        Returns:
            DataFrame with columns: seqname, start, end, feature_type, gene_id, gene_name, genome, strand
        """
        gtf_df = self.read_gtf()
        
        # Filter to relevant features
        feature_df = gtf_df.filter(
            pl.col("feature").is_in(["gene", "transcript", "exon"])
        )
        
        all_features = []
        
        # Process by gene
        for gene_id_val in feature_df["gene_id"].unique().to_list():
            gene_data = feature_df.filter(pl.col("gene_id") == gene_id_val)
            
            if gene_data.is_empty():
                continue
            
            # Get gene info
            gene_row = gene_data.filter(pl.col("feature") == "gene").head(1)
            if gene_row.is_empty():
                continue
            
            seqname = gene_row["seqname"][0]
            strand = gene_row["strand"][0]
            gene_name = gene_row.get_column("gene_name")[0] if "gene_name" in gene_row.columns else None
            genome = gene_row["genome"][0]
            
            # Get all transcripts for this gene
            transcripts = gene_data.filter(pl.col("feature") == "transcript")["transcript_id"].unique().to_list()
            
            # Collect exons
            exons = []
            for transcript_id in transcripts:
                transcript_exons = gene_data.filter(
                    (pl.col("feature") == "exon") & (pl.col("transcript_id") == transcript_id)
                )
                for row in transcript_exons.iter_rows(named=True):
                    exons.append({
                        "seqname": seqname,
                        "start": row["start_0based"],
                        "end": row["end_0based"],
                        "feature_type": "exon",
                        "gene_id": gene_id_val,
                        "gene_name": gene_name,
                        "genome": genome,
                        "strand": strand,
                    })
            
            # Calculate introns (gaps between exons within transcripts)
            introns = []
            for transcript_id in transcripts:
                transcript_exons = gene_data.filter(
                    (pl.col("feature") == "exon") & (pl.col("transcript_id") == transcript_id)
                ).sort("start_0based")
                
                if transcript_exons.height > 1:
                    for i in range(transcript_exons.height - 1):
                        exon_end = transcript_exons["end_0based"][i]
                        next_exon_start = transcript_exons["start_0based"][i + 1]
                        
                        if next_exon_start > exon_end:
                            introns.append({
                                "seqname": seqname,
                                "start": exon_end,
                                "end": next_exon_start,
                                "feature_type": "intron",
                                "gene_id": gene_id_val,
                                "gene_name": gene_name,
                                "genome": genome,
                                "strand": strand,
                            })
            
            # Calculate promoters (±2kb from TSS)
            promoters = []
            for transcript_id in transcripts:
                transcript_data = gene_data.filter(pl.col("transcript_id") == transcript_id)
                tss = self._get_transcript_start(transcript_data)
                
                # Promoter is ±2kb from TSS regardless of strand
                # For both strands, we want the region around TSS
                promoter_start = max(0, tss - self.PROMOTER_UPSTREAM)
                promoter_end = tss + self.PROMOTER_DOWNSTREAM
                
                promoters.append({
                    "seqname": seqname,
                    "start": promoter_start,
                    "end": promoter_end,
                    "feature_type": "promoter",
                    "gene_id": gene_id_val,
                    "gene_name": gene_name,
                    "genome": genome,
                    "strand": strand,
                })
            
            all_features.extend(exons)
            all_features.extend(introns)
            all_features.extend(promoters)
        
        # Convert to DataFrame
        if not all_features:
            return pl.DataFrame(
                schema={
                    "seqname": pl.Utf8,
                    "start": pl.Int64,
                    "end": pl.Int64,
                    "feature_type": pl.Utf8,
                    "gene_id": pl.Utf8,
                    "gene_name": pl.Utf8,
                    "genome": pl.Utf8,
                    "strand": pl.Utf8,
                }
            )
        
        return pl.DataFrame(all_features)
    
    def split_by_genome(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split features by genome type.
        
        Returns:
            Tuple of (human_df, pig_df) DataFrames
        """
        features_df = self.extract_features()
        human_df = features_df.filter(pl.col("genome") == "human")
        pig_df = features_df.filter(pl.col("genome") == "pig")
        
        return human_df, pig_df


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
        
        # Parse attributes using Polars expressions instead of converting to Python list
        # This is much more memory efficient for large files
        # Extract common attributes using string operations
        df = df.with_columns([
            # Extract gene_id
            pl.col("attributes")
            .str.extract(r'gene_id\s+"([^"]+)"', 1)
            .alias("gene_id"),
            # Extract transcript_id
            pl.col("attributes")
            .str.extract(r'transcript_id\s+"([^"]+)"', 1)
            .alias("transcript_id"),
            # Extract gene_name
            pl.col("attributes")
            .str.extract(r'gene_name\s+"([^"]+)"', 1)
            .alias("gene_name"),
            # Extract gene_biotype (Ensembl/RefSeq format)
            pl.col("attributes")
            .str.extract(r'gene_biotype\s+"([^"]+)"', 1)
            .alias("gene_biotype"),
            # Extract transcript_biotype (Ensembl/RefSeq format)
            pl.col("attributes")
            .str.extract(r'transcript_biotype\s+"([^"]+)"', 1)
            .alias("transcript_biotype"),
            # Extract gene_type (GENCODE format)
            pl.col("attributes")
            .str.extract(r'gene_type\s+"([^"]+)"', 1)
            .alias("gene_type"),
            # Extract transcript_type (GENCODE format)
            pl.col("attributes")
            .str.extract(r'transcript_type\s+"([^"]+)"', 1)
            .alias("transcript_type"),
        ])
        
        # Use gene_biotype/gene_type if available, otherwise use transcript_biotype/transcript_type
        # Priority: gene_biotype > gene_type > transcript_biotype > transcript_type
        df = df.with_columns(
            pl.coalesce([
                pl.col("gene_biotype"),
                pl.col("gene_type"),
                pl.col("transcript_biotype"),
                pl.col("transcript_type")
            ])
            .alias("biotype")
        )
        
        # Check if gene_id exists
        if df["gene_id"].null_count() == len(df):
            raise ValueError("GTF file must contain 'gene_id' attribute")
        
        # Columns already added above using string extraction
        
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
    
    def extract_features(self, biotype_filter: Optional[str] = None) -> pl.DataFrame:
        """
        Extract exons, introns, and promoters from GTF.
        
        Args:
            biotype_filter: Optional biotype to filter by (e.g., "protein_coding").
                           If None, includes all biotypes.
        
        Returns:
            DataFrame with columns: seqname, start, end, feature_type, gene_id, gene_name, genome, strand
        """
        gtf_df = self.read_gtf()
        
        # Filter to relevant features
        feature_df = gtf_df.filter(
            pl.col("feature").is_in(["gene", "transcript", "exon"])
        )
        
        # Filter by biotype if specified
        if biotype_filter:
            # Get gene IDs that match the biotype filter
            # Check both gene-level and transcript-level features for biotype
            matching_genes = (
                feature_df
                .filter(pl.col("biotype") == biotype_filter)
                .select("gene_id")
                .drop_nulls()
                .unique()
            )
            
            # Filter feature_df to only include matching genes
            if len(matching_genes) > 0:
                matching_gene_list = matching_genes["gene_id"].to_list()
                feature_df = feature_df.filter(pl.col("gene_id").is_in(matching_gene_list))
            else:
                # No matching genes found, return empty result
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
        
        all_features = []
        
        # Process by gene - use lazy evaluation to avoid loading all at once
        # Get unique gene IDs efficiently
        unique_gene_ids = feature_df["gene_id"].drop_nulls().unique().to_list()
        
        # Process in chunks to limit memory
        GENE_CHUNK_SIZE = 1000
        for chunk_start in range(0, len(unique_gene_ids), GENE_CHUNK_SIZE):
            gene_chunk = unique_gene_ids[chunk_start:chunk_start + GENE_CHUNK_SIZE]
            gene_data_chunk = feature_df.filter(pl.col("gene_id").is_in(gene_chunk))
            
            for gene_id_val in gene_chunk:
                gene_data = gene_data_chunk.filter(pl.col("gene_id") == gene_id_val)
            
                if gene_data.is_empty():
                    continue
                
                # Get gene info
                gene_row = gene_data.filter(pl.col("feature") == "gene").head(1)
                if gene_row.is_empty():
                    continue
                
                seqname = gene_row["seqname"][0]
                strand = gene_row["strand"][0]
                gene_name = gene_row.get_column("gene_name")[0] if "gene_name" in gene_row.columns and gene_row["gene_name"][0] is not None else None
                genome = gene_row["genome"][0]
                
                # Get all transcripts for this gene
                transcripts = gene_data.filter(pl.col("feature") == "transcript")["transcript_id"].drop_nulls().unique().to_list()
            
                # Collect exons
                exons = []
                for transcript_id in transcripts:
                    transcript_exons = gene_data.filter(
                        (pl.col("feature") == "exon") & (pl.col("transcript_id") == transcript_id)
                    )
                    # Use to_dicts() instead of iter_rows() for better performance
                    for row in transcript_exons.to_dicts():
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


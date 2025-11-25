"""SuperFast Fragment Analyzer - Fast BED file processing and GTF overlap statistics."""

__version__ = "0.1.0"

from superfast_fragment_analyzer.bed_processor import BedProcessor
from superfast_fragment_analyzer.gtf_processor import GtfProcessor
from superfast_fragment_analyzer.overlap_stats import OverlapStats

__all__ = ["BedProcessor", "GtfProcessor", "OverlapStats"]


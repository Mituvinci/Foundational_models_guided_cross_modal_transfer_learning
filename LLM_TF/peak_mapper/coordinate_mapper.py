"""
Peak Coordinate Mapper

Maps peak coordinates between different scATAC-seq datasets using genomic overlap.
Handles different peak naming conventions and genome builds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re


class PeakCoordinateMapper:
    """
    Map peak coordinates between datasets using genomic overlap.

    Supports multiple methods:
    - 'exact': Exact coordinate match
    - 'overlap': Any overlap (default)
    - 'overlap_50pct': Require 50% reciprocal overlap
    - 'nearest': Map to nearest peak within max_distance

    Example:
        mapper = PeakCoordinateMapper(method='overlap_50pct')
        mapping = mapper.map_peaks(source_peaks, target_peaks)
        aligned_matrix = mapper.align_matrix(X_target, target_peaks, source_peaks)
    """

    def __init__(self, method='overlap', max_distance=500):
        """
        Initialize peak mapper.

        Args:
            method: Mapping method ('exact', 'overlap', 'overlap_50pct', 'nearest')
            max_distance: Maximum distance for 'nearest' method (bp)
        """
        self.method = method
        self.max_distance = max_distance

    def parse_peak_name(self, peak_name: str) -> Optional[Tuple[str, int, int]]:
        """
        Parse peak name to extract chromosome, start, end.

        Supports formats:
        - chr1:1000-2000
        - chr1_1000_2000
        - 1:1000-2000
        - 1_1000_2000

        Returns:
            (chromosome, start, end) or None if parsing fails
        """
        # Try different formats
        patterns = [
            r'(chr)?(\d+|X|Y|M):(\d+)-(\d+)',   # chr1:1000-2000
            r'(chr)?(\d+|X|Y|M)_(\d+)_(\d+)',   # chr1_1000_2000
            r'(chr)(\d+|X|Y|M)-(\d+)-(\d+)',    # chr1-1000-2000 (YOUR FORMAT!)
        ]

        for pattern in patterns:
            match = re.match(pattern, peak_name)
            if match:
                groups = match.groups()
                if len(groups) == 4:
                    chr_prefix, chrom, start, end = groups
                    chrom = f"chr{chrom}" if not chr_prefix else f"{chr_prefix}{chrom}"
                    return (chrom, int(start), int(end))

        return None

    def peaks_overlap(self, peak1: Tuple[str, int, int], peak2: Tuple[str, int, int],
                      min_overlap_fraction: float = 0.0) -> bool:
        """
        Check if two peaks overlap.

        Args:
            peak1: (chr, start, end)
            peak2: (chr, start, end)
            min_overlap_fraction: Minimum reciprocal overlap (0.0 to 1.0)

        Returns:
            True if peaks overlap sufficiently
        """
        chr1, start1, end1 = peak1
        chr2, start2, end2 = peak2

        # Different chromosomes = no overlap
        if chr1 != chr2:
            return False

        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_length = max(0, overlap_end - overlap_start)

        if overlap_length == 0:
            return False

        # Check reciprocal overlap if required
        if min_overlap_fraction > 0:
            len1 = end1 - start1
            len2 = end2 - start2
            overlap_frac1 = overlap_length / len1
            overlap_frac2 = overlap_length / len2
            min_overlap = min(overlap_frac1, overlap_frac2)
            return min_overlap >= min_overlap_fraction

        return True

    def peak_distance(self, peak1: Tuple[str, int, int], peak2: Tuple[str, int, int]) -> float:
        """
        Calculate distance between two peaks (midpoint to midpoint).

        Returns:
            Distance in bp, or inf if different chromosomes
        """
        chr1, start1, end1 = peak1
        chr2, start2, end2 = peak2

        if chr1 != chr2:
            return float('inf')

        mid1 = (start1 + end1) / 2
        mid2 = (start2 + end2) / 2

        return abs(mid1 - mid2)

    def map_peaks(self, source_peaks: List[str], target_peaks: List[str]) -> Dict[str, str]:
        """
        Map target peaks to source peaks.

        Args:
            source_peaks: Reference peak names (training dataset)
            target_peaks: Peaks to map (new dataset)

        Returns:
            Dictionary: {target_peak: source_peak}
        """
        print(f"Mapping {len(target_peaks)} target peaks to {len(source_peaks)} source peaks...")
        print(f"  Method: {self.method}")

        # Parse all peaks
        source_coords = {}
        for peak in source_peaks:
            coords = self.parse_peak_name(peak)
            if coords:
                source_coords[peak] = coords

        target_coords = {}
        for peak in target_peaks:
            coords = self.parse_peak_name(peak)
            if coords:
                target_coords[peak] = coords

        print(f"  Parsed: {len(source_coords)}/{len(source_peaks)} source, {len(target_coords)}/{len(target_peaks)} target")

        # Map based on method
        mapping = {}

        if self.method == 'exact':
            # Exact coordinate match
            source_set = set(source_peaks)
            for tgt_peak in target_peaks:
                if tgt_peak in source_set:
                    mapping[tgt_peak] = tgt_peak

        elif self.method == 'overlap':
            # Any overlap
            for tgt_peak, tgt_coords in target_coords.items():
                for src_peak, src_coords in source_coords.items():
                    if self.peaks_overlap(tgt_coords, src_coords, min_overlap_fraction=0.0):
                        mapping[tgt_peak] = src_peak
                        break

        elif self.method == 'overlap_50pct':
            # 50% reciprocal overlap
            for tgt_peak, tgt_coords in target_coords.items():
                for src_peak, src_coords in source_coords.items():
                    if self.peaks_overlap(tgt_coords, src_coords, min_overlap_fraction=0.5):
                        mapping[tgt_peak] = src_peak
                        break

        elif self.method == 'nearest':
            # Nearest peak within max_distance
            for tgt_peak, tgt_coords in target_coords.items():
                min_dist = float('inf')
                best_src = None

                for src_peak, src_coords in source_coords.items():
                    dist = self.peak_distance(tgt_coords, src_coords)
                    if dist < min_dist and dist <= self.max_distance:
                        min_dist = dist
                        best_src = src_peak

                if best_src:
                    mapping[tgt_peak] = best_src
        else:
            raise ValueError(f"Unknown method: {self.method}")

        match_rate = len(mapping) / len(target_peaks) * 100
        print(f"  Mapped: {len(mapping)}/{len(target_peaks)} ({match_rate:.1f}%)")

        return mapping

    def align_matrix(self, X_target: np.ndarray, target_peaks: List[str],
                     source_peaks: List[str], mapping: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Align target peak matrix to source peak order.

        Args:
            X_target: Target data matrix (n_cells, n_target_peaks)
            target_peaks: Target peak names
            source_peaks: Source peak names (reference order)
            mapping: Pre-computed mapping (optional, will compute if not provided)

        Returns:
            X_aligned: Aligned matrix (n_cells, n_source_peaks)
                      Missing peaks filled with zeros
        """
        if mapping is None:
            mapping = self.map_peaks(source_peaks, target_peaks)

        n_cells = X_target.shape[0]
        n_source_peaks = len(source_peaks)

        # Create aligned matrix (zeros for missing peaks)
        X_aligned = np.zeros((n_cells, n_source_peaks), dtype=X_target.dtype)

        # Build reverse mapping: source_peak -> index
        source_to_idx = {peak: idx for idx, peak in enumerate(source_peaks)}
        target_to_idx = {peak: idx for idx, peak in enumerate(target_peaks)}

        # Fill in mapped peaks
        mapped_count = 0
        for tgt_peak, src_peak in mapping.items():
            if src_peak in source_to_idx and tgt_peak in target_to_idx:
                tgt_idx = target_to_idx[tgt_peak]
                src_idx = source_to_idx[src_peak]
                X_aligned[:, src_idx] = X_target[:, tgt_idx]
                mapped_count += 1

        print(f"  Aligned: {mapped_count}/{n_source_peaks} peaks copied")
        print(f"  Missing: {n_source_peaks - mapped_count} peaks (filled with 0)")

        return X_aligned


class UnionPeakMapper:
    """
    Create union of multiple peak sets with flexible window merging.

    Handles different peak sets by creating a union reference where:
    1. Collect all peaks from all datasets
    2. Sort by chromosome + position
    3. Merge peaks within window (default ±50bp)
    4. Map each dataset to union coordinates
    5. Zero-fill missing peaks

    Example:
        mapper = UnionPeakMapper(merge_window=50)
        union_peaks = mapper.create_union([reh_peaks, sup_peaks])
        reh_aligned = mapper.align_to_union(X_reh, reh_peaks, union_peaks)
        sup_aligned = mapper.align_to_union(X_sup, sup_peaks, union_peaks)
    """

    def __init__(self, merge_window=50):
        """
        Initialize union mapper.

        Args:
            merge_window: Merge peaks within ±N bp (default: 50)
        """
        self.merge_window = merge_window
        self.coord_mapper = PeakCoordinateMapper(method='overlap')

    def create_union(self, peak_sets: List[List[str]]) -> List[str]:
        """
        Create union of multiple peak sets with merging.

        Args:
            peak_sets: List of peak name lists from different datasets

        Returns:
            union_peaks: Merged union peak list
        """
        print(f"\n{'='*60}")
        print(f"CREATING UNION PEAK REFERENCE (±{self.merge_window}bp window)")
        print(f"{'='*60}")

        # Parse all peaks from all datasets
        all_peaks = []
        for i, peaks in enumerate(peak_sets):
            print(f"  Dataset {i+1}: {len(peaks)} peaks")
            print(f"    Example peaks: {peaks[:3]}")  # Debug: show examples
            parsed_count = 0
            for peak in peaks:
                coords = self.coord_mapper.parse_peak_name(peak)
                if coords:
                    all_peaks.append((peak, coords))
                    parsed_count += 1
            print(f"    Parsed successfully: {parsed_count}/{len(peaks)}")
            if parsed_count == 0:
                print(f"    ⚠️ WARNING: No peaks parsed! Check peak name format.")

        print(f"\n  Total peaks before merging: {len(all_peaks)}")

        if len(all_peaks) == 0:
            raise ValueError("No peaks could be parsed! Check peak name format. Expected: 'chr1:1000-2000' or 'chr1_1000_2000'")

        # Group by chromosome
        chr_groups = {}
        for peak_name, (chrom, start, end) in all_peaks:
            if chrom not in chr_groups:
                chr_groups[chrom] = []
            chr_groups[chrom].append((start, end, peak_name))

        # Merge overlapping peaks within each chromosome
        union_peaks = []

        for chrom in sorted(chr_groups.keys()):
            peaks = sorted(chr_groups[chrom])  # Sort by start position

            if not peaks:
                continue

            # Merge overlapping/nearby peaks
            merged = []
            current_start, current_end, current_name = peaks[0]

            for start, end, name in peaks[1:]:
                # Check if within merge window
                if start <= current_end + self.merge_window:
                    # Merge: extend current peak
                    current_end = max(current_end, end)
                    # Use first peak name as representative
                else:
                    # No overlap: save current and start new
                    merged.append(f"{chrom}:{current_start}-{current_end}")
                    current_start, current_end, current_name = start, end, name

            # Don't forget last peak
            merged.append(f"{chrom}:{current_start}-{current_end}")
            union_peaks.extend(merged)

        print(f"  Total peaks after merging: {len(union_peaks)}")
        print(f"  Reduction: {len(all_peaks) - len(union_peaks)} peaks merged")
        print(f"{'='*60}\n")

        return union_peaks

    def align_to_union(self, X: np.ndarray, original_peaks: List[str],
                       union_peaks: List[str]) -> np.ndarray:
        """
        Align dataset matrix to union peak coordinates.

        Args:
            X: Data matrix (n_cells, n_original_peaks)
            original_peaks: Original peak names for this dataset
            union_peaks: Union peak reference

        Returns:
            X_aligned: Aligned matrix (n_cells, n_union_peaks)
        """
        print(f"  Aligning {len(original_peaks)} peaks → {len(union_peaks)} union peaks...")

        # Use coordinate mapper with relaxed overlap
        mapper = PeakCoordinateMapper(method='overlap')
        mapping = mapper.map_peaks(union_peaks, original_peaks)

        # Reverse mapping: union_peak -> original_peak
        union_to_original = {}
        for orig_peak, union_peak in mapping.items():
            if union_peak not in union_to_original:
                union_to_original[union_peak] = []
            union_to_original[union_peak].append(orig_peak)

        # Build aligned matrix
        n_cells = X.shape[0]
        n_union_peaks = len(union_peaks)
        X_aligned = np.zeros((n_cells, n_union_peaks), dtype=X.dtype)

        original_to_idx = {peak: idx for idx, peak in enumerate(original_peaks)}

        mapped_count = 0
        for union_idx, union_peak in enumerate(union_peaks):
            if union_peak in union_to_original:
                # Use first matched peak (could average multiple matches)
                orig_peak = union_to_original[union_peak][0]
                if orig_peak in original_to_idx:
                    orig_idx = original_to_idx[orig_peak]
                    X_aligned[:, union_idx] = X[:, orig_idx]
                    mapped_count += 1

        if n_union_peaks > 0:
            match_rate = mapped_count / n_union_peaks * 100
            print(f"    Matched: {mapped_count}/{n_union_peaks} ({match_rate:.1f}%)")
            print(f"    Missing: {n_union_peaks - mapped_count} (filled with 0)")
        else:
            print(f"    ⚠️ WARNING: Union has 0 peaks!")

        return X_aligned

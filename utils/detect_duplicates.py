"""
Duplicate Detection Tool

Detects near-duplicate lines in text files using TF-IDF vectorization and clustering.
"""

import argparse
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path


class DuplicateDetector:
    def __init__(self, similarity_threshold=0.99, n_clusters=None, batch_size=256, verbose=False):
        self.similarity_threshold = similarity_threshold
        self.n_clusters = n_clusters  # None means use dynamic (sqrt-based) clustering
        self.batch_size = batch_size
        self.verbose = verbose

    def log(self, message, level="info"):
        """Print message if verbose mode is enabled"""
        if self.verbose or level == "error":
            print(message)

    def read_file(self, filepath):
        """Read lines from a text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            self.log(f"Read {len(lines)} lines from {filepath}")
            return lines
        except Exception as e:
            self.log(f"Error reading {filepath}: {e}", "error")
            return []

    def get_files_to_process(self, path):
        """Get list of files to process based on input path"""
        path = Path(path)

        if path.is_file():
            return [str(path)]
        elif path.is_dir():
            # Process common text file extensions
            extensions = ['*.txt', '*.csv', '*.tsv', '*.log']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(path, ext)))
                files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
            return sorted(set(files))
        else:
            self.log(f"Path {path} does not exist", "error")
            return []

    def get_deduplicated_directory_path(self, directory_path):
        """Generate deduplicated directory name"""
        path = Path(directory_path)
        return str(path.parent / f"{path.name}_deduplicated")

    def setup_deduplicated_directory(self, original_path, files):
        """Set up directory structure for deduplicated files"""
        original_path = Path(original_path)

        if original_path.is_file():
            # Single file - no directory setup needed
            return None

        # Directory processing
        dedupe_dir = Path(self.get_deduplicated_directory_path(original_path))

        # Create the base deduplicated directory
        dedupe_dir.mkdir(exist_ok=True)

        # Create subdirectory structure
        for file_path in files:
            file_path = Path(file_path)
            relative_path = file_path.relative_to(original_path)
            dedupe_file_dir = dedupe_dir / relative_path.parent
            dedupe_file_dir.mkdir(parents=True, exist_ok=True)

        return str(dedupe_dir)

    def get_deduplicated_filename(self, filepath, base_input_path=None, dedupe_base_dir=None):
        """Generate deduplicated filename"""
        path = Path(filepath)

        if dedupe_base_dir is None:
            # Single file case
            stem = path.stem
            suffix = path.suffix
            return str(path.parent / f"{stem}_deduplicated{suffix}")
        else:
            # Directory case - preserve relative structure
            base_input_path = Path(base_input_path)
            dedupe_base_dir = Path(dedupe_base_dir)
            relative_path = path.relative_to(base_input_path)

            stem = relative_path.stem
            suffix = relative_path.suffix
            dedupe_filename = f"{stem}_deduplicated{suffix}"

            return str(dedupe_base_dir / relative_path.parent / dedupe_filename)

    def cluster_with_minibatch_kmeans(self, tfidf_matrix):
        """Perform clustering using MiniBatch KMeans"""
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=self.batch_size
        )

        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        self.log("Completed mini-batch clustering")

        return cluster_labels, kmeans

    def create_tfidf_matrix(self, lines):
        """Create TF-IDF matrix from lines"""
        vectorizer = TfidfVectorizer(analyzer='char')
        tfidf_matrix = vectorizer.fit_transform(lines)
        self.log("Created TF-IDF matrix")
        return tfidf_matrix, vectorizer

    def find_duplicates_by_cluster(self, tfidf_matrix, cluster_labels):
        """
        Find near-duplicates within each cluster without storing full similarity matrix.
        Memory efficient but processes one cluster at a time.
        """
        duplicates = []
        unique_clusters = np.unique(cluster_labels)

        progress_bar = tqdm(unique_clusters, desc="Processing clusters", disable=not self.verbose)
        for cluster_id in progress_bar:
            # Get indices of items in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) <= 1:
                continue

            # Extract TF-IDF vectors for this cluster only
            cluster_tfidf = tfidf_matrix[cluster_indices]

            # Compute similarity matrix for this cluster only
            cluster_sim_matrix = cosine_similarity(cluster_tfidf)

            # Find pairs above threshold (excluding diagonal)
            rows, cols = np.where((cluster_sim_matrix >= self.similarity_threshold))

            # Convert back to original indices and store duplicates
            for i, j in zip(rows, cols):
                if i < j:  # Avoid duplicate pairs (i,j) and (j,i)
                    orig_i, orig_j = cluster_indices[i], cluster_indices[j]
                    duplicates.append((orig_i, orig_j, cluster_sim_matrix[i, j]))

        return duplicates

    def union_find(self, duplicate_pairs, total_items):
        """Group duplicate items using Union-Find data structure"""
        if len(duplicate_pairs) == 0:
            return [], {}

        # Convert to numpy array if not already
        duplicate_pairs = np.array(duplicate_pairs)

        # Initialize Union-Find
        uf = UnionFind(total_items)

        # Union all duplicate pairs
        progress_bar = tqdm(duplicate_pairs, desc="Unioning duplicates", disable=not self.verbose)
        for pair in progress_bar:
            uf.union(int(pair[0]), int(pair[1]))

        # Group items by their root parent
        groups = defaultdict(list)
        progress_bar = tqdm(range(total_items), desc="Grouping items", disable=not self.verbose)
        for i in progress_bar:
            root = uf.find(i)
            groups[root].append(i)

        # Remove groups with only one item (no duplicates)
        duplicate_groups = {root: items for root, items in groups.items() if len(items) > 1}

        # Keep first item from each group, mark rest for removal
        to_remove = []
        group_info = {}

        for root, items in duplicate_groups.items():
            items.sort()  # Consistent ordering
            to_remove.extend(items[1:])  # Remove all but first
            group_info[root] = {
                'size': len(items),
                'kept': items[0],
                'removed': items[1:]
            }

        return sorted(to_remove), group_info

    def process_file(self, filepath, create_deduplicated=False, base_input_path=None, dedupe_base_dir=None):
        """Process a single file and return results"""
        start_time = time.time()

        lines = self.read_file(filepath)
        if not lines:
            return None

        # Determine number of clusters to use
        if self.n_clusters is None:
            # Dynamic clustering based on dataset size (square root)
            n_clusters_to_use = max(1, int(np.sqrt(len(lines))))
            self.log(f"Using {n_clusters_to_use} clusters for {len(lines)} rows (sqrt-based)")
        else:
            # Use explicitly specified number of clusters
            n_clusters_to_use = self.n_clusters
            self.log(f"Using {n_clusters_to_use} clusters (user-specified)")

        # Temporarily store original n_clusters and use calculated value
        original_n_clusters = self.n_clusters
        self.n_clusters = n_clusters_to_use

        # Create TF-IDF matrix
        tfidf_matrix, vectorizer = self.create_tfidf_matrix(lines)

        # Perform clustering
        cluster_labels, _ = self.cluster_with_minibatch_kmeans(tfidf_matrix)

        # Restore original n_clusters
        self.n_clusters = original_n_clusters

        # Find duplicates
        duplicate_pairs = self.find_duplicates_by_cluster(tfidf_matrix, cluster_labels)
        self.log(f"Total near duplicate pairs found: {len(duplicate_pairs)}")

        # Group duplicates
        to_remove, group_info = self.union_find(duplicate_pairs, len(lines))

        # Calculate estimated N
        estimated_n = len(lines) / (len(lines) - len(to_remove)) if len(to_remove) < len(lines) else float('inf')

        compute_time = time.time() - start_time

        result = {
            'filename': os.path.basename(filepath),
            'num_rows': len(lines),
            'duplicate_pairs': len(duplicate_pairs),
            'estimated_n': estimated_n,
            'num_clusters': len(np.unique(cluster_labels)),
            'to_remove': len(to_remove),
            'compute_time': compute_time
        }

        # Create deduplicated file if requested
        if create_deduplicated:
            dedupe_filepath = self.create_deduplicated_file(filepath, lines, to_remove, base_input_path, dedupe_base_dir)
            if dedupe_filepath:
                result['deduplicated_file'] = dedupe_filepath

        return result

    def create_deduplicated_file(self, filepath, lines, to_remove_indices, base_input_path=None, dedupe_base_dir=None):
        """Create a deduplicated copy of the file"""
        if not to_remove_indices:
            self.log(f"No duplicates found in {filepath}, skipping deduplication")
            return None

        # Create set for faster lookup
        to_remove_set = set(to_remove_indices)

        # Keep lines that are not in the removal set
        deduplicated_lines = [line for i, line in enumerate(lines) if i not in to_remove_set]

        # Generate output filepath
        output_filepath = self.get_deduplicated_filename(filepath, base_input_path, dedupe_base_dir)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for line in deduplicated_lines:
                    f.write(line + '\n')

            self.log(f"Created deduplicated file: {output_filepath}")
            self.log(f"Removed {len(to_remove_indices)} duplicate lines from {len(lines)} total lines")
            return output_filepath

        except Exception as e:
            self.log(f"Error writing deduplicated file {output_filepath}: {e}", "error")
            return None


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Detect near-duplicate lines in text files using TF-IDF and clustering.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file.txt                    # Process single file
  %(prog)s /path/to/directory          # Process all text files in directory
  %(prog)s file.csv -v                 # Process with verbose output
  %(prog)s data/ -t 0.95 -c 5000       # Custom threshold and clusters
        """
    )

    parser.add_argument(
        'path',
        help='Path to file or directory to process'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.99,
        help='Similarity threshold for duplicates (default: 0.99)'
    )

    parser.add_argument(
        '-c', '--clusters',
        type=int,
        default=None,
        help='Number of clusters for KMeans (default: sqrt of number of rows)'
    )

    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=2048,
        help='Batch size for MiniBatch KMeans (default: 2048)'
    )

    parser.add_argument(
        '-d', '--deduplicate',
        action='store_true',
        help='Create deduplicated copies of input files with "_deduplicated" suffix'
    )

    return parser.parse_args()


def print_results_table(results_list):
    """Print results in a formatted table"""
    if not results_list:
        print("No files processed.")
        return

    # Filter out None results
    results_list = [r for r in results_list if r is not None]

    if not results_list:
        print("No valid results to display.")
        return

    # Print header
    print("\n" + "=" * 135)
    print(f"{'Filename':<30} {'Rows':<10} {'Dupe Pairs':<12} {'Estimated N':<12} {'Clusters':<10} {'To Remove':<10} {'Time (s)':<10}")
    print("=" * 135)

    # Print results
    for result in results_list:
        estimated_n_str = f"{result['estimated_n']:.4f}" if result['estimated_n'] != float('inf') else "∞"
        time_str = f"{result['compute_time']:.2f}"
        print(f"{result['filename']:<30} {result['num_rows']:<10} {result['duplicate_pairs']:<12} "
              f"{estimated_n_str:<12} {result['num_clusters']:<10} {result['to_remove']:<10} {time_str:<10}")

    print("=" * 135)
    print(f"Total files processed: {len(results_list)}")
    total_rows = sum(r['num_rows'] for r in results_list)
    total_duplicates = sum(r['duplicate_pairs'] for r in results_list)
    total_to_remove = sum(r['to_remove'] for r in results_list)
    total_time = sum(r['compute_time'] for r in results_list)

    print(f"\nSummary:")
    print(f"Total rows: {total_rows}")
    print(f"Total duplicate pairs: {total_duplicates}")
    print(f"Total items to remove: {total_to_remove}")
    print(f"Total compute time: {total_time:.2f}s")


def main():
    """Main function"""
    args = parse_args()

    # Initialize detector (n_clusters=None means dynamic sqrt-based clustering)
    detector = DuplicateDetector(
        similarity_threshold=args.threshold,
        n_clusters=args.clusters,  # None means use sqrt-based dynamic clustering
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    # Get files to process
    files = detector.get_files_to_process(args.path)

    if not files:
        print(f"No files found to process at: {args.path}")
        sys.exit(1)

    detector.log(f"Found {len(files)} file(s) to process")

    # Set up deduplication directory structure if needed
    dedupe_base_dir = None
    if args.deduplicate:
        dedupe_base_dir = detector.setup_deduplicated_directory(args.path, files)
        if dedupe_base_dir:
            detector.log(f"Created deduplicated directory: {dedupe_base_dir}")

    # Process files
    results = []
    for filepath in files:
        detector.log(f"\nProcessing: {filepath}")
        result = detector.process_file(
            filepath,
            create_deduplicated=args.deduplicate,
            base_input_path=args.path,
            dedupe_base_dir=dedupe_base_dir
        )
        if result:
            results.append(result)

    # Print results table
    print_results_table(results)

    # Print deduplication summary if enabled
    if args.deduplicate:
        deduplicated_files = [r.get('deduplicated_file') for r in results if r.get('deduplicated_file')]
        if deduplicated_files:
            print(f"\nCreated {len(deduplicated_files)} deduplicated file(s):")
            for filepath in deduplicated_files:
                print(f"  {filepath}")
        else:
            print("\nNo duplicates found in any files - no deduplicated files created.")


if __name__ == "__main__":
    main()
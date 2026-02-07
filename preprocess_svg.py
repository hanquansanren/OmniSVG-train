#!/usr/bin/env python3
"""
SVG Data Processing Script
Supports both single file and batch directory processing.
"""

import os
import argparse
import subprocess
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox


def preprocess_svg(input_path: str, output_path: str, timeout: int = 30) -> bool:
    """
    Simplify SVG syntax using picosvg, removing groups and transforms.
    
    Args:
        input_path: Path to input SVG file
        output_path: Path to save preprocessed SVG
        timeout: Timeout in seconds for picosvg (default: 30)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, "w") as output_file:
            subprocess.run(
                ["picosvg", input_path], 
                stdout=output_file, 
                check=True,
                timeout=timeout
            )
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout preprocessing {input_path} (exceeded {timeout}s)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing {input_path}: {e}")
        return False


def process_svg(
    svg: SVG,
    scale: float,
    width: int,
    height: int,
    simplify: bool = False,
    max_dist: int = 5
) -> SVG:
    """
    Apply transformations to SVG.
    
    Args:
        svg: SVG object to process
        scale: Zoom scale factor
        width: Target width for normalization
        height: Target height for normalization
        simplify: Whether to simplify paths
        max_dist: Maximum distance for path splitting
        
    Returns:
        Processed SVG object
    """
    svg.zoom(scale)
    svg.normalize(Bbox(width, height))
    
    if simplify:
        svg.simplify_arcs()
        svg.simplify_heuristic()
        svg.split(max_dist=max_dist)
    
    return svg


def process_single_file(
    input_path: str,
    output_path: str,
    scale: float,
    width: int,
    height: int,
    simplify: bool,
    max_dist: int,
    timeout: int = 30,
    skip_picosvg: bool = False
) -> bool:
    """
    Process a single SVG file.
    
    Args:
        input_path: Path to input SVG file
        output_path: Path to save processed SVG
        scale: Zoom scale factor
        width: Target width
        height: Target height
        simplify: Whether to simplify paths
        max_dist: Maximum distance for path splitting
        timeout: Timeout for picosvg preprocessing (default: 30)
        skip_picosvg: Skip picosvg preprocessing and load directly (default: False)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        if not skip_picosvg:
            # Preprocess with picosvg
            if not preprocess_svg(input_path, output_path, timeout=timeout):
                return False

            # Check if output file is empty
            if os.path.getsize(output_path) == 0:
                print(f"Skipping {input_path}: preprocessed file is empty")
                os.remove(output_path)
                return False
            
            load_path = output_path
        else:
            # Skip picosvg, load directly from input
            load_path = input_path

        # Load and process SVG
        try:
            svg = SVG.load_svg(load_path)
        except Exception as e:
            print(f"Error loading SVG {load_path}: {e}")
            if not skip_picosvg and os.path.exists(output_path):
                os.remove(output_path)
            return False
            
        try:
            svg = process_svg(svg, scale, width, height, simplify=simplify, max_dist=max_dist)
        except Exception as e:
            print(f"Error processing SVG {input_path}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        # Save processed SVG
        svg.save_svg(output_path)
        print(f"✓ Successfully processed: {os.path.basename(input_path)}")
        return True

    except Exception as e:
        print(f"Unexpected error processing {input_path}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def process_directory(
    input_dir: str,
    output_dir: str,
    scale: float,
    width: int,
    height: int,
    simplify: bool,
    max_dist: int,
    timeout: int = 30,
    skip_picosvg: bool = False
) -> tuple:
    """
    Process all SVG files in a directory.
    
    Args:
        input_dir: Input directory containing SVG files
        output_dir: Output directory for processed files
        scale: Zoom scale factor
        width: Target width
        height: Target height
        simplify: Whether to simplify paths
        max_dist: Maximum distance for path splitting
        timeout: Timeout for picosvg preprocessing (default: 30)
        skip_picosvg: Skip picosvg preprocessing and load directly (default: False)
        
    Returns:
        Tuple of (success_count, failure_count)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failure_count = 0
    
    # Collect all SVG files first
    svg_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".svg"):
                svg_files.append(os.path.join(root, filename))
    
    total_files = len(svg_files)
    print(f"Found {total_files} SVG files to process")
    if skip_picosvg:
        print("Mode: Direct loading (skipping picosvg)\n")
    else:
        print(f"Mode: With picosvg preprocessing (timeout: {timeout}s)\n")
    
    for idx, input_path in enumerate(svg_files, 1):
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        
        print(f"[{idx}/{total_files}] Processing: {os.path.basename(input_path)}")
        
        if process_single_file(
            input_path, output_path,
            scale, width, height, simplify, max_dist, timeout, skip_picosvg
        ):
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(
        description="SVG Data Processing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python preprocess_svg.py --input file.svg --output processed.svg
  
  # Process directory (with picosvg)
  python preprocess_svg.py --input_dir ./svgs --output_dir ./output
  
  # Process directory (skip picosvg, for already simplified SVGs)
  python preprocess_svg.py --input_dir ./svgs --output_dir ./output --skip_picosvg
  
  # Process with simplification
  python preprocess_svg.py --input_dir ./svgs --output_dir ./output --simplify
        """
    )
    
    # Input/Output options
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Single SVG file to process"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for single file processing"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing SVG files for batch processing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for batch processing"
    )
    
    # Processing options
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="SVG zoom scale factor (default: 1.0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=200,
        help="Output SVG width (default: 200)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=200,
        help="Output SVG height (default: 200)"
    )
    parser.add_argument(
        "--max_dist",
        type=int,
        default=5,
        help="Max path length before splitting (default: 5)"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Enable path simplification"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for picosvg in seconds (default: 30)"
    )
    parser.add_argument(
        "--skip_picosvg",
        action="store_true",
        help="Skip picosvg preprocessing (for already simplified SVGs like font glyphs)"
    )

    args = parser.parse_args()

    # Validate arguments
    single_mode = args.input is not None
    batch_mode = args.input_dir is not None
    
    if not single_mode and not batch_mode:
        parser.error("Please specify --input for single file or --input_dir for batch processing")
    
    if single_mode and batch_mode:
        parser.error("Cannot use both --input and --input_dir simultaneously")
    
    if single_mode:
        # Single file processing
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_processed{ext}"
        
        success = process_single_file(
            args.input, args.output,
            args.scale, args.width, args.height,
            args.simplify, args.max_dist, args.timeout, args.skip_picosvg
        )
        
        if success:
            print(f"\nProcessing complete: {args.output}")
        else:
            print(f"\nProcessing failed: {args.input}")
            exit(1)
    
    else:
        # Batch processing
        if args.output_dir is None:
            args.output_dir = f"{args.input_dir}_processed"
        
        success, failure = process_directory(
            args.input_dir, args.output_dir,
            args.scale, args.width, args.height,
            args.simplify, args.max_dist, args.timeout, args.skip_picosvg
        )
        
        print(f"\nBatch processing complete:")
        print(f"  Success: {success}")
        print(f"  Failed:  {failure}")


if __name__ == "__main__":
    main()

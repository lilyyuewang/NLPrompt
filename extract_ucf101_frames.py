#!/usr/bin/env python3
"""
Extract middle frames from UCF101 videos and organize them according to datasets/ucf101.py requirements.

This script:
1. Extracts middle frame from each .avi video as .jpg
2. Converts CamelCase class names to snake_case (e.g., ApplyEyeMakeup -> Apply_Eye_Makeup)
3. Organizes frames into UCF-101-midframes/ directory structure

Uses ffmpeg for frame extraction (more reliable than OpenCV for video processing).
"""

import os
import re
import subprocess
from pathlib import Path
from tqdm import tqdm

def camel_to_snake(name):
    """Convert CamelCase to snake_case (matching ucf101.py line 75-76)"""
    elements = re.findall("[A-Z][^A-Z]*", name)
    return "_".join(elements)

def extract_middle_frame_ffmpeg(video_path, output_path):
    """Extract middle frame from video using ffmpeg"""
    try:
        # Get video duration
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        
        if duration <= 0:
            return False
        
        # Calculate middle frame time
        middle_time = duration / 2.0
        
        # Extract frame at middle time
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-ss', str(middle_time), '-vframes', '1',
            '-q:v', '2',  # High quality
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    # Check dependencies
    if not check_ffmpeg():
        print("Error: ffmpeg and ffprobe are required but not found.")
        print("Install with: sudo apt install ffmpeg")
        print("Or: conda install -c conda-forge ffmpeg")
        return
    
    # Paths
    base_dir = Path(os.path.expanduser("~/datasets/nlprompt/ucf101"))
    video_dir = base_dir / "UCF-101"
    output_dir = base_dir / "UCF-101-midframes"
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    print(f"Source: {video_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Find all video files
    video_files = list(video_dir.rglob("*.avi"))
    print(f"Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("No .avi files found!")
        return
    
    # Process each video
    success_count = 0
    fail_count = 0
    
    for video_path in tqdm(video_files, desc="Extracting frames"):
        # Get relative path from UCF-101/
        rel_path = video_path.relative_to(video_dir)
        
        # Get class name (CamelCase) and filename
        class_name_camel = rel_path.parent.name
        filename = rel_path.name
        
        # Convert class name to snake_case
        class_name_snake = camel_to_snake(class_name_camel)
        
        # Convert .avi to .jpg
        jpg_filename = filename.replace(".avi", ".jpg")
        
        # Create output path
        output_path = output_dir / class_name_snake / jpg_filename
        
        # Extract frame
        if extract_middle_frame_ffmpeg(video_path, output_path):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(video_files)}")
    print()
    print(f"Frames saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

import os
import shutil
import argparse

# Files to be removed
TEMP_FILES = [
    "list_tables.py",
    "db_schema_inspector.py",
    "check_mri_tables.py",
    "test_mri.py",
    "test_roi.py",
    "mri_recommendations.md",
    "system_files_summary.md"
]

def cleanup(dry_run=True):
    """
    Clean up temporary files from the project.
    
    Args:
        dry_run: If True, just print the files that would be deleted without deleting them
    """
    print(f"{'DRY RUN: ' if dry_run else ''}Cleaning up temporary files...")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Count files deleted
    deleted_count = 0
    
    # Delete temporary files
    for filename in TEMP_FILES:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            if dry_run:
                print(f"Would delete: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    # Clean up test directories
    test_dirs = ['test_images', 'test_results']
    for dir_name in test_dirs:
        dir_path = os.path.join(current_dir, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if dry_run:
                print(f"Would delete directory: {dir_path}")
            else:
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory: {dir_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {e}")
    
    # Summary
    if dry_run:
        print(f"\nDRY RUN: Would have deleted {deleted_count} files/directories")
        print("Run with --force to actually delete files")
    else:
        print(f"\nCleanup complete. Deleted {deleted_count} files/directories")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up temporary files and directories")
    parser.add_argument('--force', action='store_true', help='Actually delete files (default is dry run)')
    args = parser.parse_args()
    
    cleanup(dry_run=not args.force) 
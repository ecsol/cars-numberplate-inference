#!/usr/bin/env python3
"""
Test backup logic - verify backup path is correct
"""

import os
import tempfile
import shutil


def test_backup_path_logic():
    """Test that backup path is in the same folder as the file"""

    # Simulate the backup path logic from fetch_today_images.py
    test_cases = [
        # (full_path, expected_backup_path)
        (
            "/mnt/s3/upfile/1234567G/image.jpg",
            "/mnt/s3/upfile/1234567G/.backup/image.jpg",
        ),
        (
            "/mnt/s3/upfile/1234567G/20220824190333_1.jpg",
            "/mnt/s3/upfile/1234567G/.backup/20220824190333_1.jpg",
        ),
        (
            "/mnt/s3/upfile/ABC123/subdir/photo.png",
            "/mnt/s3/upfile/ABC123/subdir/.backup/photo.png",
        ),
    ]

    print("=" * 60)
    print("Testing backup path logic")
    print("=" * 60)

    all_passed = True
    for full_path, expected_backup in test_cases:
        # Logic from fetch_today_images.py
        file_name = os.path.basename(full_path)
        file_dir = os.path.dirname(full_path)
        backup_dir = os.path.join(file_dir, ".backup")
        backup_path = os.path.join(backup_dir, file_name)

        passed = backup_path == expected_backup
        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"\n{status}")
        print(f"  Input:    {full_path}")
        print(f"  Expected: {expected_backup}")
        print(f"  Got:      {backup_path}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    return all_passed


def test_backup_file_operations():
    """Test actual file backup operations"""

    print("\n" + "=" * 60)
    print("Testing file backup operations")
    print("=" * 60)

    # Create temp directory to simulate S3 mount
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure: /tmpdir/upfile/車両コード/
        car_dir = os.path.join(tmpdir, "upfile", "1234567G")
        os.makedirs(car_dir)

        # Create test image
        test_file = os.path.join(car_dir, "test_image.jpg")
        with open(test_file, "wb") as f:
            f.write(b"fake image content")

        print(f"\n1. Created test file: {test_file}")

        # Apply backup logic
        file_name = os.path.basename(test_file)
        file_dir = os.path.dirname(test_file)
        backup_dir = os.path.join(file_dir, ".backup")
        backup_path = os.path.join(backup_dir, file_name)

        print(f"2. Backup dir: {backup_dir}")
        print(f"3. Backup path: {backup_path}")

        # Create backup
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)  # mkdir, not makedirs
            print(f"4. Created .backup folder with os.mkdir()")

        shutil.copy2(test_file, backup_path)
        print(f"5. Copied file to backup")

        # Verify
        backup_exists = os.path.exists(backup_path)
        print(f"6. Backup exists: {backup_exists}")

        # Verify content
        with open(backup_path, "rb") as f:
            content = f.read()
        content_match = content == b"fake image content"
        print(f"7. Content matches: {content_match}")

        # Verify structure
        print(f"\n8. Directory structure:")
        for root, dirs, files in os.walk(tmpdir):
            level = root.replace(tmpdir, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = "  " * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

        passed = backup_exists and content_match
        print(f"\n{'✓ PASS' if passed else '✗ FAIL'}")

    print("=" * 60)
    return passed


def test_restore_from_backup():
    """Test restore from backup"""
    
    print("\n" + "=" * 60)
    print("Testing restore from backup")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        car_dir = os.path.join(tmpdir, "upfile", "1234567G")
        os.makedirs(car_dir)
        
        # Create original file
        test_file = os.path.join(car_dir, "test.jpg")
        with open(test_file, "wb") as f:
            f.write(b"original content")
        
        # Create backup
        backup_dir = os.path.join(car_dir, ".backup")
        os.mkdir(backup_dir)
        backup_path = os.path.join(backup_dir, "test.jpg")
        shutil.copy2(test_file, backup_path)
        
        print(f"1. Original file: {test_file}")
        print(f"2. Backup created: {backup_path}")
        
        # Simulate processing (modify original)
        with open(test_file, "wb") as f:
            f.write(b"modified content after processing")
        
        print(f"3. File modified (simulating processing)")
        
        # Restore from backup
        shutil.copy2(backup_path, test_file)
        
        with open(test_file, "rb") as f:
            restored = f.read()
        
        passed = restored == b"original content"
        print(f"4. Restored content matches original: {passed}")
        print(f"\n{'✓ PASS' if passed else '✗ FAIL'}")
    
    print("=" * 60)
    return passed


def test_backup_never_overwritten():
    """Test that backup is NEVER overwritten once created"""
    
    print("\n" + "=" * 60)
    print("Testing backup is NEVER overwritten")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        car_dir = os.path.join(tmpdir, "upfile", "1234567G")
        os.makedirs(car_dir)
        
        # Create original file
        test_file = os.path.join(car_dir, "test.jpg")
        with open(test_file, "wb") as f:
            f.write(b"ORIGINAL - this must be preserved")
        
        # Create backup (first time)
        backup_dir = os.path.join(car_dir, ".backup")
        backup_path = os.path.join(backup_dir, "test.jpg")
        
        if not os.path.exists(backup_dir):
            os.mkdir(backup_dir)
        
        # Simulate first backup
        if not os.path.exists(backup_path):
            shutil.copy2(test_file, backup_path)
            print(f"1. First backup created: {backup_path}")
        
        # Verify backup content
        with open(backup_path, "rb") as f:
            backup_content = f.read()
        print(f"2. Backup content: {backup_content}")
        
        # Simulate file being modified (processed)
        with open(test_file, "wb") as f:
            f.write(b"MODIFIED - processed file")
        print(f"3. Original file modified to: MODIFIED - processed file")
        
        # Try to create backup again (SHOULD NOT OVERWRITE)
        # This simulates the safety check in fetch_today_images.py
        if os.path.exists(backup_path):
            print(f"4. Backup already exists - SKIPPING (no overwrite)")
        else:
            shutil.copy2(test_file, backup_path)
            print(f"4. WARNING: Backup was overwritten!")
        
        # Verify backup is still original
        with open(backup_path, "rb") as f:
            final_backup = f.read()
        
        passed = final_backup == b"ORIGINAL - this must be preserved"
        print(f"5. Backup still contains original: {passed}")
        print(f"   Backup content: {final_backup}")
        print(f"\n{'✓ PASS' if passed else '✗ FAIL'}")
    
    print("=" * 60)
    return passed


if __name__ == "__main__":
    results = []
    
    results.append(("Backup path logic", test_backup_path_logic()))
    results.append(("File operations", test_backup_file_operations()))
    results.append(("Restore from backup", test_restore_from_backup()))
    results.append(("Backup NEVER overwritten", test_backup_never_overwritten()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print("\n✗ Some tests failed!")
        exit(1)

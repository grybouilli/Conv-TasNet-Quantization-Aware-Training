#!/usr/bin/env python3
"""
Script to run VGG11 CIFAR100 experiments multiple times and collect logs.

This script will:
1. Run each experiment configuration 3 times
2. Capture stdout and stderr logs for each run
3. Copy all logs to a specified output directory
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil

# Experiment configurations
EXPERIMENTS = [
    "6_bits_fine_tuning.py",
    "7_bits_fine_tuning.py",
    "8_bits_fine_tuning.py",
    "16_bits_fine_tuning.py"
]

NUM_RUNS = 1
SCRIPT_DIR = Path(__file__).parent.absolute()


def run_experiment(script_name: str, run_number: int, temp_log_dir: Path, output_dir: Path) -> bool:
    """
    Run a single experiment and capture its output.

    Args:
        script_name: Name of the Python script to run
        run_number: Run number (1-indexed)
        temp_log_dir: Temporary directory to write logs during execution
        output_dir: Final destination directory to copy logs after completion

    Returns:
        True if successful, False otherwise
    """
    script_path = SCRIPT_DIR / script_name
    log_name = f"{script_name[:-3]}_run{run_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = temp_log_dir / log_name

    print(f"\n{'=' * 80}")
    print(f"Running: {script_name} (Run {run_number}/{NUM_RUNS})")
    print(f"Log file: {log_path}")
    print(f"{'=' * 80}\n")

    try:
        with open(log_path, 'w') as log_file:
            # Write header to log file
            log_file.write(f"Experiment: {script_name}\n")
            log_file.write(f"Run: {run_number}\n")
            log_file.write(f"Started: {datetime.now().isoformat()}\n")
            log_file.write(f"{'=' * 80}\n\n")
            log_file.flush()

            # Run the experiment
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=SCRIPT_DIR
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
                log_file.flush()

            process.wait()

            # Write footer to log file
            log_file.write(f"\n{'=' * 80}\n")
            log_file.write(f"Finished: {datetime.now().isoformat()}\n")
            log_file.write(f"Exit code: {process.returncode}\n")

            # Copy log to final destination immediately after completion
            dest_path = output_dir / log_name
            shutil.copy2(log_path, dest_path)

            if process.returncode == 0:
                print(f"\n✓ Successfully completed {script_name} (Run {run_number})")
                print(f"✓ Log copied to: {dest_path}")
                return True
            else:
                print(f"\n✗ Failed {script_name} (Run {run_number}) with exit code {process.returncode}")
                print(f"✓ Log copied to: {dest_path}")
                return False

    except Exception as e:
        print(f"\n✗ Error running {script_name} (Run {run_number}): {e}")
        return False


def main():
    """Main execution function."""
    print("VGG11 CIFAR100 Experiment Runner")
    print("=" * 80)

    # Get output directory from user
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        print("\nWhere should the logs be saved?")
        print("Enter the path to the output directory:")
        output_path = input("> ").strip()

        if not output_path:
            print("No output directory specified. Using default: ./experiment_logs")
            output_dir = SCRIPT_DIR / "experiment_logs"
        else:
            output_dir = Path(output_path)

    # Create output directory
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_log_dir = SCRIPT_DIR / f"temp_logs_{timestamp}"
    temp_log_dir.mkdir(exist_ok=True)

    print(f"\nTemporary logs during execution: {temp_log_dir}")
    print(f"Final logs will be copied to: {output_dir}")
    print(f"\nRunning {len(EXPERIMENTS)} experiments, {NUM_RUNS} times each")
    print(f"Total runs: {len(EXPERIMENTS) * NUM_RUNS}\n")

    input("Press Enter to start the experiments...")

    # Track results
    results = []
    total_runs = len(EXPERIMENTS) * NUM_RUNS
    completed = 0

    start_time = datetime.now()

    # Run all experiments
    for experiment in EXPERIMENTS:
        for run_num in range(1, NUM_RUNS + 1):
            success = run_experiment(experiment, run_num, temp_log_dir, output_dir)
            results.append({
                'experiment': experiment,
                'run': run_num,
                'success': success
            })
            completed += 1
            print(f"\nProgress: {completed}/{total_runs} runs completed")

    end_time = datetime.now()
    duration = end_time - start_time

    # Clean up temporary directory
    print(f"\n{'=' * 80}")
    print("Cleaning up temporary files...")
    print(f"{'=' * 80}\n")
    try:
        shutil.rmtree(temp_log_dir)
        print(f"✓ Removed temporary directory: {temp_log_dir}")
    except Exception as e:
        print(f"✗ Error removing temporary directory: {e}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}\n")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Duration: {duration}")
    print(f"\nLogs saved to: {output_dir}\n")

    # Print detailed results
    print("Detailed Results:")
    print("-" * 80)
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['experiment']:40} Run {result['run']}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()

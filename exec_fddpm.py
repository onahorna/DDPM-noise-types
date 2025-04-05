import os
import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

# Execute the scripts in sequence
try:
    # Step 1: Train and Evaluate Models
    run_script('forward_reverse_vis.py')

    # Step 2: Apply Best Models and Evaluate Metrics
    run_script('metric_evaluation.py')

    print("Pipeline executed successfully.")
except Exception as e:
    print(f"Error occurred: {e}")

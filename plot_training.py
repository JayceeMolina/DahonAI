from ultralytics.utils.plotting import plot_results
import os

# YOLO training results file
RESULTS_FILE = "runs/segment/train/results.csv"

# Check if training results exist
if os.path.exists(RESULTS_FILE):
    # Generate training graphs
    plot_results(file=RESULTS_FILE)
    print("Training graphs generated successfully!")
else:
    print("results.csv not found. Train the model first.")
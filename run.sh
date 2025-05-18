#!/bin/bash
# filepath: /Users/prachit/self/Working/YOLO-Pose/run.sh

# Define common paths
RESULTS_DIR="results"
VIDEO_PATH="../Person_New/input/3c.mp4"
FEATURES_CSV="${RESULTS_DIR}/gait_features.csv"
FLAT_NPY="${RESULTS_DIR}/gait_features_flat.npy"
BBOX_INFO="${RESULTS_DIR}/bbox_info.json"
MERGED_DATA="${RESULTS_DIR}/gait_features_flat_merged.npy"
MODEL_OUTPUT="${RESULTS_DIR}/best_model.pt"

# Step 1: Generate raw data from video
echo "🎥 Extracting gait features from video..."
python main.py \
  --video "${VIDEO_PATH}" \
  --start_frame 0 \
  --end_frame 500 \
  --headless \
  --buffer_size 0.05 \
  --output_video "${RESULTS_DIR}/temp_osnet_ain_1.mp4" \
  --output_features "${FEATURES_CSV}" \
  --save_bbox_info

# Step 2: Label and merge identities
echo "👤 Labeling and merging identities..."
python map_identities.py \
  --features "${FEATURES_CSV}" \
  --sequential "${FLAT_NPY}" \
  --video "${VIDEO_PATH}" \
  --bbox_info "${BBOX_INFO}" \
  --save_sequences \
  --merged_output "${MERGED_DATA}"

# Step 3: Train LSTM model
echo "🧠 Training LSTM classifier..."
python train_lstm2.py \
  --data_path "${MERGED_DATA}" \
  --model_save_path "${MODEL_OUTPUT}" \
  --results_dir "${RESULTS_DIR}"

echo "✅ Workflow complete!"
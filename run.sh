# # Colab
# python main.py \
#   --video "/content/drive/MyDrive/datasets/My Movie.mp4" \
#   --start_frame 0\
#   --headless \
#   --buffer_size 0.05 \
#   --output_video "/content/drive/MyDrive/datasets/temp_osnet_ain_1.mp4" \
#   --export_sequences \
#   --output_features "/content/drive/MyDrive/datasets/gait_features_n.csv" \
#   --save_crops

# Local
python main.py \
  --video "../Person_New/input/3c.mp4" \
  --start_frame 0\
  --end_frame 300 \
  --headless \
  --buffer_size 0.05 \
  --output_video "temp_osnet_ain_1.mp4" \
  --export_sequences \
  --output_features "gait_features.csv" \
  --save_bbox_info
  # --save_crops
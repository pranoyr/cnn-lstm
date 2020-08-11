rm -rf ./data/annotation
mkdir ./data/annotation
rm -rf ./data/image_data
mkdir ./data/image_data

python3 ./video_jpg_ucf101_hmdb51.py
python3 ./n_frames_ucf101_hmdb51.py
python3 ./gen_anns_list.py
python3 ./ucf101_json.py
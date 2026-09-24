# import json
# import os
# import subprocess
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import pandas as pd
# from tqdm import tqdm

# # Configure input and output
# INPUT_FILES = [
#     "dataset/auroracap/VDC_1k.jsonl",
# ]
# VIDEO_DIRS = [
#     "dataset/auroracap/videos/videos",
# ]
# OUTPUT_PATH = "dataset/auroracap/video_size.json"

# # Key that contains the video filename/id in the data
# # For example, if your data has {'video_uid': 'video1.mp4', ...}, set this to 'video_uid'
# VIDEO_KEY = "video_name"
# # Number of threads for parallel processing
# NUM_THREADS = 8
# # Supported video extensions (in order of preference)
# VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".mov", ".webm"]


# # Video ID extraction method - modify this function based on data format
# def extract_video_ids(data):
#     """Extract list of video IDs from input data"""
#     if isinstance(data, list):
#         # For list format, extract video IDs using the specified key
#         return [item[VIDEO_KEY] for item in data if VIDEO_KEY in item]
#     elif isinstance(data, dict):
#         # For dictionary format, video IDs might be under a specific key
#         # Modify based on actual data structure
#         if VIDEO_KEY in data:
#             return [data[VIDEO_KEY]]
#         else:
#             return list(data.keys())
#     elif isinstance(data, pd.DataFrame):
#         # For DataFrame format, use the specified column
#         return data[VIDEO_KEY].tolist() if VIDEO_KEY in data.columns else []
#     else:
#         raise ValueError(f"Unsupported data type: {type(data)}")


# def load_data(file_path):
#     """Load data from different formats based on file extension"""
#     file_ext = os.path.splitext(file_path)[1].lower()

#     try:
#         if file_ext == ".json":
#             with open(file_path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         elif file_ext == ".jsonl":
#             data = []
#             with open(file_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     data.append(json.loads(line.strip()))
#             return data
#         elif file_ext == ".parquet":
#             return pd.read_parquet(file_path)
#         else:
#             raise ValueError(f"Unsupported file format: {file_ext}")
#     except Exception as e:
#         print(f"Error loading file {file_path}: {e}")
#         return None


# def get_video_dimensions(video_path):
#     """Get video width, height and duration using ffprobe"""
#     cmd = [
#         "ffprobe",
#         "-v",
#         "error",
#         "-select_streams",
#         "v:0",
#         "-show_entries",
#         "stream=width,height,duration",
#         "-of",
#         "json",
#         video_path,
#     ]

#     result = subprocess.run(
#         cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#     )
#     if result.returncode == 0:
#         try:
#             info = json.loads(result.stdout)
#             if "streams" in info and len(info["streams"]) > 0:
#                 stream = info["streams"][0]
#                 width = stream.get("width")
#                 height = stream.get("height")
#                 duration = stream.get("duration")

#                 result = {"width": width, "height": height}
#                 if duration:
#                     result["duration"] = float(duration)
#                 return result
#         except json.JSONDecodeError:
#             pass
#     return None  # Mark as failed


# def process_item(video_id, video_dir):
#     """Process a single video file to get its dimensions"""
#     # Check if video_id already contains an extension
#     base, ext = os.path.splitext(video_id)
#     videoname = base.split("/")[-1]

#     if ext.lower() in VIDEO_EXTENSIONS:
#         # Video already has a valid extension
#         video_path = os.path.join(video_dir, video_id)
#         if os.path.exists(video_path):
#             try:
#                 dims = get_video_dimensions(video_path)
#                 if dims is not None:
#                     return {videoname: dims}
#             except Exception as e:
#                 print(f"Error processing {video_path}: {e}")
#     else:
#         # Try to guess the extension
#         for ext in VIDEO_EXTENSIONS:
#             video_path = os.path.join(video_dir, video_id + ext)
#             if os.path.exists(video_path):
#                 try:
#                     dims = get_video_dimensions(video_path)
#                     if dims is not None:
#                         return {videoname: dims}
#                 except Exception as e:
#                     print(f"Error processing {video_path}: {e}")

#     print(f"Video file not found: {video_id}")
#     return None


# def main():
#     """Main function"""
#     if len(INPUT_FILES) != len(VIDEO_DIRS):
#         print("Error: INPUT_FILES and VIDEO_DIRS must have the same length")
#         return

#     all_results = {}

#     for anno_file, video_dir in zip(INPUT_FILES, VIDEO_DIRS):
#         print(f"\nProcessing: {anno_file}")

#         # Load data
#         data = load_data(anno_file)
#         if data is None:
#             print(f"Skipping {anno_file}, unable to load data")
#             continue

#         # Extract video ID list
#         try:
#             video_list = extract_video_ids(data)
#             print(f"Found {len(video_list)} video IDs")
#         except Exception as e:
#             print(f"Error extracting video IDs: {e}")
#             continue

#         # Process video files with multithreading
#         results = {}
#         futures = []
#         with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
#             for video_id in video_list:
#                 futures.append(executor.submit(process_item, video_id, video_dir))

#             for future in tqdm(
#                 as_completed(futures),
#                 total=len(futures),
#                 desc=f"Processing video files",
#             ):
#                 result = future.result()
#                 if result is not None:
#                     results.update(result)

#         print(
#             f"Processed {len(results)}/{len(video_list)} videos ({len(results)/len(video_list)*100:.1f}%)"
#         )
#         all_results.update(results)

#     # Save results
#     os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
#     with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#         json.dump(all_results, f, indent=4, ensure_ascii=False)

#     print(f"\nAll results saved to: {OUTPUT_PATH}")
#     print(f"Total processed videos: {len(all_results)}")


# if __name__ == "__main__":
#     main()

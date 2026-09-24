#%%
import json
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import subprocess
import os


def get_video_metadata_ffmpeg(video_path):
    """
    Get comprehensive video metadata using ffmpeg/ffprobe.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing duration, fps, resolution, frame count, and other metadata.
              Returns None if video cannot be processed.
    """
    video_path = str(video_path)
    
    # Command to get all video stream information in JSON format
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,codec_name,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate,format_name",
        "-of", "json",
        video_path
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        
        if not data.get("streams") or len(data["streams"]) == 0:
            return None
            
        stream = data["streams"][0]
        format_info = data.get("format", {})
        
        # Calculate frame rate (try both r_frame_rate and avg_frame_rate)
        def parse_fraction(fraction_str):
            try:
                num, den = map(float, fraction_str.split('/'))
                return num / den if den != 0 else 0
            except:
                return 0
                
        r_frame_rate = parse_fraction(stream.get("r_frame_rate", "0/0"))
        avg_frame_rate = parse_fraction(stream.get("avg_frame_rate", "0/0"))
        fps = avg_frame_rate if avg_frame_rate > 0 else r_frame_rate
        
        # Get duration (prefer stream duration, fallback to format duration)
        duration = float(stream.get("duration", 0))
        if duration == 0 and "duration" in format_info:
            duration = float(format_info["duration"])
        
        # Get frame count (prefer nb_frames, fallback to duration*fps)
        frame_count = int(stream.get("nb_frames", 0))
        if frame_count == 0:
            frame_count = int(duration * fps)
        
        return {
            'resolution': [stream["width"], stream["height"]],
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
        }
        
    except Exception as e:
        print(f"Error processing video with ffprobe: {e}")
        return None


def seconds_to_mmss(seconds):
    """
    Convert seconds to mm:ss format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Time in mm:ss format
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--json_path',
        type=str,
        default='' 
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--video_root_dir',
        type=str,
        default=''
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of items to process (for testing)'
    )
    args = parser.parse_args()

    src_json_path = Path(args.json_path)
    output_path = Path(args.output_path)
    video_root_dir = Path(args.video_root_dir)

    # Load source data
    src = json.load(src_json_path.open('r'))
    if args.limit:
        src = src[:args.limit]
    
    tgt = []
    
    # Process each item
    for item in tqdm(src):
        try:
            # Get video path - keep original path format
            video_path = os.path.join(video_root_dir, item['image'])
            video_path = str(Path(video_path).resolve())
            
            # Get video metadata (optional, for validation)
            # info = get_video_metadata_ffmpeg(video_path=video_path)
            # if info is None:
            #     print(f"Warning: Could not get metadata for {video_path}, skipping...")
            #     continue
#             user_content =  """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event. Output your thought process within the <think> </think> tags.
# Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""
            user_content =  """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event. Provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time", For example: "12.54 to 17.83"."""
            conversations = item.get("conversations", [])
            user_question_raw = None
            reference_answer = None
            
            for conv in conversations:
                if conv.get("from") == "human":
                    user_question_raw = conv.get("value", "")
                elif conv.get("from") == "gpt":
                    reference_answer = conv.get("value", "")
            
            gt_times = item.get("gt_times")

    
            start_time, end_time = gt_times[0], gt_times[1]
            
            # 转换为 reference_response 格式（秒数格式，供奖励函数使用）
            # reference_response = convert_gt_times_to_reference_response(gt_times)
            # Keep original query text unchanged
            raw_query = user_question_raw
            # gt_duration_range = item['timestamp']  # [start_seconds, end_seconds]
            
            # Convert timestamps to mm:ss format (only format conversion, no content change)
            start_time_mmss = seconds_to_mmss(start_time)
            end_time_mmss = seconds_to_mmss(end_time)
            
            assistant_content = f"<think>\nGot it, let's find the segment that matches the description: \"{raw_query}\". Looking at the video, the timestamps for that part are {start_time_mmss} to {end_time_mmss}.\n</think>\nThe start and end timestamps of \"{raw_query}\" are: \\boxed{[{start_time_mmss}, {end_time_mmss}]}."
            
            # Create ms-swift format item
            ms_swift_item = {
                "videos": [video_path],
                "messages": [  {'role': 'system', 'content': 'You are a video analysis expert.'},
                             
                    {
                        "role": "user",
                        "content": "<video>\n"+user_content.replace("[EVENT]", raw_query)
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ]
            }
            
            tgt.append(ms_swift_item)
            
        except Exception as e:
            print(f"Error processing item: {e}\nItem: {item}")
            continue
    
    # Save to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(tgt, f, ensure_ascii=False, indent=4)
    
    print(f"\nConversion completed!")
    print(f"Total items processed: {len(tgt)}")
    print(f"Output saved to: {output_path}")

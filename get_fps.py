import cv2
import time

def get_true_fps(video_path):
    """
    Calculates the true FPS of a video by processing its frames.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the total number of frames from metadata (as a fallback)
    total_frames_metadata = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Metadata Frame Count: {total_frames_metadata}")


    frame_count = 0
    start_time = time.time()

    while True:
        # Read the next frame
        success, frame = video.read()

        # If we can't read a frame, the video has ended
        if not success:
            break

        frame_count += 1

    end_time = time.time()
    video.release()

    # Calculate the elapsed time and the true FPS
    elapsed_time = end_time - start_time
    true_fps = frame_count / elapsed_time

    print(f"Total frames counted: {frame_count}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return true_fps

if __name__ == "__main__":
    # Replace 'your_video.mov' with the path to your video file
    video_file = "2025-06-08 22-48-44.mov"
    calculated_fps = get_true_fps(video_file)

    if calculated_fps:
        print(f"\nCalculated True FPS: {calculated_fps:.2f}")
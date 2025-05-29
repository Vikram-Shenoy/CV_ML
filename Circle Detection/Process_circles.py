import cv2
import numpy as np
import os
import pandas as pd

def find_circle_data_and_annotate(image_path):
    """
    Detects circles in an image, returns their average midpoint (x, y) and radius,
    and an annotated image with the circle drawn.
    If one circle is detected, its data is used.
    If two or more circles are detected, data from the first two are averaged.
    Returns (x, y, radius, annotated_image_with_drawings) or (None, None, None, None).
    """
    img_original_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_original_color is None:
        # Error reading image will be handled by the calling function checking os.path.exists
        return None, None, None, None # x, y, radius, annotated_image

    # Prepare image for circle detection
    gray = cv2.cvtColor(img_original_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5) # Apply median blur to reduce noise

    # --- Crucial Parameters for HoughCircles ---
    # Adjust these parameters based on your specific images!
    # dp: Inverse ratio of accumulator resolution to image resolution.
    # minDist: Minimum distance between centers of detected circles.
    # param1: Higher threshold for the Canny edge detector.
    # param2: Accumulator threshold for circle centers. Smaller means more (possibly false) circles.
    # minRadius: Minimum circle radius to detect (in pixels).
    # maxRadius: Maximum circle radius to detect (in pixels).
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=30, param2=15, minRadius=5, maxRadius=30)  # EXAMPLE VALUES - TUNE THESE!

    final_x, final_y, final_r = None, None, None
    annotated_image_to_return = None # This will hold the image with drawings

    if circles is not None:
        # circles is a 3D array: [[[x1, y1, r1], [x2, y2, r2], ...]]
        detected_circles_data = []
        for c_data in circles[0, :]: # c_data is [x_center, y_center, radius]
            detected_circles_data.append((c_data[0], c_data[1], c_data[2]))

        if len(detected_circles_data) == 1:
            final_x, final_y, final_r = detected_circles_data[0]
            # You can uncomment the line below for debugging individual detections
            # print(f"  Found 1 circle in {os.path.basename(image_path)}: X={final_x:.2f}, Y={final_y:.2f}, R={final_r:.2f}")
        elif len(detected_circles_data) >= 2:
            # Average the first two detected circles
            c1_x, c1_y, c1_r = detected_circles_data[0]
            c2_x, c2_y, c2_r = detected_circles_data[1]
            final_x = (c1_x + c2_x) / 2.0
            final_y = (c1_y + c2_y) / 2.0
            final_r = (c1_r + c2_r) / 2.0
            # You can uncomment the line below for debugging individual detections
            # print(f"  Found {len(detected_circles_data)} circles in {os.path.basename(image_path)}. Averaged first two: X={final_x:.2f}, Y={final_y:.2f}, R={final_r:.2f}")
        # If len(detected_circles_data) == 0 (though circles was not None), final_x,y,r remain None.

    if final_x is not None and final_y is not None and final_r is not None:
        # A circle (or average) was successfully determined.
        # Prepare to draw on a fresh copy of the original color image.
        img_for_annotation = img_original_color.copy()
        
        center_cv = (int(round(final_x)), int(round(final_y)))
        radius_cv = int(round(final_r))

        if radius_cv > 0: # Only draw if radius is valid for drawing
            # Draw the circle outline (green)
            cv2.circle(img_for_annotation, center_cv, radius_cv, (0, 255, 0), 2)
            # Draw the midpoint (red filled circle)
            cv2.circle(img_for_annotation, center_cv, 5, (0, 0, 255), -1)
            annotated_image_to_return = img_for_annotation # Assign the image with drawings
        else:
            # Radius is not suitable for drawing (e.g., 0 or negative).
            # Data (final_x, final_y, final_r) is still valid for DataFrame.
            # annotated_image_to_return remains None, so no image is saved.
            # print(f"  Warning: Invalid radius ({final_r:.2f}) for drawing on {os.path.basename(image_path)}. Annotation drawing skipped.")
            pass
        
        return final_x, final_y, final_r, annotated_image_to_return
    else:
        # No circles found or processed (e.g., HoughCircles returned None or no valid circles extracted)
        # print(f"  No circles to annotate for {os.path.basename(image_path)}")
        return None, None, None, None # x, y, r, annotated_image

def process_all_cameras_to_dataframe(folder_path, num_images_per_camera=1500, camera_ids=['A', 'B', 'C', 'D'], output_viz_folder_path=None):
    """
    Processes images from multiple cameras, generates a DataFrame with circle data,
    and saves annotated images to the output_viz_folder_path.
    """
    all_rows_data = []

    if output_viz_folder_path:
        try:
            os.makedirs(output_viz_folder_path, exist_ok=True)
            # This message is now in __main__
            # print(f"Ensured visualization output directory exists: {output_viz_folder_path}")
        except OSError as e:
            print(f"Warning: Could not create/access visualization directory '{output_viz_folder_path}': {e}. Visualizations will be skipped.")
            output_viz_folder_path = None # Disable if creation/access failed

    print(f"Starting processing for {num_images_per_camera} image sets...")
    for i in range(1, num_images_per_camera + 1):
        image_index_str = f"{i:04d}" # Formats as 0001, 0002, ..., 1500
        current_row_data = {}

        if i % 100 == 0: # Print progress update
            print(f"Processing image index: {image_index_str}...")

        for cam_id in camera_ids:
            image_name = f"Camera_{cam_id}_{image_index_str}.png" # Assumes .png extension
            image_path = os.path.join(folder_path, image_name)

            # Initialize with NaN for DataFrame entries
            x_coord, y_coord, diameter = np.nan, np.nan, np.nan

            if os.path.exists(image_path):
                detected_x, detected_y, detected_radius, annotated_img_for_saving = find_circle_data_and_annotate(image_path)

                if detected_x is not None and detected_y is not None and detected_radius is not None:
                    x_coord = detected_x
                    y_coord = detected_y
                    diameter = detected_radius * 2.0

                    # Save the annotated image if it was created and path is valid
                    if annotated_img_for_saving is not None and output_viz_folder_path:
                        viz_image_path = os.path.join(output_viz_folder_path, image_name)
                        try:
                            cv2.imwrite(viz_image_path, annotated_img_for_saving)
                        except Exception as e_save:
                            print(f"  Error saving annotated image {viz_image_path}: {e_save}")
            else:
                # Reduce verbosity of missing image warnings: print for first, every Nth, and last.
                if i == 1 or i % 200 == 0 or i == num_images_per_camera :
                     print(f"  Warning: Image not found - {image_path} (Data for this camera will be NaN)")

            current_row_data[f'Diameter_{cam_id}'] = diameter
            current_row_data[f'X_{cam_id}'] = x_coord
            current_row_data[f'Y_{cam_id}'] = y_coord
        
        all_rows_data.append(current_row_data)

    # Define the specific column order for the final DataFrame
    final_columns_ordered = []
    for cam_id in camera_ids:
        final_columns_ordered.extend([f'Diameter_{cam_id}', f'X_{cam_id}', f'Y_{cam_id}'])
    
    df = pd.DataFrame(all_rows_data, columns=final_columns_ordered)
    return df

# --- How to use ---
if __name__ == "__main__":
    # 0. Installation (if you haven't already):
    # pip install opencv-python numpy pandas

    # 1. IMPORTANT: Specify the path to your folder containing the images
    image_folder = 'blue_ball_sim'  # <--- CHANGE THIS TO YOUR ACTUAL FOLDER PATH

    # 2. Specify the number of images per camera
    num_images = 1500 # As per your description

    # 3. Specify Camera IDs if different from default
    cameras = ['A', 'B', 'C', 'D']

    # 4. Define output folder for annotated images (will be created if it doesn't exist)
    # This folder will be placed INSIDE your main image_folder.
    output_visualization_folder = 'output_annotated'

    # Attempt to create the visualization folder before starting processing
    if output_visualization_folder:
        try:
            os.makedirs(output_visualization_folder, exist_ok=True)
            print(f"Annotated images will be saved to: {output_visualization_folder}")
        except OSError as e:
            print(f"Error creating directory '{output_visualization_folder}' in main: {e}.")
            print("The processing function will attempt to create it again or skip saving visualizations.")
            # No need to set output_visualization_folder to None here; the function will handle it.

    if not os.path.isdir(image_folder):
        print(f"Error: Input image folder '{image_folder}' not found. Please check the path.")
    else:
        print(f"--- Starting Circle Detection Process ---")
        print(f"Input Image Folder: {image_folder}")
        print(f"Number of images per camera: {num_images}")
        print(f"Camera IDs: {cameras}")
        print("Make sure your images are named like: Camera_A_0001.png, Camera_B_0001.png, etc.")
        print("\nIMPORTANT: The parameters for cv2.HoughCircles in the 'find_circle_data_and_annotate' function")
        print(" (dp, minDist, param1, param2, minRadius, maxRadius) are CRITICAL.")
        print(" You will likely need to TUNE them based on your specific images for accurate detection.\n")
        
        results_df = process_all_cameras_to_dataframe(
            folder_path=image_folder,
            num_images_per_camera=num_images,
            camera_ids=cameras,
            output_viz_folder_path=output_visualization_folder # Pass the visualization path
        )
        
        print("\n--- Processing Complete ---")
        
        if not results_df.empty:
            print("\nDataFrame Head (First 5 rows):")
            print(results_df.head())
            print(f"\nDataFrame Shape: {results_df.shape}") # Expected: (num_images, 12)
            
            # 5. Optionally, save the DataFrame to a CSV file
            output_csv_filename = 'circle_detection_results.csv'
            # Save CSV in the main image_folder or a specific results folder
            output_csv_path = os.path.join(image_folder, output_csv_filename)
            print("\nOutput CSV PATH:")
            try:
                results_df.to_csv(output_csv_path, index=False, float_format='%.2f')
                print(f"\nResults DataFrame successfully saved to: {output_csv_path}")
            except Exception as e:
                print(f"\nError saving DataFrame to CSV: {e}")
        else:
            print("The resulting DataFrame is empty. Please check image paths, names, and detection parameters.")
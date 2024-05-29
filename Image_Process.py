import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata

def estimate_dot_radius(x, y, gray, circle_radius=27, min_radius=16, max_radius=40):
    print(f"Estimating dot radius at position: ({x}, {y})")
    circle_radius = int(circle_radius)  # Ensure the radius is an integer
    # Create a mask to extract the local region around the dot
    mask = np.zeros((circle_radius*2+1, circle_radius*2+1), dtype=np.uint8)
    cv2.circle(mask, (circle_radius, circle_radius), circle_radius, 255, -1)
    
    # Ensure that the region around the dot is within the bounds of the image
    y_start = max(0, y - circle_radius)
    y_end = min(gray.shape[0], y + circle_radius + 1)
    x_start = max(0, x - circle_radius)
    x_end = min(gray.shape[1], x + circle_radius + 1)

    # Extract the local region from the grayscale image
    local_region = gray[y_start:y_end, x_start:x_end]
    
    # Resize the mask to match the size of the local region
    resized_mask = cv2.resize(mask, (local_region.shape[1], local_region.shape[0]))
    
    # Apply the resized mask to the local region
    masked_region = cv2.bitwise_and(local_region, resized_mask)
    
    # Find contours in the masked region
    contours, _ = cv2.findContours(masked_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Calculate the radius of the fitted circle
        (x_c, y_c), radius = cv2.minEnclosingCircle(contours[0])
        radius = int(radius)

        # Clamp the radius within a certain range
        # radius = min(max(radius, min_radius), max_radius)
        print(f"Estimated radius: {radius}")
        return radius
    else:
        print("No contour found, using default radius.")
        return circle_radius
    
def dynamic_radius_parameters(dot_positions, gray, std_radius):
    sample_radii = []
    for y, x in dot_positions:
        dot_radius = estimate_dot_radius(x, y, gray, std_radius)
        sample_radii.append(dot_radius)
    
    avg_radius = np.mean(sample_radii) if sample_radii else std_radius
    min_radius = int(avg_radius * 0.1)
    max_radius = int(avg_radius * 2.5)
    return int(avg_radius), min_radius, max_radius

def estimate_depth(ir_dot_positions, gray, camera_intrinsics, dot_spacing, circle_radius, min_radius, max_radius):
    depths = []
    radii = []
    # print(f"Estimating depth for {len(ir_dot_positions)} dot positions.")
    for i, dot_position in enumerate(ir_dot_positions):
        y, x = dot_position
        # print(f"Processing dot {i+1} at position: ({x}, {y})")
        dot_radius = estimate_dot_radius(x, y, gray, circle_radius, min_radius, max_radius)
        radii.append(dot_radius)

        if dot_radius == 0:
            # print(f"Skipping dot {i+1} due to zero radius.")
            continue

        depth = (camera_intrinsics[0, 0] * dot_spacing) / (2 * dot_radius)
        depth = depth * 420  # Adjusting the scale for better visualization
        depths.append(depth)
        # print(f"Estimated depth: {depth}")

    return depths, radii

#=======================================================================================================================#

# Load the image
image_path = "dotMatrix/Depth_camera/real_dot/capture_image_IR2_20cm.jpg"
original_dot = cv2.imread(image_path)

# if original_dot is None:
#     print("Error: Image not found. Check the file path.")
#     exit()

# Convert to grayscale
gray = cv2.cvtColor(original_dot, cv2.COLOR_BGR2GRAY)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Apply Gaussian blur
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Adaptive Gaussian Thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

# Otsu's Thresholding
_, otsu_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# # Invert the threshold images
# neg_adaptive = cv2.bitwise_not(adaptive_thresh)
# neg_otsu = cv2.bitwise_not(otsu_thresh)

# Find contours
# adaptive_thresh = cv2.bitwise_not(adaptive_thresh)
contours_adaptive, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_otsu, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract centroids of the dots
ir_dot_positions_adaptive = []
ir_dot_positions_otsu = []

# Extract centroids from adaptive thresholding
for contour in contours_adaptive:
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ir_dot_positions_adaptive.append((cy, cx))

# Extract centroids from Otsu's thresholding
for contour in contours_otsu:
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ir_dot_positions_otsu.append((cy, cx))

# Define camera intrinsic parameters
fx = fy = 3.6  # Focal length in mm
cx = cy = 1280 / 2  # Assuming the image dimensions are 1280x720
camera_intrinsics = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

dot_spacing = 0.01  # Example dot spacing in meters

# Adjustable parameters for dot radius estimation
circle_radius = 28
min_radius = 10
max_radius = 50
# ir_dot_positions = []
# std_circle_radius = 29
# circle_radius, min_radius, max_radius = dynamic_radius_parameters(ir_dot_positions, gray, std_circle_radius)

# Estimate depths from IR dot positions
depths_adaptive, radii_adaptive = estimate_depth(ir_dot_positions_adaptive, gray, camera_intrinsics, dot_spacing, circle_radius, min_radius, max_radius)
depths_otsu, radii_otsu = estimate_depth(ir_dot_positions_otsu, gray, camera_intrinsics, dot_spacing, circle_radius, min_radius, max_radius)

# Plot the images and results in one figure
fig, axs = plt.subplots(3, 3, figsize=(18, 18))

axs[0, 0].imshow(cv2.cvtColor(original_dot, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(gray, cmap='gray')
axs[0, 1].set_title('Grayscale Image')
axs[0, 1].axis('off')

axs[0, 2].imshow(gray_blur, cmap='gray')
axs[0, 2].set_title('Blurred Image')
axs[0, 2].axis('off')

# axs[1, 0].imshow(neg_adaptive, cmap='gray')
# axs[1, 0].set_title('Inverted Adaptive Threshold Image')
# axs[1, 0].axis('off')

axs[1, 1].imshow(adaptive_thresh, cmap='gray')
axs[1, 1].set_title('Adaptive Threshold Image')
axs[1, 1].axis('off')

axs[1, 2].imshow(otsu_thresh, cmap='gray')
axs[1, 2].set_title('Otsu Threshold Image')
axs[1, 2].axis('off')

# Scatter plot for adaptive thresholding
scatter_adaptive = axs[2, 0].scatter(
    [pos[1] for pos in ir_dot_positions_adaptive], 
    [pos[0] for pos in ir_dot_positions_adaptive], 
    c=depths_adaptive, cmap='coolwarm', s=50
)
axs[2, 0].invert_yaxis()
fig.colorbar(scatter_adaptive, ax=axs[2, 0], label='Depth (m)')
axs[2, 0].set_title('IR Dot Depth Estimation (Adaptive)')
axs[2, 0].set_xlabel('X Position')
axs[2, 0].set_ylabel('Y Position')

# Scatter plot for Otsu's thresholding
scatter_otsu = axs[2, 1].scatter(
    [pos[1] for pos in ir_dot_positions_otsu], 
    [pos[0] for pos in ir_dot_positions_otsu], 
    c=depths_otsu, cmap='coolwarm', s=50
)
axs[2, 1].invert_yaxis()
fig.colorbar(scatter_otsu, ax=axs[2, 1], label='Depth (m)')
axs[2, 1].set_title('IR Dot Depth Estimation (Otsu)')
axs[2, 1].set_xlabel('X Position')
axs[2, 1].set_ylabel('Y Position')

# Detected dots with estimated radii (adaptive)
ax_adaptive = axs[2, 2]
ax_adaptive.imshow(gray, cmap='gray')
for (y, x), radius in zip(ir_dot_positions_adaptive, radii_adaptive):
    circle = plt.Circle((x, y), radius, color='r', fill=False)
    ax_adaptive.add_patch(circle)
ax_adaptive.set_title('Detected Dots with Estimated Radii (Adaptive)')
ax_adaptive.set_xlabel('X Position')
ax_adaptive.set_ylabel('Y Position')
ax_adaptive.invert_yaxis()

plt.tight_layout()
plt.show()

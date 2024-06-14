import numpy as np
import matplotlib.pyplot as plt
import cv2

def estimate_depth_per_dot(ir_dot_positions, camera_intrinsics, dot_spacing):
    depths = []
    for i, (y, x) in enumerate(ir_dot_positions):
        # Estimating depth based on the radius of detected dots
        depth = (camera_intrinsics[0, 0] * dot_spacing) / (2 * x)  # Simplified calculation for depth per dot
        depth = depth * 1750  # Adjusting the scale for better visualization
        if(depth > 2):
            depths.append(2)
            depth = 2
        else:
            depths.append(depth)
        print(f"Dot {i+1}: Position ({x}, {y}), Estimated Depth: {depth}")

    return depths

# Load the image
image_path = "dotMatrix/Depth_camera/real_dot/capture_image_IR2_slope_20to40.jpg"
original_dot = cv2.imread(image_path)

if original_dot is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(original_dot, cv2.COLOR_BGR2GRAY)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Apply Gaussian blur
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# Adaptive Gaussian Thresholding
_, adaptive_thresh = cv2.threshold(gray_blur, 155, 255, cv2.THRESH_BINARY)

# Otsu's Thresholding
_, otsu_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
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

# Estimate depths from IR dot positions
depths_adaptive = estimate_depth_per_dot(ir_dot_positions_adaptive, camera_intrinsics, dot_spacing)
depths_otsu = estimate_depth_per_dot(ir_dot_positions_otsu, camera_intrinsics, dot_spacing)

# Plot the images and results in one figure
fig, axs = plt.subplots(3, 3, figsize=(8, 8))

axs[0, 0].imshow(cv2.cvtColor(original_dot, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(gray, cmap='gray')
axs[0, 1].set_title('Grayscale Image')
axs[0, 1].axis('off')

axs[0, 2].imshow(gray_blur, cmap='gray')
axs[0, 2].set_title('Blurred Image')
axs[0, 2].axis('off')

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

# Detected dots with centroids (adaptive)
ax_adaptive = axs[2, 2]
ax_adaptive.imshow(gray, cmap='gray')
for (y, x) in ir_dot_positions_adaptive:
    ax_adaptive.plot(x, y, 'ro')  # Plot the centroid
ax_adaptive.set_title('Detected Dots with Centroids (Adaptive)')
ax_adaptive.set_xlabel('X Position')
ax_adaptive.set_ylabel('Y Position')
ax_adaptive.invert_yaxis()

plt.tight_layout()
plt.show()

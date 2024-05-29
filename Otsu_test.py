import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata

def estimate_dot_radius(x, y, gray, circle_radius=28):
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
        
        return radius
    else:
        # Return a default radius if no contour is found
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
    for dot_position in ir_dot_positions:
        y, x = dot_position
        
        # Estimate dot radius based on local pixel intensities
        dot_radius = estimate_dot_radius(x, y, gray, circle_radius)
        radii.append(dot_radius)
        
        # Clamp the radius within a certain range
        dot_radius = min(max(dot_radius, min_radius), max_radius)
        
        # Calculate depth using depth from defocus formula
        depth = (camera_intrinsics[0, 0] * dot_spacing) / (2 * dot_radius)
        depth = depth * 15  # Adjusting the scale for better visualization
        depths.append(depth)
    return depths, radii

#===================================================================================================================#

# Load the image
image_path = "dotMatrix/Depth_camera/real_dot/capture_image_IR2_20cm.jpg"  # Change to your new image path
original_dot = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(original_dot, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# Invert the threshold image
neg = cv2.bitwise_not(threshold)

# Find contours of the dots
contours, _ = cv2.findContours(neg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pos_contours, __ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract centroids of the dots
ir_dot_positions = []

# Negative picture
for contour in contours:
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ir_dot_positions.append((cy, cx))

# Positive picture
for pos_con in pos_contours:
    M = cv2.moments(pos_con)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ir_dot_positions.append((cy, cx))

# Plot the images
plt.figure(figsize=(12, 10))

plt.subplot(221), plt.imshow(cv2.cvtColor(original_dot, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(222), plt.imshow(gray, cmap='gray'), plt.title('Grayscale Image')
plt.subplot(223), plt.imshow(neg, cmap='gray'), plt.title('Inverted Threshold Image')
plt.subplot(224), plt.imshow(threshold, cmap='gray'), plt.title('Threshold Image')

# Define camera intrinsic parameters
fx = fy = 3.6  # Focal length in mm
cx = cy = 1280 / 2  # Assuming the image dimensions are 1280x720
camera_intrinsics = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

dot_spacing = 0.3  # Example dot spacing in meters

# Standard circle radius for initial estimate
std_circle_radius = 29

# Dynamically adjust radius parameters
circle_radius, min_radius, max_radius = dynamic_radius_parameters(ir_dot_positions, gray, std_circle_radius)

# Estimate depths from IR dot positions
depths, radii = estimate_depth(ir_dot_positions, gray, camera_intrinsics, dot_spacing, circle_radius, min_radius, max_radius)

# Print estimated depths
for i, (depth, radius) in enumerate(zip(depths, radii)):
    print(f"Depth of IR dot {i+1}: {depth} meters, Estimated radius: {radius} pixels")

# Plot the positions and depths of IR dots
x_data = [pos[0] for pos in ir_dot_positions]
y_data = [pos[1] for pos in ir_dot_positions]

plt.figure()
scatter = plt.scatter(y_data, x_data, c=depths, cmap='coolwarm', s=50)
plt.colorbar(scatter, label='Depth (m)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('IR Dot Depth Estimation - Top View (X-Y Plane)')
plt.gca().invert_yaxis()

# Visualize detected dots with estimated radii
fig, ax = plt.subplots()
ax.imshow(gray, cmap='gray')
for (y, x), radius in zip(ir_dot_positions, radii):
    circle = plt.Circle((x, y), radius, color='r', fill=False)
    ax.add_patch(circle)
plt.title('Detected Dots with Estimated Radii')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.gca().invert_yaxis()
plt.show()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(y_data, x_data, depths, c=depths, cmap='coolwarm', s=50)
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Depth (m)')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Depth (m)')
ax.set_title('IR Dot Depth Estimation')

plt.show()

# Heatmap
x_grid = np.linspace(min(x_data), max(x_data), 1000)
y_grid = np.linspace(min(y_data), max(y_data), 1000)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Interpolate only for actual dot positions
depths_interp = griddata((x_data, y_data), depths, (X_grid, Y_grid), method='cubic', fill_value=np.nan)

plt.figure()
plt.imshow(depths_interp, extent=(min(y_data), max(y_data), min(x_data), max(x_data)), cmap='coolwarm', origin='lower')
plt.colorbar(label='Depth (m)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('IR Dot Depth Estimation - Heatmap (X-Y Plane)')
plt.show()

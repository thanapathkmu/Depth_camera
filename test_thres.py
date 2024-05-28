import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import griddata


def estimate_dot_radius(x, y, camera_intrinsics, dot_spacing):
    # Function to estimate dot radius based on local pixel intensities
    
    # Radius estimation pa*
    # 6
    # 6*rameters (you may need to adjust these)
    circle_radius = 1
    min_radius = 0.5
    max_radius = 3
    
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
        radius = min(max(radius, min_radius), max_radius)
        
        return radius
    else:
        # Return a default radius if no contour is found
        return circle_radius




def estimate_depth(ir_dot_positions, camera_intrinsics, dot_spacing):
    # Function to estimate depth from IR dot positions using depth from defocus

    # Placeholder for demo purposes
    depths = []

    for dot_position in ir_dot_positions:
        # Assuming dot_position is in pixel coordinates
        y, x = dot_position
        
        # Assuming dot_radius is known (you may need to estimate this)
        # Estimate dot radius based on local pixel intensities
        dot_radius = estimate_dot_radius(x, y, camera_intrinsics, dot_spacing)
        
        # Calculate depth using depth from defocus formula
        depth = (camera_intrinsics[0, 0] * dot_spacing) / (2 * dot_radius)
        depths.append(depth)

    return depths

# Load the image
original_dot = cv2.imread(r"dotMatrix/dot4.png")

# Convert to grayscale
gray = cv2.cvtColor(original_dot, cv2.COLOR_BGR2GRAY)

# Use adaptive thresholding
threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Invert
positive_mask = threshold
neg = cv2.bitwise_not(threshold)

# Find contours of the dots
contours, _ = cv2.findContours(neg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pos_contour, __ = cv2.findContours(positive_mask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

# Extract centroids of the dots
ir_dot_positions = []
for contour in contours:
    M = cv2.moments(contour)
    if M['m00'] != 0:  # Ensure m00 is non-zero
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ir_dot_positions.append((cy, cx))  # Note: swapping y and x for compatibility with estimate_depth

# Print the positions of white dots
for i, position in enumerate(ir_dot_positions, start=1):
    print(f"Dot {i}: Position {position}")

# Display images
plt.subplot(311), plt.imshow(original_dot), plt.title('Original Image')
plt.subplot(312), plt.imshow(gray, cmap='gray'), plt.title('Gray scale image')
plt.subplot(313), plt.imshow(neg, cmap='gray'), plt.title('Neg Image')

# Define camera intrinsic parameters
fx = fy = 3.6  # Focal length in mm
cx = cy = 1280 / 2  # Image dimensions are typically 2592x1944 for a 5MP camera

camera_intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])  # Example camera intrinsic matrix

dot_spacing = 0.1  # Example dot spacing in meters

# Estimate depths from IR dot positions
depths = estimate_depth(ir_dot_positions, camera_intrinsics, dot_spacing)

# Print estimated depths
for i, depth in enumerate(depths):
    print(f"Depth of IR dot {i+1}: {depth} meters")

###==========================================================================================================###

# Extract x, y, depth data
x_data = [pos[0] for pos in ir_dot_positions]
y_data = [pos[1] for pos in ir_dot_positions]

# Plot the data in the x-y plane
plt.figure()
scatter = plt.scatter(y_data, x_data, c=depths, cmap='coolwarm', s=50)
plt.colorbar(scatter, label='Depth (m)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('IR Dot Depth Estimation - Top View (X-Y Plane)')
plt.gca().invert_yaxis()  # Flip the y-axis

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
scatter = ax.scatter(y_data, x_data, depths, c=depths, cmap='coolwarm', s=50)

# Add color bar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Depth (m)')

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Depth (m)')
ax.set_title('IR Dot Depth Estimation')

# Show plot
plt.show()
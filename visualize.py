import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
# Define the rotation matrices for each camera position
rotation_vectors = [
    np.array([[-0.85913147], [0.40380124], [0.78385345]]),
    np.array([[-0.90750011], [0.69932734], [1.22749267]]),
    np.array([[-0.34659975], [0.39171762], [0.34059831]]),
    np.array([[-0.36554556], [-0.07221283], [-0.08528849]]),
    np.array([[-0.01849744], [0.0606542], [0.59606906]]),
    np.array([[-0.30965678], [-0.00408575], [-1.23395303]]),
    np.array([[-0.46457099], [0.29465183], [1.22171207]]),
    np.array([[-0.03685892], [0.0326239], [0.02356698]]),
    np.array([[-1.04384274], [0.43966879], [0.76240032]]),
    np.array([[-0.13487104], [0.13475011], [0.64607641]]),
    np.array([[-0.10542908], [0.0158558], [-0.16851786]]),
    np.array([[-0.05541833], [0.1710177], [1.26220019]]),
    np.array([[0.0912582], [-0.07217582], [-0.53615629]]),
    np.array([[-0.3048061], [0.26574635], [1.3998372]]),
    np.array([[-0.71656577], [-0.05039592], [0.03748014]]),
    np.array([[-0.35167572], [-0.19387882], [-0.18193717]])
]

# Define the translation vectors
translation_vectors = [
    np.array([[-0.46384914], [-4.84802931], [18.04085014]]),
    np.array([[0.70315967], [-4.5113684], [23.68228256]]),
    np.array([[-1.50088072], [-4.1202176], [20.14611845]]),
    np.array([[-2.61593328], [-4.6755266], [16.16116929]]),
    np.array([[0.84808828], [-6.76013567], [15.71858598]]),
    np.array([[-2.83215101], [-0.45649547], [16.74825406]]),
    np.array([[1.68314423], [-5.10503874], [18.20466334]]),
    np.array([[-0.66182918], [-5.92136399], [14.25706003]]),
    np.array([[-0.56176315], [-4.66307442], [21.57871644]]),
    np.array([[0.84685322], [-6.34785442], [18.68452998]]),
    np.array([[-2.55512409], [-3.72187095], [17.34606183]]),
    np.array([[2.52333983], [-7.68594592], [18.14252603]]),
    np.array([[-6.14670037], [-1.09930409], [15.49740136]]),
    np.array([[1.37234325], [-7.1851819], [16.81539637]]),
    np.array([[-3.47816408], [-4.52712623], [18.47959175]]),
    np.array([[-4.88261465], [-3.93119075], [14.61417232]])
]

# Convert the rotation vectors to rotation matrices
rotation_matrices = [cv2.Rodrigues(vec)[0] for vec in rotation_vectors]

# Create a list of camera positions for each rotation matrix
camera_positions = []
for i in range(len(rotation_matrices)):
    # Create a 4x4 transformation matrix from the rotation and translation
    # components of the extrinsic parameters
    camera_matrix = np.eye(4)
    camera_matrix[:3, :3] = rotation_matrices[i]
    camera_matrix[:3, 3] = translation_vectors[i].flatten()

    # Invert the camera matrix to obtain the camera position
    camera_position = np.linalg.inv(camera_matrix)[:3, 3]

    # Add the camera position to the list
    camera_positions.append(camera_position)

# Plot the camera positions in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([p[0] for p in camera_positions], [p[1] for p in camera_positions], [p[2] for p in camera_positions])
for i, p in enumerate(camera_positions):
    ax.text(p[0], p[1], p[2], str(i+1))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
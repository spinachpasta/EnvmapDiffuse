import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt


def exr_to_numpy(exr_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(exr_path)

    # Get the header to know the dimensions
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the channel types
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the channels
    channels = ["R", "G", "B"]
    channel_data = []

    for channel in channels:
        # Read the channel data as bytes
        channel_str = exr_file.channel(channel, pt)
        # Convert the channel data to a numpy array
        channel_np = np.frombuffer(channel_str, dtype=np.float32)
        # Reshape the array to the image dimensions
        channel_np = channel_np.reshape((height, width))
        channel_data.append(channel_np)

    # Stack the channels to create the final image array
    img_np = np.stack(channel_data, axis=-1)

    return img_np


def plot_image(img_np):
    plt.imshow(img_np, cmap="viridis")
    plt.axis("off")  # Hide axes
    plt.show()


def relu(x):
    return (x > 0) * x


# create kernel which represents diffuse reflection strength
def create_normal_kernel(img_shape, cx, cy):
    height, width, _ = img_shape

    # cx = width//2
    # cy = height//2

    c_theta = cx / width * np.pi * 2
    c_phi = cy / height * np.pi

    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, np.pi, height)
    X, Y = np.meshgrid(x, y)

    # ray vector in 3D Euclid space
    # [np.cos(Y), np.sin(Y) * np.cos(X), np.sin(Y) * np.sin(X)]
    # normal vector in 3D Euclid space
    # [np.cos(c_phi), np.sin(c_phi) * np.cos(c_theta), np.sin(c_phi) * np.sin(c_theta)]

    # take dot product between normal and ray vectors
    normal = relu(
        np.cos(Y) * np.cos(c_phi)
        + np.sin(Y) * np.cos(X) * np.sin(c_phi) * np.cos(c_theta)
        + np.sin(Y) * np.sin(X) * np.sin(c_phi) * np.sin(c_theta)
    )

    return np.repeat(normal[:, :, np.newaxis], 3, axis=2)


# Jacobian of equirectangular coordinates
def create_jacobian_window(img_shape):
    height, width, _ = img_shape
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, np.pi, height + 1, endpoint=False)[1:]
    _, Y = np.meshgrid(x, y)

    jacobian = abs(np.sin(Y))

    return np.repeat(jacobian[:, :, np.newaxis], 3, axis=2)


# Example usage
exr_path = "exr/city.exr"
# Original image
img_org = exr_to_numpy(exr_path)
# Convert to power density by multiplying Jacobian
img_density = img_org * create_jacobian_window(img_org.shape)
#plot_image(img_density)

# plot_image(create_normal_kernel(img_org.shape))
# plot_image(create_jacobian_window(img_np.shape))

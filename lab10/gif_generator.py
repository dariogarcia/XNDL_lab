import imageio

# List of output image paths
output_image_paths = ["output_image_epoch_"+str(i)+".jpg" for i in range(1,10)]

# Read images and create GIF
images = []
for image_path in output_image_paths:
    image = imageio.imread(image_path)
    images.append(image)

output_gif_path = "output_images.gif"  # Path to save the output GIF
imageio.mimsave(output_gif_path, images, loop=0, duration=100)  # Adjust duration as needed

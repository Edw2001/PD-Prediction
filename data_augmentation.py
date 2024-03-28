from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import random

###################ORIGIN SOURCE

health_source_dir = 'PaHaW_visualization/PaHaW_visualization/Health'
pd_source_dir = 'PaHaW_visualization/PaHaW_visualization/PD'


################################ FLIP IMAGE ###################################


def flip_image_LR(image_path, save_path):
    with Image.open(image_path) as img:
        # Flip the image horizontally
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Save the flipped image to the new path
        flipped_img.save(save_path)

def flip_image_TB(image_path, save_path):
    with Image.open(image_path) as img:
        # Flip the image horizontally
        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # Save the flipped image to the new path
        flipped_img.save(save_path)

def flip_images_in_directory(source_directory, target_directory, flip_function):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Process each file in the source directory
    for filename in os.listdir(source_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_directory, filename)
            new_file_path = os.path.join(target_directory, f"flipped_{filename}")
            flip_function(file_path, new_file_path)  # Use the provided flip function


health_target_dir_LR = 'data_augmentation/flipped_Image/Left_Right/flipped_HP'

pd_target_dir_LR = 'data_augmentation/flipped_Image/Left_Right/flipped_PD'

health_target_dir_TB = 'data_augmentation/flipped_Image/Top_Bottom/flipped_HP'

pd_target_dir_TB = 'data_augmentation/flipped_Image/Top_Bottom/flipped_PD'

# Apply the flipping and save the augmented images in new directories


# flip_images_in_directory(health_source_dir, health_target_dir_LR, flip_image_LR)
# flip_images_in_directory(pd_source_dir, pd_target_dir_LR, flip_image_LR)

# flip_images_in_directory(health_source_dir, health_target_dir_TB, flip_image_TB)
# flip_images_in_directory(pd_source_dir, pd_target_dir_TB, flip_image_TB)


################################ ROTATE IMAGE ###################################

random_degree = lambda: random.randint(1, 359)

def rotate_image(image_path, save_path, rotate_func=random_degree):
    with Image.open(image_path) as img:
        # Generate a random angle using the rotate_func
        rotate_degrees = rotate_func()
        # Rotate the image
        rotated_img = img.rotate(rotate_degrees, expand=True)
        # Save the rotated image to the new path
        rotated_img.save(save_path)

def rotate_images_in_directory(source_directory, target_directory, rotate_func=random_degree):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Process each file in the source directory
    for filename in os.listdir(source_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_directory, filename)
            new_file_path = os.path.join(target_directory, f"rotated_{filename}")
            rotate_image(file_path, new_file_path, rotate_func)


# health_rotate_target_dir = 'data_augmentation/rotate_Image/rotate_Health' 
# pd_rotate_target_dir = 'data_augmentation/rotate_Image/rotate_PD' 

# rotate_images_in_directory(health_source_dir, health_rotate_target_dir)
# rotate_images_in_directory(pd_source_dir, pd_rotate_target_dir)





################################ SCALE IMAGE ###################################

# Define a lambda function to generate a random scale factor
random_scaling_factor = lambda: random.uniform(0.2, 1.8)

def scale_image(image_path, save_path, scaling_func=random_scaling_factor):
    with Image.open(image_path) as img:
        # Generate a random scaling factor
        scale_factor = scaling_func()
        # Calculate the new size and resize the image
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        # Replace ANTIALIAS with LANCZOS for recent versions of Pillow
        scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Save the scaled image to the new path
        scaled_img.save(save_path)

def scale_images_in_directory(source_directory, target_directory, scaling_func=random_scaling_factor):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    for filename in os.listdir(source_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_directory, filename)
            new_file_path = os.path.join(target_directory, f"scaled_{filename}")
            scale_image(file_path, new_file_path, scaling_func)

# health_scale_target_dir = 'data_augmentation/scale_Image/scale_Health' 
# pd_scale_target_dir = 'data_augmentation/scale_Image/scale_PD' 

# scale_images_in_directory(health_source_dir, health_scale_target_dir)
# scale_images_in_directory(pd_source_dir, pd_scale_target_dir)

################################ BRIGNTNESS IMAGE ###################################

random_brightness_factor = lambda: random.uniform(0.1, 2.0)

def adjust_brightness(image_path, save_path, brightness_func=random_brightness_factor):
    with Image.open(image_path) as img:
        brightness_factor = brightness_func()
        enhancer = ImageEnhance.Brightness(img)
        img_enhanced = enhancer.enhance(brightness_factor)
        img_enhanced.save(save_path)

def adjust_brightness_in_directory(source_directory, target_directory, brightness_func=random_brightness_factor):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    for filename in os.listdir(source_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_directory, filename)
            new_file_path = os.path.join(target_directory, f"brightness_{filename}")
            adjust_brightness(file_path, new_file_path, brightness_func)

# health_brightness_target_dir = 'data_augmentation/brightness_Image/brightness_Health'
# pd_brightness_target_dir = 'data_augmentation/brightness_Image/brightness_PD'

# adjust_brightness_in_directory(health_source_dir, health_brightness_target_dir)
# adjust_brightness_in_directory(pd_source_dir, pd_brightness_target_dir)
            

################################ BLUR IMAGE ###################################


random_blur_radius = lambda: random.uniform(1, 5)

def apply_random_blur(image_path, save_path, blur_func=random_blur_radius):
    with Image.open(image_path) as img:
        # Generate a random blur radius using the blur_func
        blur_radius = blur_func()
        # Apply Gaussian blur with the random radius
        blurred_img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        # Save the blurred image to the new path
        blurred_img.save(save_path)

def apply_random_blur_in_directory(source_directory, target_directory, blur_func=random_blur_radius):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Process each file in the source directory
    for filename in os.listdir(source_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_directory, filename)
            new_file_path = os.path.join(target_directory, f"blurred_{filename}")
            apply_random_blur(file_path, new_file_path, blur_func)


health_blur_target_dir = 'data_augmentation/blur_Image/blur_Health'  

pd_blur_target_dir = 'data_augmentation/blur_Image/blur_PD'

apply_random_blur_in_directory(health_source_dir, health_blur_target_dir)
apply_random_blur_in_directory(pd_source_dir, pd_blur_target_dir)
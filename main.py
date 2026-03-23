import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

def load_images(source_path: str, target_path: str):
    """Load grayscale images from given paths."""
    img1 = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Check the image paths, images not found.")
    return img1, img2

def visualize_images(img1, img2, titles=("Image 1", "Image 2")):
    """Display the two images."""
    plt.figure(figsize=(8, 4))
    for i, img in enumerate([img1, img2], start=1):
        plt.subplot(1, 2, i)
        plt.imshow(img, cmap='gray')
        plt.title(titles[i-1])
        plt.axis('off')
    plt.show()

def detect_sift_features(img, nfeatures=5000):
    """Detect SIFT keypoints and descriptors."""
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_descriptors(des1, des2, top_k=100):
    """Match descriptors using BFMatcher with cross-check and return top matches."""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:top_k]

def estimate_homography(kp1, kp2, matches):
    """Estimate homography using RANSAC and return inliers mask."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H, mask.ravel().tolist()

def warp_image(img, H, shape):
    """Warp image using homography matrix H to given shape."""
    return cv2.warpPerspective(img, H, (shape[1], shape[0]))

def visualize_alignment(img1, img2, img_aligned):
    """Display original images and aligned image."""
    plt.figure(figsize=(12, 4))
    for i, (img, title) in enumerate(zip([img1, img2, img_aligned],
                                         ["Image 1", "Image 2", "Aligned Image"]), start=1):
        plt.subplot(1, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

def draw_inlier_matches(img1, kp1, img2, kp2, matches, inliers):
    """Draw matches that are inliers."""
    inlier_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12, 8))
    plt.title("Inlier Matches After RANSAC")
    plt.imshow(img_matches)
    plt.axis('off')
    plt.show()

def create_registration_gif(img1, img2, H, output_path="results/registration_process_TV.gif", num_frames=40):
    """Generate a smooth GIF showing registration process."""
    gif_frames = []
    H_start = np.eye(3)
    h, w = img2.shape

    for alpha in np.linspace(0, 1, num_frames):
        H_interp = (1 - alpha) * H_start + alpha * H
        warped = cv2.warpPerspective(img1, H_interp, (w, h))
        overlay = cv2.addWeighted(img2, 0.5, warped, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        gif_frames.append(overlay_rgb)

    imageio.mimsave(output_path, gif_frames, duration=0.08)
    print(f"Smooth registration GIF saved at {output_path}")

def main():
    # Paths
    source_img = "images/TVS.jpg"
    target_img = "images/TVT.jpg"

    # Load and visualize images
    img1, img2 = load_images(source_img, target_img)
    visualize_images(img1, img2)

    # Detect features
    kp1, des1 = detect_sift_features(img1)
    kp2, des2 = detect_sift_features(img2)
    print(f"Detected {len(kp1)} and {len(kp2)} keypoints respectively.")

    # Match descriptors
    matches = match_descriptors(des1, des2)
    print(f"Number of good matches: {len(matches)}")

    # Estimate homography
    H, inliers = estimate_homography(kp1, kp2, matches)

    # Warp and visualize
    img_aligned = warp_image(img1, H, img2.shape)
    visualize_alignment(img1, img2, img_aligned)

    # Draw inlier matches
    draw_inlier_matches(img1, kp1, img2, kp2, matches, inliers)

    # Create GIF of registration process
    create_registration_gif(img1, img2, H)

if __name__ == "__main__":
    main()
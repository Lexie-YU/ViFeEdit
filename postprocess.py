import os
import cv2
import numpy as np


def compute_lab_stats(video_path, max_frames=200, step=10, resize_ratio=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    lab_pixels = []
    idx = 0
    sampled = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            if resize_ratio != 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=resize_ratio,
                    fy=resize_ratio,
                    interpolation=cv2.INTER_AREA
                )

        
            bgr = frame.astype(np.float32) / 255.0

       
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

            lab_pixels.append(lab.reshape(-1, 3))
            sampled += 1

            if sampled >= max_frames:
                break

        idx += 1

    cap.release()

    if len(lab_pixels) == 0:
        raise RuntimeError(f"have no frames: {video_path}")

    lab_all = np.concatenate(lab_pixels, axis=0)
    mean = lab_all.mean(axis=0)   # [L, a, b]
    std = lab_all.std(axis=0)     # [L, a, b]
    return mean, std


def transfer_color_frame_lab_float(
    frame,
    mean_c,
    std_c,
    mean_s,
    std_s,
    luma_strength=0.2,
    chroma_strength=1.0
):
    bgr = frame.astype(np.float32) / 255.0

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w, _ = lab.shape
    x = lab.reshape(-1, 3)

    eps = 1e-6
    scale = std_s / np.maximum(std_c, eps)

    x_transfer = (x - mean_c) * scale + mean_s

    x_new = x.copy()
    x_new[:, 0] = x[:, 0] * (1.0 - luma_strength) + x_transfer[:, 0] * luma_strength

    x_new[:, 1] = x[:, 1] * (1.0 - chroma_strength) + x_transfer[:, 1] * chroma_strength
    x_new[:, 2] = x[:, 2] * (1.0 - chroma_strength) + x_transfer[:, 2] * chroma_strength

    x_new[:, 0] = np.clip(x_new[:, 0], 0.0, 100.0)
    x_new[:, 1] = np.clip(x_new[:, 1], -127.0, 127.0)
    x_new[:, 2] = np.clip(x_new[:, 2], -127.0, 127.0)

    lab_new = x_new.reshape(h, w, 3).astype(np.float32)

    # LAB -> BGR(float32)
    out = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

    # clip to valid range
    out = np.clip(out, 0.0, 1.0)

    # float -> uint8
    out = (out * 255.0).round().astype(np.uint8)
    return out


def sharpen_image(img, amount=0.25, sigma=1.0):
    if amount <= 0:
        return img

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp


def create_video_writer(output_video, fps, width, height):
    ext = os.path.splitext(output_video)[1].lower()

    candidates = []
    if ext == ".avi":
        candidates = ["MJPG", "XVID"]
    elif ext == ".mkv":
        candidates = ["FFV1", "XVID", "MJPG"]
    elif ext == ".mp4":
        candidates = ["mp4v"]
    else:
        candidates = ["MJPG", "XVID", "mp4v"]

    for codec in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using codec: {codec}")
            return writer

    raise RuntimeError(f"cannot create output video: {output_video}")


def process_video_color_transfer(
    content_video,
    style_video,
    output_video,
    max_frames=200,
    step=10,
    resize_ratio=0.5,
    luma_strength=0.2,
    chroma_strength=1.0,
    sharpen_amount=0.2,
    sharpen_sigma=1.0
):
    mean_s, std_s = compute_lab_stats(
        style_video,
        max_frames=max_frames,
        step=step,
        resize_ratio=resize_ratio
    )

    mean_c, std_c = compute_lab_stats(
        content_video,
        max_frames=max_frames,
        step=step,
        resize_ratio=resize_ratio
    )

    print("style mean/std :", mean_s, std_s)
    print("content mean/std:", mean_c, std_c)

    cap_c = cv2.VideoCapture(content_video)
    if not cap_c.isOpened():
        raise RuntimeError(f"cannot open content_video: {content_video}")

    fps = cap_c.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0

    width = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = create_video_writer(output_video, fps, width, height)

    frame_idx = 0
    while True:
        ret, frame = cap_c.read()
        if not ret:
            break

        corrected = transfer_color_frame_lab_float(
            frame,
            mean_c=mean_c,
            std_c=std_c,
            mean_s=mean_s,
            std_s=std_s,
            luma_strength=luma_strength,
            chroma_strength=chroma_strength
        )

        if sharpen_amount > 0:
            corrected = sharpen_image(
                corrected,
                amount=sharpen_amount,
                sigma=sharpen_sigma
            )

        out.write(corrected)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"handled {frame_idx} frames...")

    cap_c.release()
    out.release()
    print(f"post processing complete: {output_video}")


if __name__ == "__main__":
    source_video = "/path/to/source/video.mp4"
    content_video = "/path/to/edited/video.mp4"

    output_video = "output.avi"

    process_video_color_transfer(
        content_video=content_video,
        style_video=source_video,
        output_video=output_video,
        max_frames=200,
        step=10,
        resize_ratio=0.5,
        luma_strength=0.0,   
        chroma_strength=1.0,   
        sharpen_amount=0.2,  
        sharpen_sigma=1.0
    )
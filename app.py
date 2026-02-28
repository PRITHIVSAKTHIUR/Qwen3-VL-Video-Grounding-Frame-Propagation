import colorsys
import gc
import tempfile
import re
import json
import ast
import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from typing import Iterator
from gradio.themes import Soft
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID_V = "prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX"
DTYPE = torch.bfloat16

print(f"Loading {MODEL_ID_V}...")
processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
model_v = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_V, attn_implementation="kernels-community/flash-attn3", trust_remote_code=True, torch_dtype=DTYPE
).to(device).eval()
print("Model loaded successfully.")

MAX_SECONDS = 5.0

SYSTEM_PROMPT = """You are a helpful assistant to detect objects in images. When asked to detect elements based on a description you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax] with the values being scaled between 0 and 1000. When there are more than one result, answer with a list of bounding boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]."""

POINT_SYSTEM_PROMPT = """You are a precise object pointing assistant. When asked to point to an object in an image, you must return ONLY the exact center coordinates of that specific object as [x, y] with values scaled between 0 and 1000 (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner).

Rules:
1. ONLY point to objects that exactly match the description given.
2. Do NOT point to background, empty areas, or unrelated objects.
3. If there are multiple matching instances, return [[x1, y1], [x2, y2], ...].
4. If no matching object is found, return an empty list [].
5. Return ONLY the coordinate numbers, no explanations or other text.
6. Be extremely precise — place the point at the exact visual center of each matching object."""

POINTS_REGEX = re.compile(r'(?:(\d+)\s*[.:])?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)')


def try_load_video_frames(video_path_or_url: str) -> tuple[list[Image.Image], dict]:
    cap = cv2.VideoCapture(video_path_or_url)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, {"num_frames": len(frames), "fps": float(fps_val) if fps_val > 0 else None}


def parse_bboxes_from_text(text: str) -> list[list[float]]:
    text = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL)
    nested = re.findall(r'\[\s*\[[\d\s,\.]+\](?:\s*,\s*\[[\d\s,\.]+\])*\s*\]', text)
    if nested:
        try:
            all_b = []
            for m in nested:
                parsed = json.loads(m)
                all_b.extend(parsed if isinstance(parsed[0], list) else [parsed])
            return all_b
        except (json.JSONDecodeError, IndexError):
            pass
    single = re.findall(
        r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    if single:
        return [[float(v) for v in m] for m in single]
    nums = re.findall(r'(\d+(?:\.\d+)?)', text)
    return [[float(nums[i]), float(nums[i + 1]), float(nums[i + 2]), float(nums[i + 3])] for i in
            range(0, len(nums) - 3, 4)] if len(nums) >= 4 else []


def parse_precise_points(text: str, image_w: int, image_h: int) -> list[tuple[float, float]]:
    text = re.sub(r'<think>.*?</think>', '', text.strip(), flags=re.DOTALL)
    raw_points = []
    nested = re.findall(r'\[\s*\[[\d\s,\.]+\](?:\s*,\s*\[[\d\s,\.]+\])*\s*\]', text)
    if nested:
        try:
            for m in nested:
                parsed = json.loads(m)
                if isinstance(parsed[0], list):
                    for p in parsed:
                        if len(p) >= 2:
                            raw_points.append((float(p[0]), float(p[1])))
                elif len(parsed) >= 2:
                    raw_points.append((float(parsed[0]), float(parsed[1])))
        except (json.JSONDecodeError, IndexError):
            pass
    if not raw_points:
        single = re.findall(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
        if single:
            for m in single:
                raw_points.append((float(m[0]), float(m[1])))
    if not raw_points:
        for match in POINTS_REGEX.finditer(text):
            raw_points.append((float(match.group(2)), float(match.group(3))))
    validated = []
    for sx, sy in raw_points:
        if not (0 <= sx <= 1000 and 0 <= sy <= 1000):
            continue
        px = sx / 1000 * image_w
        py = sy / 1000 * image_h
        if 0 <= px <= image_w and 0 <= py <= image_h:
            validated.append((px, py))
    if len(validated) > 1:
        deduped = [validated[0]]
        for pt in validated[1:]:
            if all(((pt[0] - ex[0]) ** 2 + (pt[1] - ex[1]) ** 2) ** 0.5 >= 15 for ex in deduped):
                deduped.append(pt)
        validated = deduped
    return validated


def bbox_to_mask(bbox_scaled: list[float], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x1 = max(0, min(int(bbox_scaled[0] / 1000 * width), width - 1))
    y1 = max(0, min(int(bbox_scaled[1] / 1000 * height), height - 1))
    x2 = max(0, min(int(bbox_scaled[2] / 1000 * width), width - 1))
    y2 = max(0, min(int(bbox_scaled[3] / 1000 * height), height - 1))
    mask[y1:y2, x1:x2] = 1.0
    return mask


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0.0


def bbox_center_distance(b1, b2):
    c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2 = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def pixel_point_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def overlay_masks_on_frame(frame: Image.Image, masks: dict, colors_map: dict, alpha=0.5) -> Image.Image:
    base = np.array(frame).astype(np.float32) / 255
    overlay = base.copy()
    for oid, mask in masks.items():
        if mask is None:
            continue
        color = np.array(colors_map.get(oid, (255, 0, 0)), dtype=np.float32) / 255
        if mask.ndim == 3:
            mask = mask.squeeze()
        m = np.clip(mask, 0, 1)[..., None]
        overlay = (1 - alpha * m) * overlay + (alpha * m) * color
    return Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8))


def pastel_color_for_prompt(prompt: str):
    hue = (sum(ord(c) for c in prompt) * 2654435761 % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def get_font(image_height: int):
    font_size = max(10, int(13 * image_height / 720))
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "arial.ttf",
        ]
        for fp in font_paths:
            try:
                return ImageFont.truetype(fp, font_size), font_size
            except OSError:
                continue
    except Exception:
        pass
    return ImageFont.load_default(), 13


def detect_objects_in_frame(frame: Image.Image, prompt: str) -> list[list[float]]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": frame},
                     {"type": "text", "text": f"Detect all instances of: {prompt}"}]}
    ]
    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_v(text=[text], images=[frame], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_v.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = out[:, inputs.input_ids.shape[1]:]
    txt = processor_v.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_bboxes_from_text(txt)


def detect_precise_points_in_frame(frame: Image.Image, prompt: str) -> list[tuple[float, float]]:
    w, h = frame.size
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": frame},
                     {"type": "text",
                      "text": f"Detect all instances of: {prompt}. Return only bounding boxes for objects that exactly match this description."}]}
    ]
    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_v(text=[text], images=[frame], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_v.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = out[:, inputs.input_ids.shape[1]:]
    txt = processor_v.batch_decode(generated, skip_special_tokens=True)[0]
    bboxes = parse_bboxes_from_text(txt)
    if bboxes:
        points = []
        for b in bboxes:
            bw = abs(b[2] - b[0])
            bh = abs(b[3] - b[1])
            if bw < 5 or bh < 5:
                continue
            if bw > 950 and bh > 950:
                continue
            cx = (b[0] + b[2]) / 2 / 1000 * w
            cy = (b[1] + b[3]) / 2 / 1000 * h
            if 0 <= cx <= w and 0 <= cy <= h:
                points.append((cx, cy))
        if len(points) > 1:
            deduped = [points[0]]
            for pt in points[1:]:
                if all(pixel_point_distance(pt, ex) >= 20 for ex in deduped):
                    deduped.append(pt)
            points = deduped
        if points:
            return points
    messages2 = [
        {"role": "system", "content": [{"type": "text", "text": POINT_SYSTEM_PROMPT}]},
        {"role": "user",
         "content": [{"type": "image", "image": frame},
                     {"type": "text",
                      "text": f"Point to the exact center of each '{prompt}' in this image. Only point to objects that are clearly '{prompt}', nothing else."}]}
    ]
    text2 = processor_v.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    inputs2 = processor_v(text=[text2], images=[frame], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out2 = model_v.generate(**inputs2, max_new_tokens=512, do_sample=False)
    generated2 = out2[:, inputs2.input_ids.shape[1]:]
    txt2 = processor_v.batch_decode(generated2, skip_special_tokens=True)[0]
    return parse_precise_points(txt2, w, h)


def run_model_inference(image: Image.Image, prompt: str) -> str:
    messages = [
        {"role": "user",
         "content": [
             {"type": "image", "image": image},
             {"type": "text", "text": prompt},
         ]}
    ]
    text = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_v(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_v.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = out[:, inputs.input_ids.shape[1]:]
    result = processor_v.batch_decode(generated, skip_special_tokens=True)[0]
    result = re.sub(r'<think>.*?</think>', '', result.strip(), flags=re.DOTALL).strip()
    return result

def safe_parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def annotate_image_detection(image: Image.Image, result: dict) -> Image.Image:
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image
    image = image.convert("RGB")
    original_width, original_height = image.size
    draw = ImageDraw.Draw(image)
    font, font_size = get_font(original_height)

    if "objects" in result and result["objects"]:
        colors_list = [
            (66, 133, 244), (234, 67, 53), (251, 188, 4), (52, 168, 83),
            (255, 109, 0), (171, 71, 188), (0, 172, 193), (255, 82, 82),
            (46, 125, 50), (121, 85, 72),
        ]
        for idx, obj in enumerate(result["objects"]):
            x_min = int(obj.get("x_min", 0.0) * original_width)
            y_min = int(obj.get("y_min", 0.0) * original_height)
            x_max = int(obj.get("x_max", 0.0) * original_width)
            y_max = int(obj.get("y_max", 0.0) * original_height)
            color = colors_list[idx % len(colors_list)]
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)
            label = obj.get("label", f"Object {idx + 1}")
            padding = max(2, int(4 * original_height / 720))
            label_y = max(0, y_min - int(20 * original_height / 720))
            tb = draw.textbbox((x_min, label_y), label, font=font)
            draw.rectangle(
                [(tb[0] - padding, tb[1] - padding), (tb[2] + padding, tb[3] + padding)],
                fill=color
            )
            draw.text((x_min, label_y), label, fill="white", font=font)
    return image


def annotate_image_points(image: Image.Image, result: dict) -> Image.Image:
    if not isinstance(image, Image.Image) or not isinstance(result, dict):
        return image
    image = image.convert("RGB")
    original_width, original_height = image.size
    draw = ImageDraw.Draw(image)

    if "points" in result and result["points"]:
        for idx, p in enumerate(result["points"]):
            px = int(p["x"] * original_width)
            py = int(p["y"] * original_height)
            r_outer = max(8, int(10 * original_height / 720))
            r_inner = max(5, int(7 * original_height / 720))
            r_dot = max(1, int(2 * original_height / 720))
            draw.ellipse((px - r_outer, py - r_outer, px + r_outer, py + r_outer), outline="white", width=2)
            draw.ellipse((px - r_inner, py - r_inner, px + r_inner, py + r_inner), fill=(255, 40, 40), outline=(255, 40, 40))
            draw.ellipse((px - r_dot, py - r_dot, px + r_dot, py + r_dot), fill=(255, 200, 200))
    return image

class TrackingState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_frames: list[Image.Image] = []
        self.video_fps: float | None = None
        self.masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        self.bboxes_by_frame: dict[int, dict[int, list[float]]] = {}
        self.color_by_obj: dict[int, tuple[int, int, int]] = {}
        self.color_by_prompt: dict[str, tuple[int, int, int]] = {}
        self.text_prompts_by_frame_obj: dict[int, dict[int, str]] = {}
        self.composited_frames: dict[int, Image.Image] = {}
        self.prompts: dict[str, list[int]] = {}
        self.next_obj_id: int = 1
        self.current_frame_idx: int = 0

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)


class PointTrackingState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.video_frames: list[Image.Image] = []
        self.video_fps: float | None = None
        self.points_by_frame: dict[int, list[tuple[float, float]]] = {}
        self.trails: list[list[tuple[int, float, float]]] = []
        self.composited_frames: dict[int, Image.Image] = {}
        self.prompt_text: str = ""
        self.current_frame_idx: int = 0

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)
        

def compose_tracking_frame(state: TrackingState, frame_idx: int) -> Image.Image:
    if state is None or not state.video_frames:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))
    frame = state.video_frames[frame_idx].copy()
    w, h = frame.size
    masks = state.masks_by_frame.get(frame_idx, {})
    if masks:
        frame = overlay_masks_on_frame(frame, masks, state.color_by_obj, alpha=0.5)
    bboxes = state.bboxes_by_frame.get(frame_idx, {})
    if bboxes:
        draw = ImageDraw.Draw(frame)
        font, font_size = get_font(h)
        padding = max(2, int(4 * h / 720))
        vert_offset = int(20 * h / 720)
        for oid, bbox in bboxes.items():
            color = state.color_by_obj.get(oid, (255, 255, 255))
            x1 = int(bbox[0] / 1000 * w)
            y1 = int(bbox[1] / 1000 * h)
            x2 = int(bbox[2] / 1000 * w)
            y2 = int(bbox[3] / 1000 * h)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            prompt = state.text_prompts_by_frame_obj.get(frame_idx, {}).get(oid, "")
            if prompt:
                label = f"{prompt} - ID{oid}"
                label_y = max(0, y1 - vert_offset)
                tb = draw.textbbox((x1, label_y), label, font=font)
                draw.rectangle(
                    [(tb[0] - padding, tb[1] - padding), (tb[2] + padding, tb[3] + padding)],
                    fill=color
                )
                draw.text((x1, label_y), label, fill="white", font=font)
    state.composited_frames[frame_idx] = frame
    return frame


def compose_point_frame(pt_state: PointTrackingState, frame_idx: int, trail_length: int = 12) -> Image.Image:
    if pt_state is None or not pt_state.video_frames:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(pt_state.video_frames) - 1))
    frame = pt_state.video_frames[frame_idx].copy()
    draw = ImageDraw.Draw(frame)
    RED = (255, 40, 40)
    DARK_RED = (180, 0, 0)

    for trail in pt_state.trails:
        trail_pts = [(tx, ty) for fi, tx, ty in trail if fi <= frame_idx and fi > frame_idx - trail_length]
        if len(trail_pts) >= 2:
            for t_idx in range(len(trail_pts) - 1):
                alpha_ratio = (t_idx + 1) / len(trail_pts)
                trail_color = (
                    int(DARK_RED[0] * alpha_ratio),
                    int(DARK_RED[1] * alpha_ratio),
                    int(DARK_RED[2] * alpha_ratio)
                )
                thickness = max(1, int(2 * alpha_ratio))
                x1t, y1t = int(trail_pts[t_idx][0]), int(trail_pts[t_idx][1])
                x2t, y2t = int(trail_pts[t_idx + 1][0]), int(trail_pts[t_idx + 1][1])
                draw.line([(x1t, y1t), (x2t, y2t)], fill=trail_color, width=thickness)

    points_f = pt_state.points_by_frame.get(frame_idx, [])
    for (px, py) in points_f:
        draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline="white", width=2)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=RED, outline=RED)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 200, 200))

    pt_state.composited_frames[frame_idx] = frame
    return frame


def update_tracking_display(state: TrackingState, frame_idx: int) -> Image.Image:
    if state is None or not state.video_frames:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))
    cached = state.composited_frames.get(frame_idx)
    if cached is not None:
        return cached
    return compose_tracking_frame(state, frame_idx)


def update_point_display(pt_state: PointTrackingState, frame_idx: int) -> Image.Image:
    if pt_state is None or not pt_state.video_frames:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(pt_state.video_frames) - 1))
    cached = pt_state.composited_frames.get(frame_idx)
    if cached is not None:
        return cached
    return compose_point_frame(pt_state, frame_idx)


def _get_active_prompts_tracking(state: TrackingState) -> str:
    if state is None or not state.prompts:
        return "**Active prompts:** None"
    prompts_str = ", ".join([f"'{p}' ({len(ids)} obj)" for p, ids in state.prompts.items()])
    return f"**Active prompts:** {prompts_str}"


def _get_active_prompts_points(pt_state: PointTrackingState) -> str:
    if pt_state is None or not pt_state.prompt_text:
        return "**Active prompts:** None"
    return f"**Active prompts:** '{pt_state.prompt_text}' ({len(pt_state.trails)} tracked points)"


def init_tracking_video(state: TrackingState, video) -> tuple[TrackingState, int, int, Image.Image, str]:
    state.reset()
    if isinstance(video, dict):
        path = video.get("name") or video.get("path") or video.get("data")
    else:
        path = video
    if not path:
        raise gr.Error("Invalid video input.")
    frames, info = try_load_video_frames(path)
    if not frames:
        raise gr.Error("No frames could be loaded from the video.")
    trimmed_note = ""
    fps_in = info.get("fps")
    max_frames_allowed = int(MAX_SECONDS * fps_in) if fps_in else len(frames)
    if len(frames) > max_frames_allowed:
        frames = frames[:max_frames_allowed]
        trimmed_note = f" (trimmed to {int(MAX_SECONDS)}s = {len(frames)} frames)"
    state.video_frames = frames
    state.video_fps = float(fps_in) if fps_in else None
    first_frame = frames[0]
    max_idx = len(frames) - 1
    status = f"Loaded {len(frames)} frames @ {state.video_fps or 'unknown'} fps{trimmed_note}. Ready for text prompting."
    return state, 0, max_idx, first_frame, status


def init_point_video(pt_state: PointTrackingState, video) -> tuple[PointTrackingState, int, int, Image.Image, str]:
    pt_state.reset()
    if isinstance(video, dict):
        path = video.get("name") or video.get("path") or video.get("data")
    else:
        path = video
    if not path:
        raise gr.Error("Invalid video input.")
    frames, info = try_load_video_frames(path)
    if not frames:
        raise gr.Error("No frames could be loaded from the video.")
    trimmed_note = ""
    fps_in = info.get("fps")
    max_frames_allowed = int(MAX_SECONDS * fps_in) if fps_in else len(frames)
    if len(frames) > max_frames_allowed:
        frames = frames[:max_frames_allowed]
        trimmed_note = f" (trimmed to {int(MAX_SECONDS)}s = {len(frames)} frames)"
    pt_state.video_frames = frames
    pt_state.video_fps = float(fps_in) if fps_in else None
    first_frame = frames[0]
    max_idx = len(frames) - 1
    status = f"Loaded {len(frames)} frames @ {pt_state.video_fps or 'unknown'} fps{trimmed_note}. Ready for point tracking."
    return pt_state, 0, max_idx, first_frame, status


@spaces.GPU
def apply_tracking_prompt_on_frame(
    state: TrackingState,
    frame_idx: int,
    text_prompt: str,
) -> tuple[Image.Image, str, str, TrackingState]:
    if state is None or not state.video_frames:
        return None, "Upload a video first.", "**Active prompts:** None", state
    if not text_prompt or not text_prompt.strip():
        ap = _get_active_prompts_tracking(state)
        return update_tracking_display(state, int(frame_idx)), "Please enter a text prompt.", ap, state

    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))
    frame = state.video_frames[frame_idx]
    w, h = frame.size

    prompt_texts = [p.strip() for p in text_prompt.split(",") if p.strip()]
    if not prompt_texts:
        ap = _get_active_prompts_tracking(state)
        return update_tracking_display(state, frame_idx), "Please enter a valid text prompt.", ap, state

    status_parts = [f"Processing on frame {frame_idx}:"]

    for prompt in prompt_texts:
        bboxes = detect_objects_in_frame(frame, prompt)

        if prompt not in state.color_by_prompt:
            state.color_by_prompt[prompt] = pastel_color_for_prompt(prompt)

        masks_f = state.masks_by_frame.setdefault(frame_idx, {})
        bboxes_f = state.bboxes_by_frame.setdefault(frame_idx, {})
        texts_f = state.text_prompts_by_frame_obj.setdefault(frame_idx, {})

        obj_ids_for_prompt = []
        for bbox in bboxes:
            oid = state.next_obj_id
            state.next_obj_id += 1
            state.color_by_obj[oid] = state.color_by_prompt[prompt]
            masks_f[oid] = bbox_to_mask(bbox, w, h)
            bboxes_f[oid] = bbox
            texts_f[oid] = prompt
            state.prompts.setdefault(prompt, []).append(oid)
            obj_ids_for_prompt.append(oid)

        if obj_ids_for_prompt:
            ids_str = ", ".join(map(str, obj_ids_for_prompt))
            status_parts.append(f"  • '{prompt}': {len(obj_ids_for_prompt)} object(s) (IDs: {ids_str})")
        else:
            status_parts.append(f"  • '{prompt}': No objects detected.")

    state.composited_frames.pop(frame_idx, None)
    status = "\n".join(status_parts)
    ap = _get_active_prompts_tracking(state)
    return update_tracking_display(state, frame_idx), status, ap, state

@spaces.GPU
def propagate_tracking(state: TrackingState) -> Iterator[tuple[TrackingState, str, dict]]:
    if state is None or not state.video_frames:
        yield state, "Load a video first.", gr.update()
        return
    if not state.prompts:
        yield state, "No prompts defined. Apply text prompt(s) on a frame first.", gr.update()
        return

    total = state.num_frames
    processed = 0

    yield state, f"Propagating: {processed}/{total}", gr.update()

    for prompt, obj_ids in list(state.prompts.items()):
        seed_frame_idx = None
        seed_bboxes_by_oid = {}
        for f_idx in sorted(state.bboxes_by_frame.keys()):
            for oid in obj_ids:
                if oid in state.bboxes_by_frame.get(f_idx, {}):
                    if seed_frame_idx is None:
                        seed_frame_idx = f_idx
                    if f_idx == seed_frame_idx:
                        seed_bboxes_by_oid[oid] = state.bboxes_by_frame[f_idx][oid]

        if seed_frame_idx is None:
            continue

        # Forward propagation
        prev_tracks = [(oid, seed_bboxes_by_oid[oid]) for oid in seed_bboxes_by_oid]

        for f_idx in range(seed_frame_idx + 1, total):
            frame = state.video_frames[f_idx]
            w, h = frame.size
            new_bboxes = detect_objects_in_frame(frame, prompt)

            masks_f = state.masks_by_frame.setdefault(f_idx, {})
            bboxes_f = state.bboxes_by_frame.setdefault(f_idx, {})
            texts_f = state.text_prompts_by_frame_obj.setdefault(f_idx, {})

            used = set()
            matched = {}
            scores = [
                (bbox_iou(pbbox, nbbox), pi, ni)
                for pi, (_, pbbox) in enumerate(prev_tracks)
                for ni, nbbox in enumerate(new_bboxes)
            ]
            scores.sort(reverse=True)
            for score, pi, ni in scores:
                if pi in matched or ni in used or score <= 0.05:
                    continue
                matched[pi] = ni
                used.add(ni)

            for pi, (_, pbbox) in enumerate(prev_tracks):
                if pi in matched:
                    continue
                best = min(
                    ((bbox_center_distance(pbbox, nbbox), ni) for ni, nbbox in enumerate(new_bboxes) if ni not in used),
                    default=(float('inf'), -1)
                )
                if best[0] < 300:
                    matched[pi] = best[1]
                    used.add(best[1])

            new_prev = []
            for pi, (oid, _) in enumerate(prev_tracks):
                if pi in matched:
                    nbbox = new_bboxes[matched[pi]]
                    masks_f[oid] = bbox_to_mask(nbbox, w, h)
                    bboxes_f[oid] = nbbox
                    texts_f[oid] = prompt
                    new_prev.append((oid, nbbox))

            for ni, nbbox in enumerate(new_bboxes):
                if ni not in used:
                    oid = state.next_obj_id
                    state.next_obj_id += 1
                    state.color_by_obj[oid] = state.color_by_prompt.get(prompt, pastel_color_for_prompt(prompt))
                    masks_f[oid] = bbox_to_mask(nbbox, w, h)
                    bboxes_f[oid] = nbbox
                    texts_f[oid] = prompt
                    state.prompts.setdefault(prompt, []).append(oid)
                    new_prev.append((oid, nbbox))

            prev_tracks = new_prev
            state.composited_frames.pop(f_idx, None)
            processed += 1
            if processed % 5 == 0 or f_idx == total - 1:
                yield state, f"Propagating '{prompt}' (forward): frame {f_idx}/{total}", gr.update(value=f_idx)

        # Backward propagation
        prev_tracks = [(oid, seed_bboxes_by_oid[oid]) for oid in seed_bboxes_by_oid]
        for f_idx in range(seed_frame_idx - 1, -1, -1):
            frame = state.video_frames[f_idx]
            w, h = frame.size
            new_bboxes = detect_objects_in_frame(frame, prompt)

            masks_f = state.masks_by_frame.setdefault(f_idx, {})
            bboxes_f = state.bboxes_by_frame.setdefault(f_idx, {})
            texts_f = state.text_prompts_by_frame_obj.setdefault(f_idx, {})

            used = set()
            matched = {}
            scores = [
                (bbox_iou(pbbox, nbbox), pi, ni)
                for pi, (_, pbbox) in enumerate(prev_tracks)
                for ni, nbbox in enumerate(new_bboxes)
            ]
            scores.sort(reverse=True)
            for score, pi, ni in scores:
                if pi in matched or ni in used or score <= 0.05:
                    continue
                matched[pi] = ni
                used.add(ni)

            for pi, (_, pbbox) in enumerate(prev_tracks):
                if pi in matched:
                    continue
                best = min(
                    ((bbox_center_distance(pbbox, nbbox), ni) for ni, nbbox in enumerate(new_bboxes) if ni not in used),
                    default=(float('inf'), -1)
                )
                if best[0] < 300:
                    matched[pi] = best[1]
                    used.add(best[1])

            new_prev = []
            for pi, (oid, _) in enumerate(prev_tracks):
                if pi in matched:
                    nbbox = new_bboxes[matched[pi]]
                    masks_f[oid] = bbox_to_mask(nbbox, w, h)
                    bboxes_f[oid] = nbbox
                    texts_f[oid] = prompt
                    new_prev.append((oid, nbbox))

            for ni, nbbox in enumerate(new_bboxes):
                if ni not in used:
                    oid = state.next_obj_id
                    state.next_obj_id += 1
                    state.color_by_obj[oid] = state.color_by_prompt.get(prompt, pastel_color_for_prompt(prompt))
                    masks_f[oid] = bbox_to_mask(nbbox, w, h)
                    bboxes_f[oid] = nbbox
                    texts_f[oid] = prompt
                    state.prompts.setdefault(prompt, []).append(oid)
                    new_prev.append((oid, nbbox))

            prev_tracks = new_prev
            state.composited_frames.pop(f_idx, None)
            processed += 1
            if processed % 5 == 0 or f_idx == 0:
                yield state, f"Propagating '{prompt}' (backward): frame {f_idx}/{total}", gr.update(value=f_idx)

    yield state, f"✅ Propagation complete across {total} frames for {len(state.prompts)} prompt(s).", gr.update(value=0)

@spaces.GPU
def apply_point_prompt_on_frame(
    pt_state: PointTrackingState,
    frame_idx: int,
    text_prompt: str,
) -> tuple[Image.Image, str, str, PointTrackingState]:
    if pt_state is None or not pt_state.video_frames:
        return None, "Upload a video first.", "**Active prompts:** None", pt_state
    if not text_prompt or not text_prompt.strip():
        ap = _get_active_prompts_points(pt_state)
        return update_point_display(pt_state, int(frame_idx)), "Please enter a text prompt.", ap, pt_state

    frame_idx = int(np.clip(frame_idx, 0, len(pt_state.video_frames) - 1))
    frame = pt_state.video_frames[frame_idx]

    pt_state.prompt_text = text_prompt.strip()

    pt_state.points_by_frame.clear()
    pt_state.trails.clear()
    pt_state.composited_frames.clear()

    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
    all_points = []
    status_parts = [f"Point detection on frame {frame_idx}:"]

    for prompt in prompts:
        points = detect_precise_points_in_frame(frame, prompt)
        for pt in points:
            all_points.append(pt)
            track_idx = len(pt_state.trails)
            pt_state.trails.append([(frame_idx, pt[0], pt[1])])
        status_parts.append(f"  • '{prompt}': {len(points)} point(s)")

    pt_state.points_by_frame[frame_idx] = all_points
    pt_state.composited_frames.pop(frame_idx, None)

    status = "\n".join(status_parts)
    ap = _get_active_prompts_points(pt_state)
    return update_point_display(pt_state, frame_idx), status, ap, pt_state

@spaces.GPU
def propagate_points(pt_state: PointTrackingState) -> Iterator[tuple[PointTrackingState, str, dict]]:
    if pt_state is None or not pt_state.video_frames:
        yield pt_state, "Load a video first.", gr.update()
        return
    if not pt_state.trails:
        yield pt_state, "No points defined. Apply point prompt on a frame first.", gr.update()
        return
    if not pt_state.prompt_text:
        yield pt_state, "No prompt text. Apply a text prompt first.", gr.update()
        return

    total = pt_state.num_frames
    prompts = [p.strip() for p in pt_state.prompt_text.split(",") if p.strip()]

    seed_frame_idx = None
    for f_idx in sorted(pt_state.points_by_frame.keys()):
        if pt_state.points_by_frame[f_idx]:
            seed_frame_idx = f_idx
            break
    if seed_frame_idx is None:
        yield pt_state, "No seed points found.", gr.update()
        return

    yield pt_state, f"Propagating points: 0/{total}", gr.update()

    seed_tracks = []
    for trail_idx, trail in enumerate(pt_state.trails):
        for fi, tx, ty in trail:
            if fi == seed_frame_idx:
                seed_tracks.append((trail_idx, (tx, ty)))
                break

    prev_tracks = list(seed_tracks)
    lost_count = {t[0]: 0 for t in prev_tracks}

    for f_idx in range(seed_frame_idx + 1, total):
        frame = pt_state.video_frames[f_idx]
        w, h = frame.size

        all_new_points = []
        for prompt in prompts:
            pts = detect_precise_points_in_frame(frame, prompt)
            all_new_points.extend(pts)

        points_f = []
        diag = (w ** 2 + h ** 2) ** 0.5
        match_threshold = diag * 0.25

        if not all_new_points:
            new_prev = []
            for track_idx, prev_pt in prev_tracks:
                lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                if lost_count[track_idx] > 5:
                    continue
                points_f.append(prev_pt)
                pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                new_prev.append((track_idx, prev_pt))
            prev_tracks = new_prev
        else:
            used_new = set()
            matched = {}
            dist_pairs = []
            for pi, (_, prev_pt) in enumerate(prev_tracks):
                for ni, new_pt in enumerate(all_new_points):
                    d = pixel_point_distance(prev_pt, new_pt)
                    dist_pairs.append((d, pi, ni))
            dist_pairs.sort()
            for d, pi, ni in dist_pairs:
                if pi in matched or ni in used_new:
                    continue
                if d < match_threshold:
                    matched[pi] = ni
                    used_new.add(ni)

            new_prev = []
            for pi, (track_idx, prev_pt) in enumerate(prev_tracks):
                if pi in matched:
                    new_pt = all_new_points[matched[pi]]
                    points_f.append(new_pt)
                    pt_state.trails[track_idx].append((f_idx, new_pt[0], new_pt[1]))
                    new_prev.append((track_idx, new_pt))
                    lost_count[track_idx] = 0
                else:
                    lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                    if lost_count[track_idx] <= 5:
                        points_f.append(prev_pt)
                        pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                        new_prev.append((track_idx, prev_pt))

            for ni, new_pt in enumerate(all_new_points):
                if ni not in used_new:
                    too_close = any(pixel_point_distance(new_pt, pp) < diag * 0.08 for _, pp in new_prev)
                    if not too_close:
                        track_idx = len(pt_state.trails)
                        pt_state.trails.append([(f_idx, new_pt[0], new_pt[1])])
                        points_f.append(new_pt)
                        new_prev.append((track_idx, new_pt))
                        lost_count[track_idx] = 0

            prev_tracks = new_prev

        pt_state.points_by_frame[f_idx] = points_f
        pt_state.composited_frames.pop(f_idx, None)

        if (f_idx - seed_frame_idx) % 5 == 0 or f_idx == total - 1:
            yield pt_state, f"Propagating points (forward): frame {f_idx}/{total}", gr.update(value=f_idx)

    prev_tracks = list(seed_tracks)
    lost_count = {t[0]: 0 for t in prev_tracks}

    for f_idx in range(seed_frame_idx - 1, -1, -1):
        frame = pt_state.video_frames[f_idx]
        w, h = frame.size

        all_new_points = []
        for prompt in prompts:
            pts = detect_precise_points_in_frame(frame, prompt)
            all_new_points.extend(pts)

        points_f = []
        diag = (w ** 2 + h ** 2) ** 0.5
        match_threshold = diag * 0.25

        if not all_new_points:
            new_prev = []
            for track_idx, prev_pt in prev_tracks:
                lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                if lost_count[track_idx] > 5:
                    continue
                points_f.append(prev_pt)
                pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                new_prev.append((track_idx, prev_pt))
            prev_tracks = new_prev
        else:
            used_new = set()
            matched = {}
            dist_pairs = []
            for pi, (_, prev_pt) in enumerate(prev_tracks):
                for ni, new_pt in enumerate(all_new_points):
                    d = pixel_point_distance(prev_pt, new_pt)
                    dist_pairs.append((d, pi, ni))
            dist_pairs.sort()
            for d, pi, ni in dist_pairs:
                if pi in matched or ni in used_new:
                    continue
                if d < match_threshold:
                    matched[pi] = ni
                    used_new.add(ni)

            new_prev = []
            for pi, (track_idx, prev_pt) in enumerate(prev_tracks):
                if pi in matched:
                    new_pt = all_new_points[matched[pi]]
                    points_f.append(new_pt)
                    pt_state.trails[track_idx].append((f_idx, new_pt[0], new_pt[1]))
                    new_prev.append((track_idx, new_pt))
                    lost_count[track_idx] = 0
                else:
                    lost_count[track_idx] = lost_count.get(track_idx, 0) + 1
                    if lost_count[track_idx] <= 5:
                        points_f.append(prev_pt)
                        pt_state.trails[track_idx].append((f_idx, prev_pt[0], prev_pt[1]))
                        new_prev.append((track_idx, prev_pt))

            for ni, new_pt in enumerate(all_new_points):
                if ni not in used_new:
                    too_close = any(pixel_point_distance(new_pt, pp) < diag * 0.08 for _, pp in new_prev)
                    if not too_close:
                        track_idx = len(pt_state.trails)
                        pt_state.trails.append([(f_idx, new_pt[0], new_pt[1])])
                        points_f.append(new_pt)
                        new_prev.append((track_idx, new_pt))
                        lost_count[track_idx] = 0

            prev_tracks = new_prev

        pt_state.points_by_frame[f_idx] = points_f
        pt_state.composited_frames.pop(f_idx, None)

        if (seed_frame_idx - f_idx) % 5 == 0 or f_idx == 0:
            yield pt_state, f"Propagating points (backward): frame {f_idx}/{total}", gr.update(value=f_idx)

    yield pt_state, f"✅ Point propagation complete across {total} frames. {len(pt_state.trails)} tracks.", gr.update(value=seed_frame_idx)

def reset_tracking_prompts(state: TrackingState) -> tuple[TrackingState, Image.Image, str, str]:
    if state is None:
        return state, None, "No active session.", "**Active prompts:** None"
    state.masks_by_frame.clear()
    state.bboxes_by_frame.clear()
    state.text_prompts_by_frame_obj.clear()
    state.composited_frames.clear()
    state.color_by_obj.clear()
    state.color_by_prompt.clear()
    state.prompts.clear()
    state.next_obj_id = 1
    current_idx = max(0, min(getattr(state, 'current_frame_idx', 0), state.num_frames - 1))
    preview = update_tracking_display(state, current_idx)
    return state, preview, "Prompts and outputs reset. Video preserved.", "**Active prompts:** None"


def reset_tracking_session(state: TrackingState) -> tuple[TrackingState, Image.Image, dict, dict, str, str]:
    if not state.video_frames:
        return state, None, gr.update(minimum=0, maximum=0), gr.update(value=0), "Session reset.", "**Active prompts:** None"
    state.masks_by_frame.clear()
    state.bboxes_by_frame.clear()
    state.text_prompts_by_frame_obj.clear()
    state.composited_frames.clear()
    state.color_by_obj.clear()
    state.color_by_prompt.clear()
    state.prompts.clear()
    state.next_obj_id = 1
    gc.collect()
    current_idx = max(0, min(getattr(state, 'current_frame_idx', 0), state.num_frames - 1))
    preview = update_tracking_display(state, current_idx)
    return (
        state, preview,
        gr.update(minimum=0, maximum=max(state.num_frames - 1, 0), interactive=True),
        gr.update(value=current_idx),
        "Session reset. Prompts cleared; video preserved.",
        "**Active prompts:** None"
    )


def reset_point_prompts(pt_state: PointTrackingState) -> tuple[PointTrackingState, Image.Image, str, str]:
    if pt_state is None:
        return pt_state, None, "No active session.", "**Active prompts:** None"
    pt_state.points_by_frame.clear()
    pt_state.trails.clear()
    pt_state.composited_frames.clear()
    pt_state.prompt_text = ""
    current_idx = max(0, min(getattr(pt_state, 'current_frame_idx', 0), pt_state.num_frames - 1))
    preview = update_point_display(pt_state, current_idx)
    return pt_state, preview, "Point prompts reset. Video preserved.", "**Active prompts:** None"


def reset_point_session(pt_state: PointTrackingState) -> tuple[PointTrackingState, Image.Image, dict, dict, str, str]:
    if not pt_state.video_frames:
        return pt_state, None, gr.update(minimum=0, maximum=0), gr.update(value=0), "Session reset.", "**Active prompts:** None"
    pt_state.points_by_frame.clear()
    pt_state.trails.clear()
    pt_state.composited_frames.clear()
    pt_state.prompt_text = ""
    gc.collect()
    current_idx = max(0, min(getattr(pt_state, 'current_frame_idx', 0), pt_state.num_frames - 1))
    preview = update_point_display(pt_state, current_idx)
    return (
        pt_state, preview,
        gr.update(minimum=0, maximum=max(pt_state.num_frames - 1, 0), interactive=True),
        gr.update(value=current_idx),
        "Session reset. Video preserved.",
        "**Active prompts:** None"
    )

def render_tracking_video(state: TrackingState) -> str:
    if state is None or state.num_frames == 0:
        raise gr.Error("Load a video first.")
    fps = state.video_fps if state.video_fps and state.video_fps > 0 else 12
    frames_bgr = []
    w, h = state.video_frames[0].size
    for idx in range(state.num_frames):
        img = state.composited_frames.get(idx)
        if img is None:
            img = compose_tracking_frame(state, idx)
        frames_bgr.append(np.array(img)[:, :, ::-1])
        if (idx + 1) % 60 == 0:
            gc.collect()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames_bgr:
            writer.write(fr)
        writer.release()
        return tmp.name


def render_point_video(pt_state: PointTrackingState) -> str:
    if pt_state is None or pt_state.num_frames == 0:
        raise gr.Error("Load a video first.")
    fps = pt_state.video_fps if pt_state.video_fps and pt_state.video_fps > 0 else 12
    frames_bgr = []
    w, h = pt_state.video_frames[0].size
    for idx in range(pt_state.num_frames):
        img = pt_state.composited_frames.get(idx)
        if img is None:
            img = compose_point_frame(pt_state, idx)
        frames_bgr.append(np.array(img)[:, :, ::-1])
        if (idx + 1) % 60 == 0:
            gc.collect()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames_bgr:
            writer.write(fr)
        writer.release()
        return tmp.name

def _on_video_change_tracking(state: TrackingState, video) -> tuple[TrackingState, dict, Image.Image, str, str]:
    if video is None:
        return state, gr.update(), None, "", "**Active prompts:** None"
    state, min_idx, max_idx, first_frame, status = init_tracking_video(state, video)
    ap = _get_active_prompts_tracking(state)
    return (
        state,
        gr.update(minimum=min_idx, maximum=max_idx, value=min_idx, interactive=True),
        first_frame,
        status,
        ap,
    )


def _on_video_change_points(pt_state: PointTrackingState, video) -> tuple[PointTrackingState, dict, Image.Image, str, str]:
    if video is None:
        return pt_state, gr.update(), None, "", "**Active prompts:** None"
    pt_state, min_idx, max_idx, first_frame, status = init_point_video(pt_state, video)
    ap = _get_active_prompts_points(pt_state)
    return (
        pt_state,
        gr.update(minimum=min_idx, maximum=max_idx, value=min_idx, interactive=True),
        first_frame,
        status,
        ap,
    )

@spaces.GPU
def process_image_detection(image: Image.Image, prompt: str) -> tuple[Image.Image, str]:
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a detection prompt.")

    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    original_width, original_height = image.size

    full_prompt = f"Provide bounding box coordinates for {prompt}. Report in JSON format."
    output_text = run_model_inference(image, full_prompt)

    parsed_json = safe_parse_json(output_text)
    objects_result = {"objects": []}

    if isinstance(parsed_json, list):
        for item in parsed_json:
            if "bbox_2d" in item and len(item["bbox_2d"]) == 4:
                xmin, ymin, xmax, ymax = item["bbox_2d"]
                label = item.get("label", "object")
                objects_result["objects"].append({
                    "x_min": xmin / 1000.0,
                    "y_min": ymin / 1000.0,
                    "x_max": xmax / 1000.0,
                    "y_max": ymax / 1000.0,
                    "label": label,
                })
    elif isinstance(parsed_json, dict):
        if "bbox_2d" in parsed_json and len(parsed_json["bbox_2d"]) == 4:
            xmin, ymin, xmax, ymax = parsed_json["bbox_2d"]
            label = parsed_json.get("label", "object")
            objects_result["objects"].append({
                "x_min": xmin / 1000.0,
                "y_min": ymin / 1000.0,
                "x_max": xmax / 1000.0,
                "y_max": ymax / 1000.0,
                "label": label,
            })

    if not objects_result["objects"]:
        bboxes = parse_bboxes_from_text(output_text)
        for idx, bbox in enumerate(bboxes):
            objects_result["objects"].append({
                "x_min": bbox[0] / 1000.0,
                "y_min": bbox[1] / 1000.0,
                "x_max": bbox[2] / 1000.0,
                "y_max": bbox[3] / 1000.0,
                "label": prompt.strip(),
            })

    annotated = annotate_image_detection(image.copy(), objects_result)
    result_text = json.dumps(objects_result, indent=2) if objects_result["objects"] else f"No objects detected for '{prompt}'.\n\nRaw output:\n{output_text}"

    return annotated, result_text

@spaces.GPU
def process_image_pointer(image: Image.Image, prompt: str) -> tuple[Image.Image, str]:
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please provide a pointing prompt.")

    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    original_width, original_height = image.size

    full_prompt = f"Provide 2d point coordinates for {prompt}. Report in JSON format."
    output_text = run_model_inference(image, full_prompt)

    parsed_json = safe_parse_json(output_text)
    points_result = {"points": []}

    if isinstance(parsed_json, list):
        for item in parsed_json:
            if "point_2d" in item and len(item["point_2d"]) == 2:
                x, y = item["point_2d"]
                points_result["points"].append({"x": x / 1000.0, "y": y / 1000.0})
    elif isinstance(parsed_json, dict):
        if "point_2d" in parsed_json and len(parsed_json["point_2d"]) == 2:
            x, y = parsed_json["point_2d"]
            points_result["points"].append({"x": x / 1000.0, "y": y / 1000.0})

    if not points_result["points"]:
        detected_points = parse_precise_points(output_text, original_width, original_height)
        for px, py in detected_points:
            points_result["points"].append({
                "x": px / original_width,
                "y": py / original_height,
            })

    if not points_result["points"]:
        bboxes = parse_bboxes_from_text(output_text)
        for bbox in bboxes:
            cx = (bbox[0] + bbox[2]) / 2 / 1000.0
            cy = (bbox[1] + bbox[3]) / 2 / 1000.0
            points_result["points"].append({"x": cx, "y": cy})

    annotated = annotate_image_points(image.copy(), points_result)
    result_text = json.dumps(points_result, indent=2) if points_result["points"] else f"No points detected for '{prompt}'.\n\nRaw output:\n{output_text}"

    return annotated, result_text

css = """
#col-container {
    margin: 0 auto;
    max-width: 900px;
}
#main-title h1 { font-size: 2.6em !important; }
"""


with gr.Blocks() as demo:
    gr.Markdown("# **Qwen3-VL-Video-Grounding**", elem_id="main-title")
    gr.Markdown(
        """
        Perform text-guided object tracking, point tracking, image detection, and image pointing with the Qwen3-VL multimodal model.
        **Video tabs:** Upload a video → Select a frame → Apply text prompt(s) → Preview → Propagate → Render MP4.
        **Image tabs:** Upload an image → Enter prompt → Get instant detection or pointing results.
        Due to compute constraints, this app only supports stop-frame object detection or tracking on propagated frames. For dense-frame full video processing, please visit the [GitHub](https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-Video-Grounding) page.
        """
    )

    tracking_state = gr.State(TrackingState())
    point_state = gr.State(PointTrackingState())

    with gr.Tabs() as main_tabs:

        with gr.Tab("Video Object Tracking"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Quick start**
                        - **Load a video**: Upload your own or pick an example below.
                        - Select a frame and enter text description(s) to detect objects (e.g., "red car", "person"). Multiple prompts separated by commas.
                        - The text prompt detects all instances on the **selected frame only**.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Working with results**
                        - **Preview**: Use the slider to navigate frames and see the current masks/bboxes.
                        - **Propagate**: Click "Propagate across video" to track all defined objects through every frame.
                        - **Export**: Render an MP4 for smooth playback using the original video FPS.
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    video_in_tracking = gr.Video(label="Upload video", sources=["upload", "webcam"])
                    load_status_tracking = gr.Markdown(visible=True)
                    reset_btn_tracking = gr.Button("Reset Session", variant="secondary")
                with gr.Column(scale=2):
                    preview_tracking = gr.Image(label="Preview")
                    with gr.Row():
                        frame_slider_tracking = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
                        with gr.Column(scale=0):
                            propagate_btn_tracking = gr.Button("Propagate across video", variant="primary")
                            propagate_status_tracking = gr.Markdown(visible=True)
                    with gr.Row():
                        text_prompt_tracking = gr.Textbox(
                            label="Text Prompt(s)",
                            placeholder="Enter text description(s) (e.g., 'person' or 'person, car, dog' for multiple)",
                            lines=2,
                        )
                        with gr.Column(scale=0):
                            apply_btn_tracking = gr.Button("Apply Text Prompt(s)", variant="primary")
                            reset_prompts_btn_tracking = gr.Button("Reset Prompts", variant="secondary")
                    active_prompts_tracking = gr.Markdown("**Active prompts:** None", visible=True)
                    text_status_tracking = gr.Markdown(visible=True)

            with gr.Row():
                render_btn_tracking = gr.Button("Render MP4 for smooth playback", variant="primary")
            playback_video_tracking = gr.Video(label="Rendered Playback", interactive=False)

            gr.Examples(
                examples=[
                    ["examples/1.mp4"],
                    ["examples/2.mp4"],
                    ["examples/3.mp4"],
                ],
                inputs=[video_in_tracking],
                label="Examples"
            )

        with gr.Tab("Video Points Tracker"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Quick start**
                        - **Load a video**: Upload your own or pick an example below.
                        - Select a frame and enter text description(s) (e.g., `person, ball`).
                        - The model locates the **center point** of each detected object on the **selected frame only**.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Working with results**
                        - **Preview**: Use the slider to see detected points and motion trails.
                        - **Propagate**: Click "Propagate across video" to track points through every frame.
                        - **Export**: Render an MP4 with red dot tracking and trails.
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    video_in_points = gr.Video(label="Upload video", sources=["upload", "webcam"])
                    load_status_points = gr.Markdown(visible=True)
                    reset_btn_points = gr.Button("Reset Session", variant="secondary")
                with gr.Column(scale=2):
                    preview_points = gr.Image(label="Preview")
                    with gr.Row():
                        frame_slider_points = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
                        with gr.Column(scale=0):
                            propagate_btn_points = gr.Button("Propagate across video", variant="primary")
                            propagate_status_points = gr.Markdown(visible=True)
                    with gr.Row():
                        text_prompt_points = gr.Textbox(
                            label="Text Prompt(s)",
                            placeholder="Enter text description(s) (e.g., 'person' or 'person, ball' for multiple)",
                            lines=2,
                        )
                        with gr.Column(scale=0):
                            apply_btn_points = gr.Button("Apply Point Prompt(s)", variant="primary")
                            reset_prompts_btn_points = gr.Button("Reset Prompts", variant="secondary")
                    active_prompts_points = gr.Markdown("**Active prompts:** None", visible=True)
                    text_status_points = gr.Markdown(visible=True)

            with gr.Row():
                render_btn_points = gr.Button("Render MP4 for smooth playback", variant="primary")
            playback_video_points = gr.Video(label="Rendered Playback", interactive=False)

            gr.Examples(
                examples=[
                    ["examples/1.mp4"],
                    ["examples/2.mp4"],
                    ["examples/3.mp4"],
                ],
                inputs=[video_in_points],
                label="Examples"
            )

        with gr.Tab("Image Detection"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Image Object Detection**
                        - Upload an image and enter what you want to detect.
                        - The model returns bounding boxes around all matching objects.
                        - Results are displayed as colored boxes with labels.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Tips**
                        - Be specific: "red car" works better than just "car" if you want a specific one.
                        - You can detect multiple types: try "person", "headlight", "window", etc.
                        - The JSON output shows normalized coordinates (0-1 range).
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    img_det_input = gr.Image(type="pil", label="Upload Image", height=400)
                    img_det_prompt = gr.Textbox(
                        label="Detection Prompt",
                        placeholder="e.g., headlight, person, red car, laptop",
                        lines=2,
                    )
                    img_det_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column(scale=1):
                    img_det_output = gr.Image(label="Detection Result", height=400)
                    img_det_text = gr.Textbox(label="Detection Output (JSON)", lines=10, interactive=False)

            gr.Examples(
                examples=[
                    ["examples-images/5.jpg", "children"],
                    ["examples-images/4.jpg", "headlight"],
                    ["examples-images/3.jpg", "gun"],
                    ["examples-images/1.jpg", "boat"],
                ],
                inputs=[img_det_input, img_det_prompt],
                label="Examples"
            )

        with gr.Tab("Image Pointer"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Image Point Detection**
                        - Upload an image and describe what you want to point to.
                        - The model returns precise center-point coordinates for each matching object.
                        - Results are displayed as red dots on the image.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Tips**
                        - Great for locating specific parts: "the gun held by the person", "nose of the dog".
                        - Multiple instances are supported: all matching objects get a point.
                        - The JSON output shows normalized coordinates (0-1 range).
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    img_pt_input = gr.Image(type="pil", label="Upload Image", height=400)
                    img_pt_prompt = gr.Textbox(
                        label="Pointing Prompt",
                        placeholder="e.g., the gun held by the person, nose of the dog",
                        lines=2,
                    )
                    img_pt_btn = gr.Button("Point to Objects", variant="primary")
                with gr.Column(scale=1):
                    img_pt_output = gr.Image(label="Pointing Result", height=400)
                    img_pt_text = gr.Textbox(label="Points Output (JSON)", lines=10, interactive=False)

            gr.Examples(
                examples=[
                    ["examples-images/5.jpg", "children who are out of focus and wearing a white T-shirt"],
                    ["examples-images/3.jpg", "gun"],
                    ["examples-images/4.jpg", "headlight"],
                    ["examples-images/1.jpg", "boat"],
                ],
                inputs=[img_pt_input, img_pt_prompt],
                label="Examples"
            )

    video_in_tracking.change(
        fn=_on_video_change_tracking,
        inputs=[tracking_state, video_in_tracking],
        outputs=[tracking_state, frame_slider_tracking, preview_tracking, load_status_tracking, active_prompts_tracking],
        show_progress=True,
    )

    def _sync_tracking_frame(state_in: TrackingState, idx: int) -> Image.Image:
        if state_in is not None:
            state_in.current_frame_idx = int(idx)
        return update_tracking_display(state_in, int(idx))

    frame_slider_tracking.change(
        fn=_sync_tracking_frame,
        inputs=[tracking_state, frame_slider_tracking],
        outputs=preview_tracking,
    )

    apply_btn_tracking.click(
        fn=apply_tracking_prompt_on_frame,
        inputs=[tracking_state, frame_slider_tracking, text_prompt_tracking],
        outputs=[preview_tracking, text_status_tracking, active_prompts_tracking, tracking_state],
    )

    propagate_btn_tracking.click(
        fn=propagate_tracking,
        inputs=tracking_state,
        outputs=[tracking_state, propagate_status_tracking, frame_slider_tracking],
    )

    reset_prompts_btn_tracking.click(
        fn=reset_tracking_prompts,
        inputs=tracking_state,
        outputs=[tracking_state, preview_tracking, text_status_tracking, active_prompts_tracking],
    )

    reset_btn_tracking.click(
        fn=reset_tracking_session,
        inputs=tracking_state,
        outputs=[tracking_state, preview_tracking, frame_slider_tracking, frame_slider_tracking, load_status_tracking, active_prompts_tracking],
    )

    render_btn_tracking.click(
        fn=render_tracking_video,
        inputs=tracking_state,
        outputs=playback_video_tracking,
    )

    video_in_points.change(
        fn=_on_video_change_points,
        inputs=[point_state, video_in_points],
        outputs=[point_state, frame_slider_points, preview_points, load_status_points, active_prompts_points],
        show_progress=True,
    )

    def _sync_point_frame(state_in: PointTrackingState, idx: int) -> Image.Image:
        if state_in is not None:
            state_in.current_frame_idx = int(idx)
        return update_point_display(state_in, int(idx))

    frame_slider_points.change(
        fn=_sync_point_frame,
        inputs=[point_state, frame_slider_points],
        outputs=preview_points,
    )

    apply_btn_points.click(
        fn=apply_point_prompt_on_frame,
        inputs=[point_state, frame_slider_points, text_prompt_points],
        outputs=[preview_points, text_status_points, active_prompts_points, point_state],
    )

    propagate_btn_points.click(
        fn=propagate_points,
        inputs=point_state,
        outputs=[point_state, propagate_status_points, frame_slider_points],
    )

    reset_prompts_btn_points.click(
        fn=reset_point_prompts,
        inputs=point_state,
        outputs=[point_state, preview_points, text_status_points, active_prompts_points],
    )

    reset_btn_points.click(
        fn=reset_point_session,
        inputs=point_state,
        outputs=[point_state, preview_points, frame_slider_points, frame_slider_points, load_status_points, active_prompts_points],
    )

    render_btn_points.click(
        fn=render_point_video,
        inputs=point_state,
        outputs=playback_video_points,
    )

    img_det_btn.click(
        fn=process_image_detection,
        inputs=[img_det_input, img_det_prompt],
        outputs=[img_det_output, img_det_text],
        show_progress=True,
    )
    
    img_pt_btn.click(
        fn=process_image_pointer,
        inputs=[img_pt_input, img_pt_prompt],
        outputs=[img_pt_output, img_pt_text],
        show_progress=True,
    )


demo.queue(api_open=False).launch(theme=Soft(primary_hue="orange", secondary_hue="rose"), css=css, ssr_mode=False)
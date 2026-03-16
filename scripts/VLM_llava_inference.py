import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===== Paths =====
input_imgs = "/Users/srimanmohapatra/Downloads/occupational_safety_/output/video_frames"
out_file   = "/Users/srimanmohapatra/Downloads/occupational_safety_/output/vlm_video_results.txt"
os.makedirs(os.path.dirname(out_file), exist_ok=True)

# ===== Single concise question (for speed) =====
Q_LIST = [
    "Describe the situation in this frame and explain any unsafe condition or risk you can see, with one safety recommendation."
]

# ===== Vision-language model (LLaVA) =====
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"  # same model id

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)


def vlm_qa(image_path, queries, model, processor):
    """Run LLaVA on one image for a list of questions."""
    img = Image.open(image_path).convert("RGB")
    results = []

    for q in queries:
        # LLaVA requires the <image> token in the prompt
        prompt = f"USER: <image>\n{q}\nASSISTANT:"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,   # smaller for faster CPU inference
                do_sample=False,
            )

        answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        results.append((q, answer.strip()))

    return results


def main():
    img_files = sorted(
        f for f in os.listdir(input_imgs)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not img_files:
        print(f"No images found in {input_imgs}")
        return

    # ===== LIMIT TO 1 FRAME FOR MAC DEMO =====
    MAX_FRAMES = 1
    img_files = img_files[:MAX_FRAMES]

    with open(out_file, "w", encoding="utf-8") as f_out:
        for idx, fname in enumerate(img_files):
            img_path = os.path.join(input_imgs, fname)
            print(f"[{idx+1}/{len(img_files)}] Processing {img_path}")

            qa_pairs = vlm_qa(img_path, Q_LIST, model, processor)

            f_out.write(f"===== IMAGE: {fname} =====\n")
            for q, a in qa_pairs:
                f_out.write(f"Q: {q}\n")
                f_out.write(f"A: {a}\n\n")
            f_out.write("\n")

    print(f"Done. Results saved to: {out_file}")


if __name__ == "__main__":
    main()

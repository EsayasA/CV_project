import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from main import ProjectPipeline
from utils import overlay_mask, get_bbox_from_mask

pipeline = ProjectPipeline()


# =========================
# IMAGE PROCESSING
# =========================
def process_image(img, text, threshold):
    if img is None or not text:
        return None, "Please upload image and enter query"

    image, masks, scores, labels = pipeline.run_image(
        img,
        text,
        threshold,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    if len(masks) > 0:
        colors = [cm.hsv(i / len(masks)) for i in range(len(masks))]

        for mask, score, label, color in zip(masks, scores, labels, colors):
            rgba_color = (color[0], color[1], color[2], 0.5)

            overlay_mask(mask, ax, rgba_color)

            bbox = get_bbox_from_mask(mask)

            if bbox:
                x, y, w, h = bbox

                rect = plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor=color,
                    fill=False,
                    linewidth=2,
                )

                ax.add_patch(rect)

                label_text = f"{label} {score:.2f}"

                ax.text(
                    x,
                    y - 5,
                    label_text,
                    color="white",
                    fontsize=12,
                    weight="bold",
                    bbox=dict(
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.8,
                        pad=3,
                    ),
                )

        ax.set_title(f'Query: "{text}"')
        status = f"Found {len(masks)} object(s)"

    else:
        ax.set_title("No object detected")
        status = "No object found"

    ax.axis("off")
    plt.tight_layout()

    return fig, status


# =========================
# VIDEO PROCESSING
# =========================
def process_video(video, text, threshold):
    if not video or not text:
        return None

    return pipeline.run_video(video, text, threshold)


# =========================
# UI
# =========================
with gr.Blocks(title="Open-Vocab Locator") as demo:
    gr.Markdown("# 🔍 Open-Vocabulary Object Locator")

    with gr.Tab("Image Search"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil")

                img_query = gr.Textbox(
                    label="Query",
                    placeholder="a dog"
                )

                img_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="Confidence Threshold"
                )

                img_btn = gr.Button("Locate")

            with gr.Column():
                img_output = gr.Plot()
                img_status = gr.Textbox()

        img_btn.click(
            fn=process_image,
            inputs=[img_input, img_query, img_threshold],
            outputs=[img_output, img_status],
        )

    with gr.Tab("Video Search"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video()

                vid_query = gr.Textbox(
                    label="Query",
                    placeholder="a person"
                )

                vid_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="Confidence Threshold"
                )

                vid_btn = gr.Button("Track")

            with gr.Column():
                vid_output = gr.Video()

        vid_btn.click(
            fn=process_video,
            inputs=[vid_input, vid_query, vid_threshold],
            outputs=vid_output,
        )


if __name__ == "__main__":
    demo.launch()
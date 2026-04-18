import gradio as gr
import matplotlib.pyplot as plt
from main import ProjectPipeline
from utils import overlay_mask, get_bbox_from_mask

# Initialize the pipeline
pipeline = ProjectPipeline()

def interface_fn(image, text):
    if image is None or not text:
        return None, "Error: Upload an image and enter a query."

    # Call the updated run function
    img, mask, score = pipeline.run(image, text)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    if mask is not None:
        overlay_mask(mask, ax)
        bbox = get_bbox_from_mask(mask)
        if bbox:
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                 edgecolor="lime", fill=False, linewidth=3)
            ax.add_patch(rect)
        ax.set_title(f"Result for: {text} (Score: {score:.2f})")
    else:
        ax.set_title("No object detected.")

    ax.axis("off")
    plt.tight_layout()
    return fig, f"Confidence Score: {score:.2f}"

# UI setup
with gr.Blocks(title="Open-Vocabulary Search") as demo:
    gr.Markdown("# 🔍 Open-Vocabulary Object Locator (CV 2026)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Source Image")
            query_txt = gr.Textbox(label="What are you looking for?")
            btn = gr.Button("Locate Object")
        with gr.Column():
            output_plot = gr.Plot()
            output_status = gr.Textbox(label="Status")

    btn.click(interface_fn, inputs=[input_img, query_txt], outputs=[output_plot, output_status])

if __name__ == "__main__":
    demo.launch()

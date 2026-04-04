import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"

import gradio as gr
from transformers import pipeline

print("Loading model... please wait ⏳")
summarizer = pipeline(
    "summarization",
    model="t5-small",
    framework="pt",
    device=-1
)
print("Model loaded! ✅")


def summarize_text(text):
    if not text or len(text.strip()) < 50:
        return "⚠️ Please enter a longer text (at least 50 characters) to summarize."

    input_length = len(text.split())
    max_len = min(100, input_length // 2)
    min_len = min(20, max_len - 5)

    result = summarizer(
        "summarize: " + text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    return result[0]['summary_text']


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container { background: #0a0a0f !important; font-family: 'DM Sans', sans-serif !important; }
.gradio-container { max-width: 900px !important; margin: 0 auto !important; padding: 40px 24px !important; }
#badge { display: inline-block; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4); color: #a5b4fc; font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; padding: 6px 16px; border-radius: 100px; margin-bottom: 20px; }
#title { font-family: 'Syne', sans-serif !important; font-size: clamp(36px, 6vw, 58px) !important; font-weight: 800 !important; color: #ffffff !important; line-height: 1.1 !important; letter-spacing: -1.5px !important; margin-bottom: 16px !important; }
#title span { color: #818cf8; }
#subtitle { color: #6b7280; font-size: 16px; font-weight: 300; line-height: 1.6; max-width: 480px; margin: 0 auto; }
.card { background: #111118; border: 1px solid #1f1f2e; border-radius: 20px; padding: 28px; margin-bottom: 20px; }
.card-label { font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #4b5563; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
.card-label::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: #6366f1; display: inline-block; }
textarea { background: transparent !important; border: none !important; color: #e2e8f0 !important; font-family: 'DM Sans', sans-serif !important; font-size: 15px !important; line-height: 1.7 !important; resize: none !important; outline: none !important; box-shadow: none !important; padding: 0 !important; width: 100% !important; }
textarea::placeholder { color: #374151 !important; }
.gradio-textbox { background: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-textbox > label { display: none !important; }
.gradio-textbox > div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
#submit-btn { width: 100% !important; padding: 16px 32px !important; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important; border: none !important; border-radius: 14px !important; color: white !important; font-family: 'Syne', sans-serif !important; font-size: 15px !important; font-weight: 700 !important; cursor: pointer !important; box-shadow: 0 4px 24px rgba(99,102,241,0.3) !important; margin-bottom: 20px !important; }
#output-card textarea { color: #a7f3d0 !important; font-size: 16px !important; }
footer { display: none !important; }
"""

header_html = """
<div style="text-align:center; margin-bottom:48px">
    <div id="badge">⚡ Powered by PyTorch + Hugging Face</div>
    <div id="title">Summarize<br><span>Anything.</span></div>
    <div id="subtitle">Paste any article or long text — get a crisp, intelligent summary in seconds using T5.</div>
</div>
"""

with gr.Blocks(css=custom_css, title="AI Text Summarizer") as demo:
    gr.HTML(header_html)
    with gr.Column(elem_classes="card"):
        gr.HTML('<div class="card-label">Input Text</div>')
        input_box = gr.Textbox(lines=10, placeholder="Paste your article, blog post, research paper, or any long text here...", show_label=False)
    submit_btn = gr.Button("✦  Summarize Now", elem_id="submit-btn")
    with gr.Column(elem_classes="card", elem_id="output-card"):
        gr.HTML('<div class="card-label">Summary Output</div>')
        output_box = gr.Textbox(lines=6, placeholder="Your summary will appear here...", show_label=False, interactive=False)
    gr.Examples(
        examples=[
            ["Artificial intelligence (AI) is transforming the way humans interact with technology. From virtual assistants like Siri and Alexa to self-driving cars and medical diagnosis systems, AI is becoming an integral part of modern life. Machine learning enables computers to learn from data and improve over time without being explicitly programmed."],
            ["Climate change is one of the most pressing challenges facing humanity today. Rising global temperatures caused by burning of fossil fuels are leading to more frequent weather events such as hurricanes, droughts, and wildfires. Scientists warn that emissions must be drastically reduced to prevent catastrophic changes."],
        ],
        inputs=input_box,
        label="Try an example",
    )
    gr.HTML('<div style="text-align:center;margin-top:40px;color:#1f2937;font-size:12px">Built for <span style="color:#374151">Meta × PyTorch × Hugging Face × Scaler Hackathon</span></div>')
    submit_btn.click(fn=summarize_text, inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    demo.launch()
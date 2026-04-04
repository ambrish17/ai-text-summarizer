import gradio as gr
from transformers import pipeline

print("Loading model... please wait ⏳")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
print("Model loaded! ✅")

def summarize_text(text):
    if not text or len(text.strip()) < 50:
        return "⚠️ Please enter a longer text (at least 50 characters) to summarize."

    input_length = len(text.split())
    max_len = min(150, input_length // 2)
    min_len = min(30, max_len - 10)

    result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return result[0]['summary_text']


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 40px 24px !important;
}

/* ── Header ── */
#header {
    text-align: center;
    margin-bottom: 48px;
    position: relative;
}

#header::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

#badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}

#title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(36px, 6vw, 58px) !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
    letter-spacing: -1.5px !important;
    margin-bottom: 16px !important;
}

#title span { color: #818cf8; }

#subtitle {
    color: #6b7280;
    font-size: 16px;
    font-weight: 300;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}

/* ── Cards ── */
.card {
    background: #111118;
    border: 1px solid #1f1f2e;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}

.card:hover { border-color: #2d2d45; }

.card-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.card-label::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #6366f1;
    display: inline-block;
}

/* ── Textareas ── */
textarea {
    background: transparent !important;
    border: none !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    resize: none !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
}

textarea::placeholder { color: #374151 !important; }
textarea:focus { border: none !important; box-shadow: none !important; }

.gradio-textbox { background: transparent !important; border: none !important; box-shadow: none !important; }
.gradio-textbox > label { display: none !important; }
.gradio-textbox > div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }

/* ── Button ── */
#submit-btn {
    width: 100% !important;
    padding: 16px 32px !important;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.3) !important;
    margin-bottom: 20px !important;
}

#submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.45) !important;
}

#submit-btn:active { transform: translateY(0) !important; }

/* ── Output card accent ── */
#output-card { border-color: #1e1e35; }
#output-card .card-label::before { background: #34d399; }
#output-card .card-label { color: #4b5563; }

/* Output text color */
#output-card textarea { color: #a7f3d0 !important; font-size: 16px !important; }

/* ── Stats row ── */
#stats {
    display: flex;
    gap: 12px;
    margin-top: 20px;
}

.stat-box {
    flex: 1;
    background: #111118;
    border: 1px solid #1f1f2e;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #818cf8;
}

.stat-label {
    font-size: 11px;
    color: #4b5563;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Examples ── */
.examples-header {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #374151;
    margin: 32px 0 14px;
    text-align: center;
}

.gradio-examples table { width: 100% !important; border-collapse: collapse !important; }
.gradio-examples td {
    background: #111118 !important;
    border: 1px solid #1f1f2e !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    color: #6b7280 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    max-width: 260px !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.gradio-examples td:hover { background: #16161f !important; border-color: #6366f1 !important; color: #d1d5db !important; }

/* ── Footer ── */
#footer {
    text-align: center;
    margin-top: 40px;
    color: #1f2937;
    font-size: 12px;
    letter-spacing: 0.5px;
}

#footer span { color: #374151; }

/* Hide default Gradio chrome */
footer { display: none !important; }
.svelte-1gfkn6j { display: none !important; }
#component-0 > div.prose { display: none !important; }
"""

header_html = """
<div id="header">
    <div id="badge">⚡ Powered by Team BTECH </div>
    <div id="title">Summarize<br><span>Anything.</span></div>
    <div id="subtitle">Paste any article or long text — get a crisp, intelligent summary in seconds using Facebook BART.</div>
</div>
"""

input_label_html = '<div class="card-label">Input Text</div>'
output_label_html = '<div class="card-label" style="--dot-color:#34d399">Summary Output</div>'

with gr.Blocks(css=custom_css, title="AI Text Summarizer") as demo:

    gr.HTML(header_html)

    with gr.Column(elem_classes="card"):
        gr.HTML(input_label_html)
        input_box = gr.Textbox(
            lines=10,
            placeholder="Paste your article, blog post, research paper, or any long text here...",
            show_label=False,
        )

    submit_btn = gr.Button("✦  Summarize Now", elem_id="submit-btn")

    with gr.Column(elem_classes="card", elem_id="output-card"):
        gr.HTML(output_label_html)
        output_box = gr.Textbox(
            lines=6,
            placeholder="Your summary will appear here...",
            show_label=False,
            interactive=False,
        )

    gr.HTML('<div class="examples-header">— Try an example —</div>')
    gr.Examples(
        examples=[
            ["Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills associated with the human mind, such as learning and problem-solving. Some machine learning models are trained to mimic human cognitive abilities. For example, some programs have learned to communicate in human languages using deep learning, while image recognition software uses neural networks to identify and describe the content of digital images."],
            ["The Python programming language was created by Guido van Rossum and was first released in 1991. Python is designed to be easy to read and simple to implement. It is open source, which means it is free to use. Python is a general-purpose programming language, meaning it can be used for a wide variety of applications. It has become one of the most popular languages in the world, especially for data science, machine learning, and web development. Python's design philosophy emphasizes code readability and simplicity, making it an excellent choice for beginners and experienced developers alike."],
        ],
        inputs=input_box,
        label="",
    )

    gr.HTML("""
    <div id="footer">
        Built for <span>Meta × PyTorch × Hugging Face × Scaler Hackathon</span>
    </div>
    """)

    submit_btn.click(fn=summarize_text, inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    demo.launch()
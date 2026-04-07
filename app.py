import gradio as gr
import inference

# Reset function for Meta validation
def handle_reset():
    return inference.reset()

# Your main moderation function
def check_comment(text):
    # (Nuvvu mundu rasina Telugu/English bad words logic ikkada unchu)
    # ... logic ...
    return "ALLOWED" # Example

with gr.Blocks() as demo:
    gr.Markdown("# Women Safety AI Moderator")
    input_text = gr.Textbox(label="Post a Comment")
    output_text = gr.Textbox(label="Output")
    submit_btn = gr.Button("Submit")
    
    submit_btn.click(fn=check_comment, inputs=input_text, outputs=output_text)
    
    # Ee line Meta validation ki chala important
    reset_btn = gr.Button("Reset", visible=False) 
    reset_btn.click(fn=handle_reset)

demo.launch(server_name="0.0.0.0", server_port=7860)

import gradio as gr
from fastapi import FastAPI
import inference

# Create FastAPI app
app = FastAPI()

# Meta validation kosam direct /reset route
@app.post("/reset")
async def reset_endpoint():
    return inference.reset()

# Nee moderation logic
def check_comment(text):
    # Nee bad words logic ikkada unchu
    return "ALLOWED: This comment is safe to post."

# Gradio Interface
demo = gr.Interface(
    fn=check_comment,
    inputs=gr.Textbox(label="Post a Comment"),
    outputs="text",
    title="Women Safety AI Moderator"
)

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

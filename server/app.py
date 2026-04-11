import gradio as gr
from fastapi import FastAPI
import server.inference as inference
import uvicorn

app = FastAPI()

@app.post("/reset")
async def reset_endpoint():
    return inference.reset()

def check_comment(text):
    return "The system is running and monitoring safety."

# Gradio Interface setup
demo = gr.Interface(
    fn=check_comment, 
    inputs="text", 
    outputs="text", 
    title="Women Safety AI"
)

# Ee line Validator ki chala important
app = gr.mount_gradio_app(app, demo, path="/")

# Validator aduguthunna main() function idhe
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

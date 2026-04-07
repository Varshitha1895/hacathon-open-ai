import gradio as gr

def check_comment(text):
    # Ikkada manam Telugu boothulu add chestunnam
    telugu_bad_words = ["lanja", "dengu", "pichi", "kodaka", "na kodaka", "thittu"]
    english_bad_words = ["abuse", "harass", "cheap", "badword", "stupid", "sexy" , "fuck you" , "shit" ,"Bitch "]
    
    combined_list = telugu_bad_words + english_bad_words
    
    # Check if any bad word is in the text
    if any(word in text.lower() for word in combined_list):
        return "❌ BLOCKED: Harmful or Vulgar content detected!"
    else:
        return "✅ ALLOWED: This comment is safe to post."

demo = gr.Interface(
    fn=check_comment, 
    inputs=gr.Textbox(label="Post a Comment", placeholder="Type Telugu/Manglish/English here..."), 
    outputs="text",
    title="🛡️ Women Safety AI Moderator",
    description="Now updated with Telugu word detection!"
)

if __name__ == "__main__":
   https://huggingface.co/spaces/Varshitha189/hacathon-open-ai

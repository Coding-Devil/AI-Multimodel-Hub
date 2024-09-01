import streamlit as st
import streamlit.components.v1 as components
 
# Define the Hugging Face spaces URLs
urls = {
    "DRA : Text 2 Img": "https://ehristoforu-dalle-3-xl-lora-v2.hf.space/",
    "RAG : PDF Assistant": "https://cvachet-pdf-chatbot.hf.space/",
    "CONVO 4 : AI Chat": "https://gokulnath2003-simplechatbot.hf.space/",
    "Parler : Voice Chat": "https://parler-tts-parler-tts-mini.hf.space/"
}

# Title of the app
st.title("ANTI-GPT")
st.header("AI Multi-Model Hub")
st.markdown("---")

# Instructions for users
st.markdown("Select a task below to start interacting with the respective model.")

# Define a single row layout with Streamlit columns
cols = st.columns(len(urls))  # Create columns for each option

# Display each task with a button
for i, (task, url) in enumerate(urls.items()):
    with cols[i]:
        if st.button(task, key=task):
            st.session_state.selected_task = task

# Check if a task has been selected
if "selected_task" in st.session_state:
    task = st.session_state.selected_task
    st.subheader(f"{task}")
    st.markdown("---")

    # Embed the Hugging Face space in an iframe with maximum dimensions
    components.html(
        f'''
            <iframe src="{urls[task]}" 
                    style="position:fixed; top:0; left:0; width:100%; height:100%; border:none; margin:0; padding:0; overflow:hidden; z-index:999999;">
            </iframe>
            ''',
        height=800,  # This height is for the Streamlit container; the iframe will take the full window height
        scrolling=True
    )

    # Footer
    st.markdown("---")
    st.write("Created by Gokulnath and Omkar, An Open-Source contribution and integration :)")

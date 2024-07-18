
import numpy as np
import streamlit as st
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv, dotenv_values
load_dotenv()





# initialize the client
client = OpenAI(
  base_url="https://api-inference.huggingface.co/v1",
  api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')#"hf_xxx" # Replace with your token
) 




#Create supported models
model_links ={
    "Meta-Llama-3-8B":"meta-llama/Meta-Llama-3-8B-Instruct", 
    "Mistral-7B":"mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma-7B":"google/gemma-1.1-7b-it",
    "Gemma-2B":"google/gemma-1.1-2b-it",
    "Zephyr-7B-β":"HuggingFaceH4/zephyr-7b-beta",

}

#Pull info about the model to display
model_info ={
    "Mistral-7B":
        {'description':"""The Mistral model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nIt was created by the [**Mistral AI**](https://mistral.ai/news/announcing-mistral-7b/) team as has over  **7 billion parameters.** \n""",
        'logo':'https://mistral.ai/images/logo_hubc88c4ece131b91c7cb753f40e9e1cc5_2589_256x0_resize_q97_h2_lanczos_3.webp'},
    "Gemma-7B":        
        {'description':"""The Gemma model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nIt was created by the [**Google's AI Team**](https://blog.google/technology/developers/gemma-open-models/) team as has over  **7 billion parameters.** \n""",
        'logo':'https://pbs.twimg.com/media/GG3sJg7X0AEaNIq.jpg'},
    "Gemma-2B":        
    {'description':"""The Gemma model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
        \nIt was created by the [**Google's AI Team**](https://blog.google/technology/developers/gemma-open-models/) team as has over  **2 billion parameters.** \n""",
    'logo':'https://pbs.twimg.com/media/GG3sJg7X0AEaNIq.jpg'},
    "Zephyr-7B":        
    {'description':"""The Zephyr model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
        \nFrom Huggingface: \n\
        Zephyr is a series of language models that are trained to act as helpful assistants. \
        [Zephyr 7B Gemma](https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-v0.1)\
        is the third model in the series, and is a fine-tuned version of google/gemma-7b \
        that was trained on on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO)\n""",
    'logo':'https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-v0.1/resolve/main/thumbnail.png'},
    "Zephyr-7B-β":        
    {'description':"""The Zephyr model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
        \nFrom Huggingface: \n\
        Zephyr is a series of language models that are trained to act as helpful assistants. \
        [Zephyr-7B-β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)\
        is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1 \
        that was trained on on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO)\n""",
    'logo':'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'},
    "Meta-Llama-3-8B":
    {'description':"""The Llama (3) model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
        \nIt was created by the [**Meta's AI**](https://llama.meta.com/) team and has over  **8 billion parameters.** \n""",
    'logo':'Llama_logo.png'},
}


#Random dog images for error message
random_dog = ["0f476473-2d8b-415e-b944-483768418a95.jpg",
              "1bd75c81-f1d7-4e55-9310-a27595fa8762.jpg",
              "526590d2-8817-4ff0-8c62-fdcba5306d02.jpg",
              "1326984c-39b0-492c-a773-f120d747a7e2.jpg",
              "42a98d03-5ed7-4b3b-af89-7c4876cb14c3.jpg",
              "8b3317ed-2083-42ac-a575-7ae45f9fdc0d.jpg",
              "ee17f54a-83ac-44a3-8a35-e89ff7153fb4.jpg",
              "027eef85-ccc1-4a66-8967-5d74f34c8bb4.jpg",
              "08f5398d-7f89-47da-a5cd-1ed74967dc1f.jpg",
              "0fd781ff-ec46-4bdc-a4e8-24f18bf07def.jpg",
              "0fb4aeee-f949-4c7b-a6d8-05bf0736bdd1.jpg",
              "6edac66e-c0de-4e69-a9d6-b2e6f6f9001b.jpg",
              "bfb9e165-c643-4993-9b3a-7e73571672a6.jpg"]



def reset_conversation():
    '''
    Resets Conversation
    '''
    st.session_state.conversation = []
    st.session_state.messages = []
    return None
    



# Define the available models
models =[key for key in model_links.keys()]

# Create the sidebar with the dropdown for model selection
selected_model = st.sidebar.selectbox("Select Model", models)

#Create a temperature slider
temp_values = st.sidebar.slider('Select a temperature value', 0.0, 1.0, (0.5))


#Add reset button to clear conversation
st.sidebar.button('Reset Chat', on_click=reset_conversation) #Reset button


# Create model description
st.sidebar.write(f"You're now chatting with **{selected_model}**")
st.sidebar.markdown(model_info[selected_model]['description'])
st.sidebar.image(model_info[selected_model]['logo'])
st.sidebar.markdown("*Generated content may be inaccurate or false.*")





if "prev_option" not in st.session_state:
    st.session_state.prev_option = selected_model

if st.session_state.prev_option != selected_model:
    st.session_state.messages = []
    # st.write(f"Changed to {selected_model}")
    st.session_state.prev_option = selected_model
    reset_conversation()



#Pull in the model we want to use
repo_id = model_links[selected_model]


st.subheader(f'AI - {selected_model}')
# st.title(f'ChatBot Using {selected_model}')

# Set a default model
if selected_model not in st.session_state:
    st.session_state[selected_model] = model_links[selected_model] 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Accept user input
if prompt := st.chat_input(f"Hi I'm {selected_model}, ask me a question"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        try:
            stream = client.chat.completions.create(
                model=model_links[selected_model],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=temp_values,#0.5,
                stream=True,
                max_tokens=3000,
            )
    
            response = st.write_stream(stream)

        except Exception as e:
            # st.empty()
            response = "😵‍💫 Looks like someone unplugged something!\
                    \n Either the model space is being updated or something is down.\
                    \n\
                    \n Try again later. \
                    \n\
                    \n Here's a random pic of a 🐶:"
            st.write(response)
            random_dog_pick = 'https://random.dog/'+ random_dog[np.random.randint(len(random_dog))]
            st.image(random_dog_pick)
            st.write("This was the error message:")
            st.write(e)



        
    st.session_state.messages.append({"role": "assistant", "content": response})
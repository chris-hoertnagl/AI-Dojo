# Code borrowed from https://github.com/Jagadeesha89/VistaAI/tree/main

import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import time
import requests
from PIL import Image
from io import BytesIO

from model import LlavaNext, ModelConfig

model_config_json = {
    "bnb_config": {"load_in_4bit":True, "bnb_4bit_quant_type":"nf4", "bnb_4bit_compute_dtype":"bfloat16"},
    "troch_dtype":"bfloat16",
    "max_new_tokens":256
}

#Set the page config
st.set_page_config("AIChatBot",layout="wide",page_icon=":vhs:")

@st.cache_resource
def load_model(model_config):
    model = LlavaNext(ModelConfig(**model_config))
    model.load_model()
    return model

model = load_model(model_config_json)

#Function to get the PDF dcouments
def get_pdf_text(pdf_docs):
    text=""
    pdf_reader=PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text

#Function to extract the text from provided PDF
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


#TODO: fix embedding setup
#embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
#docs=new_db.similarity_search(user_question)
#def get_vector_store(text_chunks):
#    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
#    vector_store.save_local('faiss_index')


#Function to stream the animations
def load_lottiefiles(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

lottie_hi = load_lottiefiles(r'Images/AI.json')
st_lottie(lottie_hi, loop=True, quality="high", speed=1.65, key=None, height=100)

#Home page functions     
def home_page():
    st.markdown("""## <span style='color:#AE194B'>Learning about AI</span>""", unsafe_allow_html=True)
    st.markdown("""
        ###          
        <span style='color:#0D7177'>  This is a streamlit app showing different ways of interacting with AI in form of specialized ChatBots. 
        This project is focused on working only with open-source models that can be run on CPU or consumer GPUs. 
        </span>""", 
        unsafe_allow_html=True)
           
    st.divider()
    st.markdown("""## <span style='color:#9D801C'>ChatBot Overview</span>""", unsafe_allow_html=True)
    Text_chat,Image_chat,PDF_chat=st.columns(3,gap="large")
    with Text_chat:
        st.markdown("""
            ###          
            <span style='color:#0D7177'>  📝 **Navigate to TEXT CHAT** : To have a plain text conversation with the AI model.</span>""", unsafe_allow_html=True)
       
    with Image_chat:
        st.markdown("""
            ###                            
            <span style='color:#34786A'>  🛣️ **Navigate to IMAGE CHAT**: To be able to chat with text and provide images. This shows the multimodal capabilities of open-source AI.</span>""", unsafe_allow_html=True)
    with PDF_chat:
        st.markdown("""
            ###           
            <span style='color:#365B4F'>  📚 **Navigate PDF CHAT**: To chat about PDF files, showing how to build your own knowledge base and interact with it.</span>""", unsafe_allow_html=True)

#Function to chat with pdf documents
def chat_with_multipdf():

    #Initialize the session for chat
    if "chat_history_pdf" not in st.session_state:
        st.session_state.chat_history_pdf = []

    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")
    st.markdown("""<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be **Deleted**)</span>""", unsafe_allow_html=True)

    #Take the user input
    user_question=st.chat_input(placeholder="Ask a Question from the PDF Files uploaded .. ✍️📝")

    with st.chat_message("user"):
            st.markdown("Once the PDF is  uplodaded please write your question 👇")

    #Genrate the response 
    if user_question: 
        with st.spinner('Please wait Generating answer.....📖'):
            # TODO: fix pdf chat (RAG)
            # user_input(user_question)
    
             #   Append question and answer to chat history
            st.session_state.chat_history_pdf.append({"role": "user", "content": user_question})
            st.session_state.chat_history_pdf.append({"role": "assistant", "content": st.session_state.response}) 

                # Display chat history
    for message in st.session_state.chat_history_pdf:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 

    #Take the PDF documents from the user                 
    with st.sidebar:
        st.title("📁 PDF File's Section")
        lottie_hi = load_lottiefiles(r'Images/PDF.json')
        st_lottie(lottie_hi, loop=True, quality="high", speed=1.65, key=None, height=100)
        st.divider()
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ")

        if pdf_docs is not None:
            if not pdf_docs.name.endswith('.pdf'):
                st.warning("Uploaded file is not PDF format,Please upload a PDF file.")
            elif st.button("Submit & Process"):
                with st.status("Processing....",expanded=True) as status:
                    st.write("Extracting Text...")
                    time.sleep(2)
                    st.write("Converting Text into emmeddings")
                    time.sleep(1)
                    st.write("Storing all the chunks")
                    time.sleep(1)
                    status.update(label="Sucessfully processed", state="complete", expanded=False)
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks= get_text_chunks(raw_text)
                    # TODO: fix pdf chat (RAG)
                    # get_vector_store(text_chunks)  

#Function to text based response genration                   
def text_chat():

    st.header("Your AI powered Chat Agent 🤖 ")
    st.markdown("""<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be **Deleted**)</span>""", unsafe_allow_html=True)

    with st.chat_message("user"):
        st.markdown("Hi I am your AI chat Bot Ask me anything 👍")

    #Initialize the session for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history =[]

        # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])   

    if prompt := st.chat_input("Ask a Question your AI chat Agent ✍️"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            # Display assistant response in chat message container
        with st.spinner('Generating response....'):
            with st.chat_message("assistant"):
                try:
                    response = model.chat(st.session_state.chat_history)
                    print(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.markdown(response)
                except Exception as e:
                    st.error(f"An error occurred during response generation: {str(e)}")
                    # Update the chat history with the error message
                    st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})

#Functions to genrate the response based on the image
def chat_with_image():
    #Initialize the session for chat
    if "chat_history_image" not in st.session_state:
        st.session_state.chat_history_image = []

    st.header("Image 🖼️ - Chat Agent 🤖 ")
    st.markdown("""<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be **Deleted**)</span>""", unsafe_allow_html=True)

    with st.chat_message("user"):
        st.markdown("Once the Image is uploaded sucessfully, please write your question 👇") 

    #Take the user input
    user_text=st.chat_input(placeholder="Ask a Question from the Image Files uploaded .. ✍️📝")

    #Create the side bar to upload the image and process    
    with st.sidebar:
        st.title("📁 Image File's Section")
        lottie_hi = load_lottiefiles(r'Images/imag.json')
        st_lottie(lottie_hi, loop=True, quality="high", speed=1.65, key=None, height=250)
        st.divider()
        #OPtions to swt btw image upload and image URL
        option = st.radio("Choose an option", ["Upload Image", "Provide Image URL"])
        image = None
        if option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    if st.button("Process"):
                        st.image(image, caption='Uploaded Image', use_column_width=True)
                except Exception as e:
                        st.error(f"Error loading uploaded image: {str(e)}")
        elif option == "Provide Image URL":
            image_url = st.text_input("Enter Image URL:")
            if image_url:
                try:
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        if st.button("Get the Image"):
                            st.image(image, caption='Image from URL', use_column_width=True)
                    else:
                        st.error(f"Failed to retrieve image from URL. Status code: {response.status_code}")
                except Exception as e:
                    st.error(f"Error loading image from URL: {str(e)}")  
        if image is None:
            st.warning("You did not provide a valid image.")
    #Genrate the response based on the user input
    if user_text:
        with st.spinner('Please wait Generating your answer.....🕵️‍♂️'): 
            st.session_state.chat_history_image.append({"role": "user", "content": user_text})
            # Image will automaticall appened to latest message in the chat
            response=model.chat(st.session_state.chat_history_image,image)
            st.session_state.chat_history_image.append({"role": "assistant", "content": response}) 

    #Display the message
    for message in st.session_state.chat_history_image:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

#Function to clear the history               
def clear_chat_history():
    # Clear chat history when user selects an option other than "Home page"
    st.session_state.chat_history = []
    st.session_state.chat_history_image = []
    st.session_state.chat_history_pdf = []

#Main page and option menu
def main():
    selected = option_menu(
        menu_title=None,
        options=["HOME","TEXT CHAT", "IMAGE CHAT" ,"PDF CHAT"],
        icons=['house',"pen" ,'image','book'],
        default_index=0,
        menu_icon='user',
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#DCE669"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#E0EEEE",
            },
            "nav-link-selected": {"background-color": "#458B74"},
        },
    )     
    #When user select the option functions will selected
    if selected == "PDF CHAT":
        chat_with_multipdf()
    elif selected == "TEXT CHAT":
        text_chat()
    elif selected == "IMAGE CHAT":
        chat_with_image()
    else:
        clear_chat_history()
        home_page()      

if __name__ == "__main__":
    main()

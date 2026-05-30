import streamlit as st
import ollama

# Streamlit UI setup
st.title("🔊 Voice Chatbot 📝")
st.markdown("""
This app generates text using local Ollama models and converts the response to speech using Google TTS. 
""")

# Fetch available Ollama models
try:
    response = ollama.list()
    # Handle different versions of the ollama python client
    models_list = getattr(response, 'models', [])
    if not models_list and isinstance(response, dict):
        models_list = response.get('models', [])
    
    available_models = []
    for m in models_list:
        m_name = getattr(m, 'model', None) or (m.get('model') if isinstance(m, dict) else str(m))
        if m_name and "embed" not in m_name.lower():
            available_models.append(m_name)
    
    if not available_models:
        available_models = ["llama3:8b", "phi3:latest"]
except Exception:
    available_models = ["llama3:8b", "phi3:latest"]

# Select box for model selection in sidebar
st.sidebar.title("Configuration")
selected_model = st.sidebar.selectbox("Select Ollama Model", available_models)

# Language Configuration for Speech-to-Text and Text-to-Speech
LANGUAGE_MAPPING = {
    "English": {"stt": "en-US", "tts": "en"},
    "Urdu": {"stt": "ur-PK", "tts": "ur"},
    "Spanish": {"stt": "es-ES", "tts": "es"},
    "German": {"stt": "de-DE", "tts": "de"},
    "French": {"stt": "fr-FR", "tts": "fr"},
    "Arabic": {"stt": "ar-SA", "tts": "ar"},
    "Hindi": {"stt": "hi-IN", "tts": "hi"}
}
selected_language = st.sidebar.selectbox("Select Spoken Language", list(LANGUAGE_MAPPING.keys()))
stt_code = LANGUAGE_MAPPING[selected_language]["stt"]
tts_code = LANGUAGE_MAPPING[selected_language]["tts"]


if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.document_context = ""
    st.rerun()

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_context" not in st.session_state:
    st.session_state.document_context = ""

# Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "images" in msg and msg["images"]:
            st.image(msg["images"][0], caption="Uploaded Image")
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/mp3")




# Chat input with file upload feature
audio_value = None
if hasattr(st, "audio_input"):
    audio_value = st.audio_input("Record a voice message", label_visibility="collapsed")
    
prompt_value = st.chat_input("Enter your prompt for the AI...", accept_file="multiple")

process_message = False
user_text = ""
uploaded_files = []

if prompt_value:
    if isinstance(prompt_value, str):
        user_text = prompt_value
    else:
        # ChatInputValue
        user_text = getattr(prompt_value, "text", "")
        files_attr = getattr(prompt_value, "files", [])
        if isinstance(files_attr, list):
            uploaded_files = files_attr
        elif files_attr is not None:
            uploaded_files = [files_attr]
    process_message = True

elif audio_value:
    file_id = getattr(audio_value, "file_id", str(audio_value.size))
    if "last_audio_id" not in st.session_state or st.session_state.last_audio_id != file_id:
        st.session_state.last_audio_id = file_id
        with st.spinner("Transcribing audio..."):
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile(audio_value) as source:
                    audio_data = r.record(source)
                user_text = r.recognize_google(audio_data, language=stt_code)
                if user_text:
                    process_message = True
                    st.toast("Audio transcribed successfully!")
            except ImportError:
                st.error("Please install the SpeechRecognition package: `pip install SpeechRecognition`")
            except sr.UnknownValueError:
                st.warning("Could not understand the audio. Please try speaking a bit louder or more clearly.")
            except sr.RequestError as e:
                st.error(f"Network error from Google Speech Recognition service: {e}")
            except Exception as e:
                st.error(f"Audio transcription error: {str(e) or 'Unknown audio format'}")

if process_message and (user_text or uploaded_files):
    # Process uploaded files
    image_bytes = None
    new_context = ""
    
    for f in uploaded_files:
        if f.type == "application/pdf":
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(f)
            extracted_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            if not extracted_text.strip():
                st.warning(f"Could not extract text from the PDF. It might be a scanned image-based PDF without selectable text.")
            else:
                new_context += f"\n[Uploaded PDF Content]:\n{extracted_text}\n"
                st.toast("PDF text extracted and added to context.")
        elif f.type in ["image/png", "image/jpeg", "image/jpg"]:
            image_bytes = f.getvalue()

    if new_context:
        st.session_state.document_context += new_context

    # Store user message for display
    user_msg_to_store = {"role": "user", "content": user_text}
    if image_bytes:
        user_msg_to_store["images"] = [image_bytes]
    
    st.session_state.messages.append(user_msg_to_store)
    
    with st.chat_message("user"):
        st.write(user_text)
        if image_bytes:
            st.image(image_bytes, caption="Uploaded Image")

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response with Ollama..."):
            try:
                # Prepare messages for Ollama API
                ollama_messages = []
                for idx, m in enumerate(st.session_state.messages):
                    msg_obj = {'role': m['role'], 'content': m['content']}
                    if 'images' in m:
                        msg_obj['images'] = m['images']
                    
                    # Inject the document context into the current latest query
                    if idx == len(st.session_state.messages) - 1:
                        enhanced_content = m['content']
                        if st.session_state.document_context:
                            enhanced_content = f"Context:\n{st.session_state.document_context}\n\nUser Question: {enhanced_content}"
                        # Enforce response in the selected language
                        enhanced_content += f"\n\n[System Instruction: Please respond in {selected_language}.]"
                        msg_obj['content'] = enhanced_content

                    ollama_messages.append(msg_obj)
                
                has_images = any('images' in m for m in ollama_messages)
                vision_models = ['llava', 'vision', 'bakllava']
                if has_images and not any(v in selected_model.lower() for v in vision_models):
                    st.warning(f"⚠️ You attached an image, but '{selected_model}' is a text-only model and cannot see images! Please select a vision model like 'llava' from the sidebar to analyze images.")
                    st.stop()
                    
                response = ollama.chat(model=selected_model, messages=ollama_messages)
                ai_text = response['message']['content']
                st.write(ai_text)
                
                ai_msg_to_store = {"role": "assistant", "content": ai_text}
                
                # Generate audio response using gTTS
                try:
                    from gtts import gTTS
                    import io
                    tts = gTTS(text=ai_text, lang=tts_code)
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    audio_bytes = fp.getvalue()
                    ai_msg_to_store["audio"] = audio_bytes
                    st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                except Exception as e:
                    st.error(f"Failed to generate audio response: {e}")
                
                st.session_state.messages.append(ai_msg_to_store)
            except Exception as e:
                st.error(f"An error occurred: {e}")


# Adding the HTML footer
footer_css = """
<style>
.footer {
    position: fixed;
    right: 0;
    bottom: 0;
    width: auto;
    background-color: transparent;
    color: var(--text-color);
    text-align: right;
    padding-right: 10px;
}
</style>
"""

footer_html = """
<div class="footer">
    <p>Credit: Dr. Aammar Tufail | Phd | Data Scientist | Bioinformatician (<a href="https://www.youtube.com/@Codanics" target="_blank">CODANICS</a>)</p>
</div>
"""
st.markdown(footer_css, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)

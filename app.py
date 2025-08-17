import streamlit as st
from pathlib import Path
import time
import asyncio

# -------------------------------------------------
# 1️⃣ Page config & global style
# -------------------------------------------------
st.set_page_config(
    page_title="My Streamlit Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# 2️⃣ Caching heavy resources (model, API client)
# -------------------------------------------------
@st.experimental_singleton
def load_model():
    # Example: load a huggingface model once per session
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------------------------
# 3️⃣ Session state helpers
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []   # chat history

# -------------------------------------------------
# 4️⃣ UI layout
# -------------------------------------------------
st.sidebar.title("⚙️ Controls")
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_len = st.sidebar.number_input("Max tokens", 10, 200, 50)

st.title("🗨️ Simple LLM Chat Demo")
user_input = st.chat_input("Ask me anything…")

# -------------------------------------------------
# 5️⃣ Chat handling with streaming output
# -------------------------------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Run generation in a background thread to keep UI responsive
        async def generate():
            input_ids = tokenizer(user_input, return_tensors="pt").input_ids
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_len,
                temperature=temp,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=False,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Simple streaming simulation
            for i in range(0, len(text), 20):
                placeholder.markdown(text[: i + 20] + "▍")
                await asyncio.sleep(0.05)
            placeholder.markdown(text)
        asyncio.run(generate())
    st.session_state.messages.append({"role": "assistant", "content": placeholder.markdown})

# -------------------------------------------------
# 6️⃣ Render full conversation
# -------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

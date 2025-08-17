# -------------------------------------------------
# 1️⃣  Imports & page config
# -------------------------------------------------
import streamlit as st
import asyncio
import time
from pathlib import Path
import torch

st.set_page_config(
    page_title="LLM Chat Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# 2️⃣  Cache heavy resources (model & tokenizer)
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the model *once* per session (or per process).  The
    `cache_resource` decorator guarantees the object lives for the
    whole app lifetime, avoiding repeated downloads.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- Model & tokenizer -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # ---- Move to GPU if possible -------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = model.to(device).to(dtype)

    return tokenizer, model, device, dtype

tokenizer, model, device, dtype = load_model()

# -------------------------------------------------
# 3️⃣  Session‑state helpers (chat history)
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []   # list of dicts: {"role": "...", "content": "…"}

# -------------------------------------------------
# 4️⃣  Sidebar controls
# -------------------------------------------------
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
max_new_tokens = st.sidebar.number_input(
    "Max new tokens", min_value=10, max_value=200, value=50, step=5
)

# Optional “clear chat” button
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# -------------------------------------------------
# 5️⃣  Main UI
# -------------------------------------------------
st.title("🗨️ Simple LLM Chat Demo")
user_prompt = st.chat_input("Ask me anything…")

# -------------------------------------------------
# 6️⃣  Generate answer (with streaming effect)
# -------------------------------------------------
if user_prompt:
    # ---- store user message -------------------------------------------------
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # ---- placeholder for assistant response ---------------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()

        # ---- generation loop -------------------------------------------------
        with st.spinner("🧠 Thinking…"):
            # Tokenise & move to proper device
            input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(device)

            # Generation (no grad, GPU‑aware)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode once – we’ll stream the text ourselves
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Simulate streaming (you can tweak chunk size / sleep time)
            chunk_size = 25
            for i in range(0, len(full_text), chunk_size):
                displayed = full_text[: i + chunk_size]
                placeholder.markdown(displayed + "▍")
                time.sleep(0.03)          # small pause for the “typing” effect

            # Final clean render
            placeholder.markdown(full_text)

        # ---- store assistant reply -------------------------------------------
        st.session_state.messages.append(
            {"role": "assistant", "content": full_text}
        )

# -------------------------------------------------
# 7️⃣  Render the full conversation (chronological order)
# -------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:  # assistant
        st.chat_message("assistant").write(msg["content"])

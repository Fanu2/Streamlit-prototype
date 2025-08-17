import streamlit as st
from pathlib import Path
import asyncio

# -------------------------------------------------
# 1️⃣ Page config
# -------------------------------------------------
st.set_page_config(
    page_title="LLM Chat Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# 2️⃣ Cache heavy resources (model, tokenizer)
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the model *once* per session (or per process, depending on the
    deployment).  Using `cache_resource` tells Streamlit that the returned
    object is a heavyweight resource that should be kept alive.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------------------------
# 3️⃣ Session‑state helpers (chat history)
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []   # list of dicts: {"role": "...", "content": "…"}

# -------------------------------------------------
# 4️⃣ Sidebar controls
# -------------------------------------------------
st.sidebar.title("⚙️ Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
max_new_tokens = st.sidebar.number_input("Max new tokens", 10, 200, 50)

# -------------------------------------------------
# 5️⃣ Main UI
# -------------------------------------------------
st.title("🗨️ Simple LLM Chat Demo")
user_prompt = st.chat_input("Ask me anything…")

# -------------------------------------------------
# 6️⃣ Streaming generation (async loop)
# -------------------------------------------------
if user_prompt:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Show assistant placeholder while we generate
    with st.chat_message("assistant"):
        placeholder = st.empty()

        async def generate():
            # Encode input
            input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids

            # Generate tokens (no need for async here, but we stream manually)
            output_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Simulate streaming by revealing chunks
            for i in range(0, len(full_text), 20):
                chunk = full_text[: i + 20]
                placeholder.markdown(chunk + "▍")
                await asyncio.sleep(0.03)   # tweak for smoother feel
            placeholder.markdown(full_text)   # final clean output

        # Run the async generator
        asyncio.run(generate())

    # Store assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": placeholder.markdown}
    )

# -------------------------------------------------
# 7️⃣ Render the full conversation
# -------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

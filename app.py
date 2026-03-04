import streamlit as st
import requests

# 🔴 For local testing only (NOT for GitHub)
API_KEY = "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"

st.set_page_config(page_title="AI Tutor", page_icon="🎓")

st.title("🎓 AI Tutor")
st.caption("Ask any topic and learn step-by-step.")

if not API_KEY:
    st.error("API key missing.")
    st.stop()

user_question = st.text_input("Ask me anything:")

if user_question:
    with st.spinner("Thinking..."):

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openrouter/free",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an AI Tutor.
Use this structure:
1) Simple explanation
2) Real-world example
3) Mathematical explanation
4) 2 quiz questions
5) Suggest next topic
"""
                    },
                    {"role": "user", "content": user_question}
                ],
                "temperature": 0.7
            },
            timeout=60
        )

        result = response.json()

        if response.status_code == 200 and "choices" in result:
            answer = result["choices"][0]["message"]["content"]
            st.markdown(answer)
        else:
            st.error("API request failed.")
            st.write(result)
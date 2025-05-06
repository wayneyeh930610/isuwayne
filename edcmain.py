import streamlit as st
from wsx import qa_agent_with_local_faiss
import os

MAX_HISTORY = 4  # 最多保留 5 筆歷史對話

def main():
    st.title("義守大學AI小幫手")

    # 初始化對話記憶
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 確認 API Key 是否存在
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("❌ 未偵測到 OpenAI API 金鑰，請設定環境變數 OPENAI_API_KEY。")
        return

    # 問題輸入框
    user_question = st.text_input("輸入你的問題：")

    if user_question and st.button("進行查詢"):
        response = qa_agent_with_local_faiss(
            question=user_question,
            chat_history=st.session_state.chat_history,
            faiss_folder_path="faiss_index"
        )
        if "answer" in response:
            st.session_state.chat_history.append((user_question, response["answer"]))
            # 限制保留近 5 筆對話紀錄
            if len(st.session_state.chat_history) > MAX_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
            st.write("回答：", response["answer"])
        else:
            st.write("❌ 沒有回答可以提供，請檢查向量資料庫是否正確載入。")

    # 顯示過往對話
    if st.session_state.chat_history:
        st.markdown("#### 🧠 對話紀錄")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.write(f"**Q{i+1}：** {q}")
            st.write(f"**A{i+1}：** {a}")

if __name__ == "__main__":
    main()


#pip install streamlit langchain langchain-community openai faiss-cpu

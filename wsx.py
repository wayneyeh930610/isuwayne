import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

def qa_agent_with_local_faiss(question, chat_history, faiss_folder_path=r"C:\Users\cindy\OneDrive\桌面\義守大學.pdf"):
    """
    使用本地端的 FAISS 向量資料庫進行查詢並產生回答（MMR 檢索方式）。
    """

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("❌ 找不到 OpenAI API 金鑰，請確認已在環境變數中設定 OPENAI_API_KEY。")

    # 初始化語言模型與 embedding 模型
    model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key
    )

    # 載入 FAISS 向量資料庫
    try:
        db = FAISS.load_local(faiss_folder_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise FileNotFoundError(f"❌ 找不到 FAISS 向量資料庫，請確認資料庫位置：{faiss_folder_path}\n錯誤訊息: {str(e)}")

    # 使用 MMR 檢索（避免重複段落，提升多樣性）
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    response = qa.invoke({
        "chat_history": chat_history,
        "question": question
    })

    return response




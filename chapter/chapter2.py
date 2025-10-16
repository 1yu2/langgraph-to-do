from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from config import  OPENAI_API_KEY,OPENAI_API_BASE

# ========== 工具定义 ==========
@tool
def search(query: str):
    """模拟网页搜索功能，用于查询天气"""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 16 degrees and foggy."
    return "It's 32 degrees and sunny."

tools = [search]
tool_node = ToolNode(tools)

# ========== 模型定义 ==========
# 使用 SiliconFlow 提供的 Qwen 模型
model = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0
).bind_tools(tools)

# ========== 工作流定义 ==========
workflow = StateGraph(MessagesState)

def call_model(state):
    """agent节点逻辑：调用模型进行推理"""
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 起始节点
workflow.set_entry_point("agent")

# 定义条件跳转
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END
# 条件边
workflow.add_conditional_edges("agent", should_continue)
# 普通边
workflow.add_edge("tools", "agent")

# 编译图
app = workflow.compile()

# ========== 执行 ==========
final_state = app.invoke({
    "messages": [{"role": "user", "content": "What is the weather in San Francisco"}]
})

# 输出结果
print(final_state["messages"][-1].content)

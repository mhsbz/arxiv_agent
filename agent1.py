from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOpenAI 

from langchain_deepseek import ChatDeepSeek
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import json
from fastapi.middleware.cors import CORSMiddleware
from chat_arxiv_search import Reader, ArxivParams

# 定义论文搜索工具
def search_papers(query: str,key_word: str) -> List[dict]:
    """
    论文搜索工具函数
    """
    args_dict = {
            "query": query,
            "key_word": key_word,
            "page_num": 1,
            "max_results": 1,
            "days": 3,
            "sort": "web",
            "save_image": False,
            "file_format": "md",
            "language": "zh"
        }
        # 实例化 ArxivParams 与 Reader

    # return ["深度学习论文1", "深度学习论文2", "深度学习论文3"]

    args_obj = ArxivParams(**args_dict)
    reader = Reader(key_word=args_obj.key_word, query=args_obj.query, args=args_obj)

    paper_list = reader.get_arxiv_web(args_obj, page_num=args_obj.page_num, days=args_obj.days)
    if not paper_list:
        return {"result": "未找到符合条件的论文。"}
    # 生成论文摘要总结，注意该过程可能比较耗时
    result_text = reader.summary_with_chat(paper_list)
    return {"result": result_text}

# 创建工具列表
tools = [
    Tool(
            name="SearchPapers",
            func=lambda params: search_papers(*params.split(",", 1)),
            description="用于搜索学术论文的工具。参数格式为：'query,key_word'，其中query为搜索查询语句，key_word为搜索关键词。两个参数必须用英文逗号分隔，且都不能为空。示例：'traffic flow prediction,AI prediction'"
        )
]

# 定义Agent提示模板
template = """你是一个学术助手，你的任务是帮助用户搜索相关论文。你必须且只能通过使用提供的工具来完成用户的需求，不能自行搜索或提供未经工具验证的信息。

可用的工具列表（{tool_names}）:
{tools}

用户输入: {input}

请严格按照以下步骤思考和执行:
1. 仔细分析用户需求，明确搜索目标
2. 从用户输入中提取关键搜索词
3. 使用工具列表中的工具进行搜索，不要试图自行搜索或猜测
4. 直接返回工具的搜索结果，不要进行任何加工或整理
5. 工具调用完成后立即结束，不要重复调用工具

注意：
- 你不能提供任何未经工具验证的论文信息或搜索结果
- 不要对工具返回的结果做任何修改或整理，原样返回即可
- 每次对话中只调用一次工具，调用后直接返回结果

请使用以下格式：
Action: 工具名称
Action Input: 工具输入
Observation: 工具输出

完成后使用：
Final Answer: 工具输出

{agent_scratchpad}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]  # 添加缺少的输入变量
)

# os.environ["OPENAI_API_KEY"] = "sk-hZmoJI9LTLilVwgS4aFa08028bDa4dAbAf60Fe0f9dCc214e"
# os.environ["OPENAI_BASE_URL"] = "https://api.zeroai.link/v1"

# 初始化LLM
# 从环境变量获取OpenAI配置
# 初始化LLM - 只保留一个定义
# llm = ChatOpenAI(
#     model="gpt-4o", 
#     temperature=0.7,
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     openai_api_base=os.getenv("OPENAI_BASE_URL")
# )

llm = ChatDeepSeek(
        api_key="sk-2a4f73d6bd80475989822890a1456025",
        model_name="deepseek-chat",  # 使用适合代码生成的模型
        temperature=0.7,
    )

# 创建Agent执行器
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,  # 添加配置
    output_key="output"  # 明确指定输出键
)

# if __name__ == "__main__":
#     # 使用agent_executor而不是agent
#     result = agent_executor.invoke({"input": "请推荐几篇关于深度学习的最新论文"})
#     print("\n最终结果:", result)

# 创建FastAPI应用
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 定义请求模型
class SearchRequest(BaseModel):
    prompt: str

# 定义响应模型
class SearchResponse(BaseModel):
    result: str

# 修改API端点支持POST和GET方法
@app.get("/search")
async def search_papers_get(prompt: str):
    """
    处理论文搜索GET请求的API端点
    """
    try:
        # 调用agent执行搜索
        print("prompt--", prompt)
        result = agent_executor.invoke({"input": prompt})

        print("result--", result)
        
        # 提取需要的结果信息
        output = result.get("output", "")
        
        # 构建可序列化的响应数据
        response_data = {
            "output": output,
            "status": "success"
        }
        
        # 将结果转换为JSON字符串
        result_json = json.dumps(response_data, ensure_ascii=False)
        return SearchResponse(result=result_json)
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务器配置
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOpenAI 

from langchain_deepseek import ChatDeepSeek
from typing import List
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import json
from fastapi.middleware.cors import CORSMiddleware
import requests
from fastapi import UploadFile, File
import os

# 定义论文搜索工具
def search_papers(query: str) -> dict:
    """
    论文搜索工具函数
    """
    
    # 构建请求参数
    params = {
        'query': query
    }
    
    try:
        # 调用本地API
        response = requests.get('http://localhost:8002/api/arxiv', params=params)
        response.raise_for_status()  # 检查请求是否成功
        
        # 解析响应
        result = response.json()
        
        if not result:
            return {"result": "搜索论文失败，未获取到结果。"}
        print("request success...")
        print(result)
        # 解析响应
        return result
        
    except requests.RequestException as e:
        return {"result": f"API请求失败: {str(e)}"}

def translate_pdf(pdf_path: str) -> dict:
    """
    PDF翻译工具函数
    """
    try:
        # 构建请求数据
        data = {
            "pdf_path": pdf_path,
            "bilingual": False
        }
        
        # 调用翻译API
        response = requests.post(
            'http://localhost:8003/translate',
            headers={"Content-Type": "application/json"},
            json=data
        )
        response.raise_for_status()
            
        result = response.json()

        print("request success...")
        
        if not result:
            return {"result": "翻译本文失败，未获取到结果。"}
        
        # 解析响应
        return result
        
    except requests.RequestException as e:
        return {"result": f"翻译API请求失败: {str(e)}"}

def polish_text(pdf_path: str, target_lang: str) -> dict:
    """
    文本润色工具函数
    """
    try:
        # 构建请求数据
        data = {
            "pdf_path": pdf_path,
            "target_lang": target_lang,
            "bilingual": True
        }
        
        # 调用润色API
        response = requests.post(
            'http://localhost:8003/polish',
            headers={"Content-Type": "application/json"},
            json=data
        )
        response.raise_for_status()

        result = response.json()
        print("request success...")
        if not result:
            return {"result": "润色本文失败，未获取到结果。"}
        
        # 解析响应
        return result
        
    except requests.RequestException as e:
        return {"result": f"润色API请求失败: {str(e)}"}

def summarize_pdf(pdf_path: str, target_lang: str="中文") -> dict:
    """
    PDF摘要生成工具函数
    """
    try:
        # 构建请求数据
        data = {
            "pdf_path": pdf_path,
            "target_lang": target_lang
        }
        
        # 调用摘要生成API
        response = requests.post(
            'http://localhost:8003/summarize',
            headers={"Content-Type": "application/json"},
            json=data
        )
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        print("request success...")
        if not result:
            return {"result": "摘要生成失败，未获取到结果。"}
            
        return result
        
    except requests.RequestException as e:
        return {"result": f"摘要生成API请求失败: {str(e)}"}

# 创建工具列表
tools = [
    Tool(
        name="SearchPapers",
        func=search_papers,
        description="用于搜索学术论文的工具。输入参数为搜索查询语，需要把查询语句转化成英文。示例输入：'traffic flow prediction'"
    ),
    Tool(
        name="TranslatePDF",
        func=translate_pdf,
        description="用于翻译PDF文档的工具。输入参数为PDF文件路径。工具会返回翻译后的文本内容。示例输入：'/path/to/paper.pdf'"
    ),
    Tool(
        name="PolishText",
        func=lambda params: polish_text(*params.split(",", 1)),
        description="用于润色论文的工具。参数格式为：'pdf_path,target_lang'，其中pdf_path为需要润色的论文的文件地址，target_lang为对应的语言（中文/英语）。两个参数必须用英文逗号分隔，且都不能为空。示例：'./papers/paper.pdf,中文'"
    ),
    Tool(
        name="SummarizePDF",
        func=lambda params: summarize_pdf(*params.split(",", 1)),
        description="用于生成PDF文档摘要的工具。输入参数为PDF文件路径和目标语言。工具会返回文档的关键内容摘要。两个参数必须用英文逗号分隔，且都不能为空。示例输入：'./papers/paper.pdf,中文'"
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


llm = ChatDeepSeek(
        api_key="sk-660fc5900fe04d55b490dfe51eb0ecd3",
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


# 移除响应模型定义，直接返回字典数据
# 修改API端点支持POST和GET方法
@app.get("/invoke")
async def search_papers_get(prompt: str):
    """
    处理论文搜索GET请求的API端点
    """
    try:
        # 调用agent执行搜索
        print("prompt--", prompt)
        result = agent_executor.invoke({"input": prompt})

       
        
        # 提取需要的结果信息
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        # 构建可序列化的响应数据
        response_data = {
            "output": output,
            "intermediate_steps": intermediate_steps,
            "status": "success"
        }

        print("response_data--", response_data)
        
        # 直接返回字典数据,FastAPI会自动处理JSON序列化
        return response_data
        
    except Exception as e:
        # 记录详细错误信息
        print(f"Error in search_papers_get: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "处理请求时发生错误",
                "error": str(e)
            }
        )

# 定义上传文件的保存目录
UPLOAD_DIR = "uploads"

# 确保上传目录存在
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    处理PDF文件上传的API端点
    """
    try:
        # 检查文件类型
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件上传")
        
        # 构建文件保存路径
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "status": "success",
            "message": "文件上传成功",
            "file_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务器配置
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



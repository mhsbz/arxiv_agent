from flask import Flask, request, jsonify
import configparser
import os
import sys
from openai import OpenAI
import tenacity
import tiktoken
import re
from functools import lru_cache
import fitz  # PyMuPDF

app = Flask(__name__)

class LazyloadTiktoken:
    def __init__(self, model):
        self.model = model

    @staticmethod
    @lru_cache(maxsize=128)
    def get_encoder(model):
        return tiktoken.encoding_for_model(model)

    def encode(self, *args, **kwargs):
        encoder = self.get_encoder(self.model)
        return encoder.encode(*args, **kwargs)

def parse_pdf(path):
    try:
        doc = fitz.open(path)
        meta = doc.metadata
        pdf_data = {
            'title': meta.get('title', '').strip(),
            'authors': [a.strip() for a in meta.get('author', '').split(',') if a.strip()],
            'abstract': '',
            'sections': []
        }

        toc = doc.get_toc()
        sections = []
        current_section = None
        
        for level, title, page in toc:
            if level == 1:
                if current_section:
                    sections.append(current_section)
                current_section = {'heading': title, 'text': ''}
        
        if not sections:
            for page_num in range(len(doc)):
                text = doc.load_page(page_num).get_text("text")
                sections.append({'heading': f"Page {page_num+1}", 'text': text})

        for section in sections:
            start_page = next((i+1 for i, t in enumerate(toc) if t[1] == section['heading']), 1)
            end_page = len(doc)
            
            content = []
            for page_num in range(start_page-1, end_page):
                content.append(doc.load_page(page_num).get_text("text"))
            section['text'] = '\n'.join(content)

        pdf_data['section_names'] = [s['heading'] for s in sections]
        pdf_data['section_texts'] = [s['text'] for s in sections]
        
        return pdf_data
    except Exception as e:
        raise RuntimeError(f"PDF解析失败: {str(e)}")

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                stop=tenacity.stop_after_attempt(5),
                reraise=True)
def chat_translate(client, text, model, domain="", is_title=False, bilingual=False):
    messages = [
        {"role": "system", "content": "您是专业的学术论文翻译专家"},
        {"role": "user", "content": f"""
            请将以下内容翻译为中文，领域：{domain}
            {'【标题翻译要求】保留英文原标题' if is_title else ''}
            {'【对照格式】需要中英对照' if bilingual else ''}
            输入内容：{text}
            
            输出要求：
            1. 专业术语保留英文并括号标注
            2. 使用Markdown格式
            3. 标题层级保持原样
            {'' if is_title else '4. 保留原文段落结构'}
            """}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=3000
    )
    return response.choices[0].message.content

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                stop=tenacity.stop_after_attempt(5),
                reraise=True)
def chat_polish(client, text, model, target_lang="英语", bilingual=False):
    lang_instruction = {
        "英语": "将以下学术内容润色为更地道的英文",
        "中文": "将以下学术内容润色为更地道的中文"
    }.get(target_lang, "润色以下学术内容")
    
    messages = [
        {"role": "system", "content": "您是专业的学术论文润色专家"},
        {"role": "user", "content": f"""
            {lang_instruction}
            {'【对照格式】需要原文与润色后对照' if bilingual else ''}
            输入内容：{text}
            
            输出要求：
            1. 专业术语保留原文并括号标注解释（如有必要）
            2. 使用Markdown格式
            3. 保持原文段落结构和标题层级
            4. 改善学术表达的专业性和流畅性
            """}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=3000
    )
    return response.choices[0].message.content

def load_config():
    config = configparser.ConfigParser()
    config.read('apikey.ini')
    return {
        'api_key': config.get('OpenAI', 'OPENAI_API_KEYS').strip("[]' ").split(', ')[0],
        'api_base': config.get('OpenAI', 'OPENAI_API_BASE'),
        'model': config.get('OpenAI', 'CHATGPT_MODEL')
    }

def get_openai_client():
    config = load_config()
    return OpenAI(
        api_key=config['api_key'],
        base_url=config['api_base']
    )

def generate_markdown(client, pdf_data, model, bilingual):
    md_content = []
    tokenizer = LazyloadTiktoken(model)
    
    if 'title' in pdf_data:
        title_trans = chat_translate(
            client=client,
            text=pdf_data['title'],
            model=model,
            is_title=True,
            bilingual=bilingual
        )
        md_content.append(f"# {title_trans}\n")
        if bilingual:
            md_content.append(f"## {pdf_data['title']}\n")

    if 'abstract' in pdf_data:
        abstract_trans = chat_translate(
            client=client,
            text=pdf_data['abstract'],
            model=model,
            bilingual=bilingual
        )
        md_content.append(f"\n## 摘要\n{abstract_trans}\n")
        if bilingual:
            md_content.append(f"\n## Abstract\n{pdf_data['abstract']}\n")

    for name, text in zip(pdf_data['section_names'], pdf_data['section_texts']):
        section_trans = chat_translate(
            client=client,
            text=f"{name}\n{text}",
            model=model,
            bilingual=bilingual
        )
        md_content.append(f"\n{section_trans}\n")
    
    return "\n".join(md_content)

def generate_polish_markdown(client, pdf_data, model, target_lang, bilingual):
    md_content = []
    
    if 'title' in pdf_data:
        title_polish = chat_polish(
            client=client,
            text=pdf_data['title'],
            model=model,
            target_lang=target_lang,
            bilingual=bilingual
        )
        md_content.append(f"# {title_polish}\n")
        if bilingual:
            md_content.append(f"## 原文标题\n{pdf_data['title']}\n")

    if 'abstract' in pdf_data:
        abstract_polish = chat_polish(
            client=client,
            text=pdf_data['abstract'],
            model=model,
            target_lang=target_lang,
            bilingual=bilingual
        )
        md_content.append(f"\n## 摘要\n{abstract_polish}\n")
        if bilingual:
            md_content.append(f"\n## 原文摘要\n{pdf_data['abstract']}\n")

    for name, text in zip(pdf_data['section_names'], pdf_data['section_texts']):
        section_polish = chat_polish(
            client=client,
            text=f"{name}\n{text}",
            model=model,
            target_lang=target_lang,
            bilingual=bilingual
        )
        md_content.append(f"\n{section_polish}\n")
    
    return "\n".join(md_content)

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                stop=tenacity.stop_after_attempt(5),
                reraise=True)
def chat_summary(client, text, model, target_lang="中文"):
    lang_map = {"中文": "中文", "英语": "英文"}
    messages = [
        {"role": "system", "content": "您是专业的学术论文总结专家"},
        {"role": "user", "content": f"""
            请用{lang_map[target_lang]}对以下学术论文内容进行结构化总结：
            
            输入内容：{text}
            
            总结要求：
            1. 包含论文标题、作者、核心贡献
            2. 分点总结研究方法与创新点
            3. 提炼关键结论与学术价值
            4. 使用Markdown格式
            5. 重要专业术语保留英文原词
            6. 结构清晰包含以下部分：
               - 论文概览
               - 核心贡献
               - 研究方法
               - 关键结论
               - 学术价值
            """}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=2000
    )
    return response.choices[0].message.content

def generate_summary_markdown(client, pdf_data, model, target_lang):
    full_text = []
    if pdf_data.get('title'):
        full_text.append(f"Title: {pdf_data['title']}")
    if pdf_data.get('authors'):
        full_text.append(f"Authors: {', '.join(pdf_data['authors'])}")
    if pdf_data.get('abstract'):
        full_text.append(f"Abstract: {pdf_data['abstract']}")
    for text in pdf_data.get('section_texts', []):
        full_text.append(text)
    
    full_content = "\n\n".join(full_text)
    return chat_summary(client, full_content, model, target_lang)

@app.route('/translate', methods=['POST'])
def translate_handler():
    print("Received request to translate PDF")
    try:
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({"error": "请求头需设置为 application/json"}), 415
            
        data = request.get_json(silent=True) or {}
        pdf_path = data.get('pdf_path')
        bilingual = data.get('bilingual', False)
        
        if not pdf_path:
            print("PDF path is missing")
            return jsonify({"error": "缺少pdf_path参数"}), 400
        if not os.path.exists(pdf_path):
            print("PDF file does not exist")
            return jsonify({"error": "PDF文件不存在"}), 404

        client = get_openai_client()
        config = load_config()
        pdf_data = parse_pdf(pdf_path)
        
        markdown_output = generate_markdown(
            client=client,
            pdf_data=pdf_data,
            model=config['model'],
            bilingual=bilingual
        )
        
        return jsonify({
            "status": "success",
            "markdown": markdown_output,
            "metadata": {
                "title": pdf_data.get('title', ''),
                "authors": pdf_data.get('authors', []),
                "sections": len(pdf_data.get('section_names', []))
            }
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/polish', methods=['POST'])
def polish_handler():
    try:
        if not request.is_json:
            return jsonify({"error": "请求头需设置为 application/json"}), 415
            
        data = request.get_json(silent=True) or {}
        pdf_path = data.get('pdf_path')
        target_lang = data.get('target_lang', '中文')
        bilingual = data.get('bilingual', False)
        
        if not pdf_path:
            return jsonify({"error": "缺少pdf_path参数"}), 400
        if not os.path.exists(pdf_path):
            return jsonify({"error": "PDF文件不存在"}), 404
        if target_lang not in ["英语", "中文"]:
            return jsonify({"error": "无效的目标语言，仅支持英语/中文"}), 400

        client = get_openai_client()
        config = load_config()
        pdf_data = parse_pdf(pdf_path)
        
        markdown_output = generate_polish_markdown(
            client=client,
            pdf_data=pdf_data,
            model=config['model'],
            target_lang=target_lang,
            bilingual=bilingual
        )
        
        return jsonify({
            "status": "success",
            "markdown": markdown_output,
            "metadata": {
                "title": pdf_data.get('title', ''),
                "authors": pdf_data.get('authors', []),
                "sections": len(pdf_data.get('section_names', []))
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/summarize', methods=['POST'])
def summarize_handler():
    try:
        if not request.is_json:
            return jsonify({"error": "请求头需设置为 application/json"}), 415
            
        data = request.get_json(silent=True) or {}
        pdf_path = data.get('pdf_path')
        target_lang = data.get('target_lang', '中文')
        
        if not pdf_path:
            return jsonify({"error": "缺少pdf_path参数"}), 400
        if not os.path.exists(pdf_path):
            return jsonify({"error": "PDF文件不存在"}), 404
        if target_lang not in ["英语", "中文"]:
            return jsonify({"error": "无效的目标语言，仅支持英语/中文"}), 400

        client = get_openai_client()
        config = load_config()
        pdf_data = parse_pdf(pdf_path)
        
        markdown_output = generate_summary_markdown(
            client=client,
            pdf_data=pdf_data,
            model=config['model'],
            target_lang=target_lang
        )
        
        return jsonify({
            "status": "success",
            "markdown": markdown_output,
            "metadata": {
                "title": pdf_data.get('title', ''),
                "authors": pdf_data.get('authors', []),
                "sections": len(pdf_data.get('section_names', []))
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=True)
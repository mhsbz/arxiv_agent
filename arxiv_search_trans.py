#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import datetime
import openai
import configparser
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 读取配置文件
config = configparser.ConfigParser()
config.read('apikey.ini')

# 初始化OpenAI客户端
openai_api_keys = config.get('OpenAI', 'OPENAI_API_KEYS')[
    1:-1].replace('\'', '').split(',')
openai_api_base = config.get('OpenAI', 'OPENAI_API_BASE')
client = openai.OpenAI(
    api_key=openai_api_keys[0].strip(),
    base_url=openai_api_base
)


def get_arxiv_info(url):
    """
    从 arXiv 页面直接获取论文详细信息
    """
    try:
        # 处理不同 URL 格式
        if 'arxiv.org/abs/' in url:
            url_abs = url
        elif 'arxiv.org/pdf/' in url:
            url_abs = url.replace('pdf', 'abs').replace('.pdf', '')
        else:
            raise ValueError("无效的 arXiv 链接")
        # 发送请求获取页面内容
        response = requests.get(url_abs, timeout=10)
        response.raise_for_status()

        bs = BeautifulSoup(response.text, 'html.parser')
        info = {'title': '', 'abstract': '', 'authors': '', 'year': ''}

        # 提取标题
        title_tag = bs.find('h1', {'class': 'title mathjax'})
        if title_tag:
            info['title'] = title_tag.text.replace('Title:', '').strip()

        # 提取摘要
        abstract_tag = bs.find('blockquote', {'class': 'abstract mathjax'})
        if abstract_tag:
            info['abstract'] = abstract_tag.text.replace(
                'Abstract:', '').replace('\n', ' ').strip()

        # 提取作者
        authors_tag = bs.find('div', {'class': 'authors'})
        if authors_tag:
            info['authors'] = authors_tag.text.replace('Authors:', '').strip()

        # 提取年份
        date_tag = bs.find('div', {'class': 'dateline'})
        if date_tag:
            match = re.search(r'\d{4}', date_tag.text)
            if match:
                info['year'] = match.group(0)

        return info

    except Exception as e:
        raise Exception(f"获取论文信息失败: {str(e)}")


def search_arxiv(query, max_results=3, days=10, page_num=1):
    """
    arXiv 论文搜索函数
    """
    base_url = "https://arxiv.org/search/?"
    params = {
        "query": query,
        "searchtype": "all",
        "abstracts": "show",
        "order": "-announced_date_first",
        "size": 50
    }
    results = []

    for page in range(page_num):
        params["start"] = page * 50
        search_url = base_url + requests.compat.urlencode(params)

        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            today = datetime.date.today()
            last_days = datetime.timedelta(days=days)

            for article in soup.find_all("li", class_="arxiv-result"):
                try:
                    # 提取标题
                    title_tag = article.find("p", class_="title")
                    title = title_tag.text.strip() if title_tag else "未知标题"

                    # 提取链接
                    link_tag = article.find("a", href=True)
                    link = link_tag["href"] if link_tag else ""

                    # 提取提交时间
                    date_text = article.find("p", class_="is-size-7").text
                    date_str = re.search(
                        r"Submitted (\d+ \w+,? \d+)", date_text)
                    if date_str:
                        date_str = date_str.group(1).replace(',', '')
                        date_obj = datetime.datetime.strptime(
                            date_str, "%d %B %Y").date()
                        if (today - date_obj) <= last_days:
                            results.append((title, link))
                            if len(results) >= max_results:
                                return results
                except Exception:
                    continue

        except Exception as e:
            print(f"搜索请求错误: {str(e)}")
            continue

    return results


def translate_text(text):
    """
    使用 OpenAI 翻译文本，将英文学术摘要翻译为中文
    """
    prompt = f"将以下英文学术摘要准确提取核心内容，输出为中文，保持专业术语不变，以'本文'开头：\n\n{text}"
    try:
        response = client.chat.completions.create(
            model=config.get('OpenAI', 'CHATGPT_MODEL',
                             fallback="gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"翻译失败: {str(e)}"


@app.route('/api/arxiv', methods=['GET'])
def arxiv_api():
    """
    接口说明：
    通过 GET 请求传入以下参数：
      - query：搜索关键词（必传），支持 arXiv 高级语法
      - max_results：最多返回的论文数量（默认 3）
      - days：筛选最近 N 天内提交的论文（默认 30）
      - page_num：搜索页数（每页 50 篇，默认 1）
      
    返回 JSON 格式结果，每个论文包含：
      - url：论文链接
      - content：论文主要内容，包括标题、作者、年份以及提取出后的摘要
    """
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "缺少 query 参数"}), 400
    max_results = int(request.args.get('max_results', 3))
    days = int(request.args.get('days', 30))
    page_num = int(request.args.get('page_num', 1))

    try:
        papers = search_arxiv(
            query=query, max_results=max_results, days=days, page_num=page_num)
        if not papers:
            print("未找到符合条件的论文")
            return jsonify({"message": "未找到符合条件的论文"}), 404

        result_list = []
        for title, link in papers:
            # 获取论文详细信息
            paper_info = get_arxiv_info(link)
            abstract = paper_info.get('abstract', '无摘要内容')
            translated = translate_text(abstract)
            # 整合论文主要内容：标题、作者、年份和翻译后的摘要
            content = {
                "title": paper_info.get('title', title),
                "authors": paper_info.get('authors', ''),
                "year": paper_info.get('year', ''),
                "abstract": translated
            }
            result_list.append({
                "url": link,
                "content": content
            })

        return jsonify({"papers": result_list})

    except Exception as e:
        print(f"搜索论文时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=8002, debug=True)

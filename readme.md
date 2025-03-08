# 这是arxiv文献搜索问答项目
## 文件和使用说明
## arxiv_search_trans.py
这是arxiv文献搜索和提取主要内容的接口，使用说明
```bash
curl -G http://localhost:8002/api/arxiv \
    --data-urlencode 'query="llm agent"' \
    --data-urlencode 'max_results=3' \
    --data-urlencode 'days=30' \
    --data-urlencode 'page_num=1'
```
其中 query是关键词请求，max_results是返回最多文献数量，days是最近多少天的论文，page_num是多少页数的论文


## pdf_translate_polish.py
这是上传pdf进行翻译和润色的接口，使用说明：

翻译接口：
```bash
curl -X POST http://localhost:8003/translate \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "test_pdf/pdf-test.pdf",
    "bilingual": false
  }'
```

润色接口：
```bash
curl -X POST http://localhost:8003/polish \
-H "Content-Type: application/json" \
-d '{
    "pdf_path": "test_pdf/pdf-test.pdf",
    "target_lang": "中文", //或“英语”
    "bilingual": true
}'
```

总结接口：
```bash
curl -X POST http://localhost:8003/summarize \
-H "Content-Type: application/json" \
-d '{
    "pdf_path": "test_pdf/pdf-test.pdf",
    "target_lang": "中文"
}'
```
其中bilingual控制是否为双语翻译

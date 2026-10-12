# RAGzhiyan

一个基于 FastAPI + pgvector + Ollama 的 RAG 项目骨架，面向考研资料问答。

## 技术栈

- FastAPI：HTTP 接口
- PostgreSQL + pgvector：文档与向量存储、余弦相似度检索
- Ollama：`bge-m3` 向量化，`qwen2.5:7b` 对话生成
- pypdf：PDF 文本解析

## 项目结构

```text
app/
  main.py              # 应用入口与生命周期
  config.py            # 环境配置
  database.py          # 异步引擎、vector 扩展与建表
  models.py            # documents / chunks 表结构
  schemas.py           # 请求与响应模型
  routers/
    documents.py       # 上传 PDF 接口
    ask.py             # /ask 问答接口
  services/
    ollama.py          # Ollama embedding / chat 客户端
    pdf.py             # PDF 文本提取
    chunking.py        # 文本切分
```

## 准备工作

1. 启动 pgvector 容器（如果还没有的话）：

```bash
docker run -d --name kaoyan-pg \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=password123 \
  -e POSTGRES_DB=kaoyan_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

2. 拉取 Ollama 模型：

```bash
ollama pull bge-m3
ollama pull qwen2.5:7b
```

3. 安装 Python 依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

4. 按需复制配置：

```bash
copy .env.example .env
```

默认配置已经指向本地的 5432 端口 PostgreSQL 和 11434 端口 Ollama。

## 启动

```bash
uvicorn app.main:app --reload
```

接口文档：http://127.0.0.1:8000/docs

## 接口示例

上传 PDF：

```bash
curl -F "file=@./考研资料.pdf" http://127.0.0.1:8000/documents/upload
```

提问：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"什么是马克思主义中国化？\", \"top_k\": 4}"
```

`/ask` 会返回模型回答和引用出处（文档名、页码、片段、相似度）。

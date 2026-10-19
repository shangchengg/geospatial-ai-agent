# GeoSpatial AI Agent

### 基于本地大语言模型、RAG 与 GIS Tool Calling 的地理空间智能分析系统

**LLM-powered Geospatial Research Assistant with RAG, GIS Tools and 3D WebGIS Integration**

本项目基于 **Python、Qwen3.5、Ollama、LangChain、LangGraph、Chroma、GeoPandas 和 FastAPI**，开发一个能够理解自然语言、检索遥感研究文档并执行 GIS 数据分析的地理空间 AI Agent。

系统将 **检索增强生成（Retrieval-Augmented Generation, RAG）** 与 **GIS 工具调用（GIS Tool Calling）** 相结合，使大语言模型不仅能够回答研究报告中的专业问题，还能够通过 Python 工具查询真实空间数据，并返回结构化的地理空间分析结果。

项目已与独立开发的 **CesiumJS 三维森林变化监测 WebGIS** 实现联动，支持根据自然语言查询结果自动定位和高亮森林样地。

当前版本已完成本地 Docker 容器化，并通过 **Cloudflare Tunnel + Vercel** 实现临时 HTTPS 公网演示。

---

## 1. 项目概述（Project Overview）

传统 GIS 分析通常要求使用者熟悉 GIS 软件、空间数据结构或 Python 编程。

本项目探索通过大语言模型降低地理空间数据分析的使用门槛，使用户能够直接通过自然语言完成：

- 遥感与 GIS 研究文档问答
- 森林生物量变化查询
- 土地覆盖转换属性分析
- 多工具协同调用
- 研究报告与 GIS 数据联合分析
- 空间查询结果结构化输出
- Cesium 三维地图联动与自动高亮

### 研究背景

本项目使用加拿大 British Columbia 的 **Mudge Island 和 Link Island** 森林变化研究成果进行开发与验证。

原始研究涉及：

- 2019–2024 年森林地上生物量（AGB）变化
- LiDAR 森林冠层结构指标
- PlanetScope 多时相遥感影像
- 土地覆盖分类与转换分析
- 森林样地尺度的生物量建模
- 土地覆盖变化与生物量变化关系

AI Agent 在此基础上整合研究报告和 GIS 数据，实现自然语言驱动的研究知识检索与空间数据查询。

---

## 2. 系统架构（System Architecture）

系统采用 LLM Agent + Tool Calling 架构，将模型推理与实际 GIS 数据计算分离。

```text
                Natural Language Question
                           |
                           v
                  Qwen3.5 + Ollama
                           |
                           v
                    LangGraph Agent
                           |
                    Tool Selection
                           |
              +------------+------------+
              |                         |
              v                         v
          RAG Tool                  GIS Tools
              |                         |
       LangChain + Chroma            GeoPandas
              |                         |
       PDF Research Report       GeoJSON / GeoPackage
              |                         |
       Research Evidence        Spatial Data Query
              |                         |
              +------------+------------+
                           |
                           v
                    Agent Response
                           |
                           v
                    FastAPI /chat
                           |
                           v
                  Structured JSON
                           |
                           v
                   CesiumJS WebGIS
                           |
                           v
             Forest Stand Highlighting
```

### 核心设计

**LLM 负责：**

- 理解自然语言问题
- 判断需要调用的工具
- 组织多工具调用
- 整合研究解释与 GIS 分析结果
- 生成自然语言回答

**Python GIS 工具负责：**

- 读取空间数据
- 执行属性筛选
- 计算统计结果
- 返回真实森林样地 ID
- 输出结构化数据

这种设计避免让 LLM 直接猜测 GIS 数值，提高空间查询的可验证性。

---

## 3. 技术栈（Technology Stack）

| 模块 | 技术 | 用途 |
|---|---|---|
| 编程语言 | Python | 后端开发与空间数据处理 |
| 大语言模型 | Qwen3.5:4b | 自然语言理解与回答生成 |
| 模型运行 | Ollama | 本地 LLM 推理服务 |
| Agent 框架 | LangGraph | Agent 状态管理与工具调用 |
| LLM 集成 | LangChain | 模型与工具接口封装 |
| Embedding | Qwen3-Embedding:0.6b | 文档向量化 |
| 向量数据库 | Chroma | 文档向量存储与检索 |
| GIS 数据处理 | GeoPandas | 矢量数据读取、属性关联与查询 |
| GIS 文件格式 | GeoJSON / GeoPackage | 森林空间数据 |
| Web API | FastAPI | HTTP 接口与结构化输出 |
| 数据验证 | Pydantic | API 请求与响应结构 |
| 容器化 | Docker / Docker Compose | 服务构建与管理 |
| 公网连接 | Cloudflare Tunnel | 临时 HTTPS 访问 |
| 前端集成 | CesiumJS + Vite | 三维 WebGIS 交互 |
| 前端托管 | Vercel | WebGIS 静态页面部署 |

---

## 4. 核心功能（Core Features）

### 4.1 RAG 研究文档问答

使用 LangChain、Qwen Embedding 和 Chroma 构建研究报告知识检索系统。

基本流程：

```text
Research PDF
     |
     v
PDF Parsing
     |
     v
Text Chunking
     |
     v
Qwen Embedding
     |
     v
Chroma Vector Database
     |
     v
Similarity Retrieval
     |
     v
LLM Answer Generation
```

系统可回答：

- 研究使用了哪些 LiDAR 指标？
- AGB 回归模型采用了什么预测变量？
- 研究的土地覆盖分类方法是什么？
- 2019 年和 2024 年的分类精度分别是多少？
- 为什么稳定森林仍可能发生生物量变化？

主要功能：

- PDF 文本提取
- 文本分块与向量化
- Top-K 相似度检索
- 中英文跨语言问答
- 文档来源与页码引用
- 证据不足时的回答限制

当前使用：

```python
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3.5:4b"
TOP_K = 4
```

模型和检索参数可根据后续评估结果调整。

### 4.2 GIS Tool Calling

使用 GeoPandas 开发可由 Agent 调用的 GIS 查询工具。

当前实现的工具包括：

| Tool | 功能 |
|---|---|
| `get_biomass_loss()` | 查询生物量变化值小于零的森林样地 |
| `get_stable_forest()` | 查询稳定森林转换比例满足阈值的样地 |
| `search_research_report()` | 检索研究报告内容与证据 |

GIS 工具独立执行数据查询，LLM 不直接编造空间属性结果。

### 4.3 森林生物量变化查询

示例问题：

> 哪些 Stand 的 AGB 下降了？

Agent 自动调用：

```python
get_biomass_loss()
```

在当前研究数据中识别出 2 个生物量下降的样地：

| POLYGON_ID | Biomass Change | Dominant Group |
|---|---:|---|
| 35790310 | -3.045 | Stable_forest |
| 46639286 | -0.708 | Stable_forest |

查询结果来自真实 GeoJSON 与 GeoPackage 属性，而非 LLM 生成的假设数据。

### 4.4 土地覆盖转换分析

示例问题：

> 哪些 Stand 的 Stable Forest 比例超过 80%？

Agent 可调用：

```python
get_stable_forest(min_proportion=0.8)
```

查询结果包含：

- `POLYGON_ID`
- `FEATURE_ID`
- `dom_group`
- `dom_prop`
- `biomass_change`

其中 `dom_prop` 表示样地内主导土地覆盖转换组的比例。

### 4.5 Multi-tool Calling

当问题同时涉及空间数据与研究解释时，Agent 可以调用多个工具。

示例：

> 哪些 Stand 的 AGB 下降了？结合研究报告解释可能原因。

Agent 工作流程：

```text
User Question
      |
      v
LangGraph Agent
      |
      +----> get_biomass_loss()
      |             |
      |             v
      |       GIS Query Results
      |
      +----> search_research_report()
                    |
                    v
              PDF Evidence
                    |
                    v
              Qwen3.5
                    |
                    v
          Integrated Answer
```

模型将查询得到的 Stand ID、AGB 变化信息与研究报告内容结合，并区分观测结果与尚未证实的可能解释。

---

## 5. GIS 数据结构（Spatial Data）

当前系统使用两类主要空间数据。

### 5.1 biomass.geojson

包含 13 个森林样地的生物量属性。

主要字段：

```text
FEATURE_ID
POLYGON_ID
WHOLE_STEM_BIOMASS_PER_HA
biomass_change
geometry
```

数据使用 EPSG:3005 坐标参考系统。

### 5.2 transition_biomass.gpkg

GeoPackage 包含以下图层：

```text
island_boundary
Mudge_Link_Biomass
Mudge_Link_Biomass_2019
stands_transition_group
```

其中 `stands_transition_group` 包含：

```text
FEATURE_ID
POLYGON_ID
dom_group
dom_prop
dom_code
geometry
```

用于研究土地覆盖转换与生物量变化之间的关系。

### 5.3 数据关联

GIS 工具使用 `POLYGON_ID` 关联 Biomass 和 Transition 属性。

系统同时保留 `FEATURE_ID`，用于将 GIS 查询结果映射到 Cesium WebGIS 中的森林样地。

```text
GIS Query
    |
    v
POLYGON_ID
    |
    v
FEATURE_ID
    |
    v
Cesium Entity Matching
    |
    v
Map Highlight
```

开发过程中已验证两个项目的 13 个森林样地具有一致且不重复的 `FEATURE_ID`。

---

## 6. FastAPI 接口（REST API）

使用 FastAPI 将 LangGraph Agent 封装为 HTTP 服务。

### 主要接口

| Endpoint | Method | 功能 |
|---|---|---|
| `/` | GET | API 基础信息 |
| `/health` | GET | 服务基础状态检查 |
| `/chat` | POST | 自然语言问题与 Agent 调用 |
| `/docs` | GET | FastAPI Swagger UI |

### 请求示例

```http
POST /chat
Content-Type: application/json
```

```json
{
  "question": "哪些 Stand 的 AGB 下降了？"
}
```

### 响应结构示例

```json
{
  "answer": "共有2个森林样地的AGB出现下降。",
  "stand_ids": [
    35790310,
    46639286
  ],
  "feature_ids": [
    15482873,
    15483050
  ],
  "sources": [],
  "tool_calls": [
    "get_biomass_loss"
  ]
}
```

其中：

- `answer`：模型生成的自然语言回答
- `stand_ids`：GIS Tool 返回的 POLYGON_ID
- `feature_ids`：用于 WebGIS 联动的 FEATURE_ID
- `sources`：RAG 工具检索到的文档来源
- `tool_calls`：Agent 实际调用的工具名称

`stand_ids`、`feature_ids` 和工具记录由程序从工具执行结果中提取，不依赖模型自由生成这些字段。

---

## 7. Cesium WebGIS 集成

本项目已与独立开发的 Cesium 森林变化监测应用实现联动。

**相关项目：**

[Cesium Forest Change Monitor](https://github.com/)

> 请将以上链接替换为实际的 Cesium GitHub 仓库地址。

主要联动功能：

- Cesium WebGIS 中输入自然语言问题
- 通过 FastAPI 调用 GeoSpatial AI Agent
- 显示研究解释与 GIS 查询结果
- 接收结构化 `feature_ids`
- 自动匹配 Cesium Forest Stand Entity
- 高亮符合查询条件的森林样地
- 保留原有 AGB / Land-cover Transition 图层切换

前端使用：

```javascript
const AI_API_URL =
  import.meta.env.VITE_AGENT_API_URL ||
  "http://127.0.0.1:8000";
```

通过环境变量切换本地 API 与公网 HTTPS API。

---

## 8. 项目目录（Repository Structure）

```text
geospatial-ai-agent/
│
├── README.md
├── .gitignore
├── .dockerignore
├── Dockerfile
├── compose.yaml
│
├── requirements.txt
├── requirements-docker.txt
│
├── ingest.py
├── retrieve.py
├── rag.py
├── agent.py
├── api.py
├── schemas.py
│
├── tools/
│   ├── __init__.py
│   ├── geojson_query.py
│   ├── gis_tools.py
│   ├── rag_search.py
│   ├── inspect_vector.py
│   └── compare_geojson.py
│
├── evaluation/
│   ├── questions.json
│   └── evaluate.py
│
├── data/
│   ├── docs/
│   ├── vector/
│   └── raster/
│
└── chroma_db/
```

其中 `data/` 和 `chroma_db/` 为本地研究数据与持久化数据库目录，默认不提交到公共仓库。

---

## 9. 本地开发（Local Development）

### 9.1 环境要求

建议环境：

- Python 3.12
- Ollama
- Git
- Windows / Linux
- 足够运行 Qwen 模型的内存资源

### 9.2 克隆项目

```bash
git clone <YOUR_REPOSITORY_URL>
cd geospatial-ai-agent
```

### 9.3 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 执行策略阻止激活，可以直接使用：

```powershell
.\.venv\Scripts\python.exe
```

运行 Python 命令。

### 9.4 安装依赖

```bash
python -m pip install -r requirements-docker.txt
```

### 9.5 准备模型

安装 Ollama 后：

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

### 9.6 准备研究数据

由于研究 PDF、Chroma 数据库及 GIS 文件默认不上传，克隆仓库后需要准备对应的数据文件：

```text
data/docs/
    Research_Report.pdf

data/vector/
    biomass.geojson
    transition_biomass.gpkg
```

当前代码中的部分文件路径与图层名称针对本研究数据配置。

使用其他数据时，需要对应调整路径、字段及图层配置。

对于 RAG，还需要使用 `ingest.py` 构建 Chroma 向量数据库，确保使用与查询阶段一致的 Embedding 模型。

### 9.7 启动 Agent

```bash
python agent.py
```

### 9.8 启动 FastAPI

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

打开：

http://127.0.0.1:8000/docs

即可通过 Swagger UI 测试 API。

---

## 10. Docker 容器化部署

本项目支持 Docker Compose 同时管理 FastAPI 和 Ollama。

主要容器：

```text
Docker Compose
    |
    +---- FastAPI Container
    |       |
    |       +---- LangGraph
    |       +---- GeoPandas
    |       +---- Chroma
    |
    +---- Ollama Container
            |
            +---- Qwen3.5:4b
            +---- Qwen Embedding
```

### 10.1 构建镜像

```bash
docker compose build
```

### 10.2 启动容器

```bash
docker compose up -d
```

### 10.3 下载模型

```bash
docker compose exec ollama ollama pull qwen3.5:4b
docker compose exec ollama ollama pull qwen3-embedding:0.6b
```

### 10.4 检查状态

```bash
docker compose ps
```

### 10.5 查看日志

```bash
docker compose logs -f api
```

### 10.6 停止服务

```bash
docker compose down
```

Docker Compose 使用数据卷保存 Ollama 模型，并通过本地挂载提供 Chroma 和 GIS 数据。

请勿在没有备份的情况下随意删除 Docker 数据卷。

---

## 11. 在线演示（Online Demo）

当前系统使用：

**Vercel + Cloudflare Tunnel + Local Docker**

实现临时公网访问。

```text
Public Browser
      |
      v
Vercel Cesium WebGIS
      |
      v
Cloudflare Tunnel (HTTPS)
      |
      v
Local Docker FastAPI
      |
      v
LangGraph Agent
      |
      v
Ollama + Qwen
```

### 部署方式

1. Vercel 托管 CesiumJS 前端。
2. Windows 本地 Docker 运行 FastAPI 与 Ollama。
3. Cloudflare Quick Tunnel 提供临时 HTTPS 公网地址。
4. Vercel 使用 `VITE_AGENT_API_URL` 连接 FastAPI。
5. Agent 返回查询结果并触发 Cesium 地图高亮。

### 重要限制

当前 AI 服务并不是全天候云端托管。

AI 功能正常使用需要：

- 本地电脑保持运行
- Docker Desktop 正常启动
- Ollama 模型服务可用
- Cloudflare Tunnel 保持连接

Cloudflare Quick Tunnel 地址可能在重新启动后变化。

当前公网连接主要用于开发验证和临时演示，不提供持续可用性保证。

---

## 12. RAG 测试与验证

项目已构建小型 RAG 基础测试集，覆盖：

- 事实问答
- 数值检索
- 中英文跨语言检索
- 综合分析
- 负样本问题

当前 Evaluation 包括：

- Keyword Matching
- Evidence Page Retrieval
- 基础 Hit@K 统计
- 人工答案核对

初步测试中，系统能够成功生成回答，并检索到相应研究资料。

需要注意：当前测试集规模有限，Keyword Pass Rate 与页码级 Hit@K 不能直接等同于完整系统准确率。

后续将扩展 Benchmark，并进一步评估：

- Retrieval Accuracy
- Answer Faithfulness
- Citation Accuracy
- Hallucination Rate
- End-to-end Latency

---

## 13. 当前开发状态（Development Status）

### 已完成

- [x] PDF Parsing 与 Text Chunking
- [x] Qwen Embedding
- [x] Chroma Vector Database
- [x] RAG 文档问答
- [x] 中英文跨语言查询
- [x] 基础 RAG Evaluation
- [x] GeoJSON / GeoPackage 数据读取
- [x] 森林生物量变化查询
- [x] Stable Forest 比例筛选
- [x] LangGraph Tool Calling
- [x] RAG + GIS Multi-tool Calling
- [x] FastAPI REST API
- [x] Pydantic Structured Output
- [x] Cesium Stand ID Mapping
- [x] AI 查询结果地图高亮
- [x] Docker Compose 双容器部署
- [x] Cloudflare Tunnel HTTPS 演示
- [x] Vercel WebGIS 前后端联动

### 后续计划

- [ ] Rasterio 栅格分析工具
- [ ] GeoTIFF 区域统计
- [ ] 多轮对话与会话记忆
- [ ] 更完整的 GIS 空间分析能力
- [ ] RAG Benchmark 扩充
- [ ] Tool Calling 可靠性评估
- [ ] API 身份认证与请求限流
- [ ] 模型推理性能优化
- [ ] 固定公网访问地址
- [ ] 独立云端部署
- [ ] CI/CD 自动化测试与部署

---

## 14. 项目技术价值（Technical Highlights）

本项目展示以下技术能力：

**AI / LLM Engineering**

- 本地 LLM 部署与推理
- RAG Pipeline 开发
- Embedding 与 Vector Database
- LangGraph Agent 工作流
- Tool Calling
- Structured Output
- LLM 与工具结果协同

**Geospatial Data Engineering**

- GeoPandas 空间数据处理
- GeoJSON / GeoPackage 数据读取
- GIS 属性关联与过滤
- 坐标参考系统管理
- Stand-level Biomass Analysis
- Land-cover Transition Analysis

**Backend Engineering**

- FastAPI REST API
- Pydantic Schema
- 环境变量配置
- Docker 容器化
- Docker Compose 服务编排
- HTTPS 公网连接

**WebGIS Integration**

- CesiumJS 三维地图
- GIS 查询结果与地图联动
- Feature ID Mapping
- 自动定位与高亮
- 自然语言驱动的空间信息查询

---

## 15. 项目定位与发展方向

GeoSpatial AI Agent 旨在探索：

**从传统 GIS 软件操作转向自然语言驱动的地理空间分析。**

当前版本主要实现：

```text
Research Knowledge
        +
Geospatial Data
        +
Local LLM
        +
Tool Calling
        =
GeoSpatial AI Agent
```

未来计划将现有属性查询能力扩展至栅格分析、复杂空间运算、多轮任务规划和更完善的 WebGIS 智能交互。

---

## 16. 作者（Author）

**Shangcheng (Brian) Li**

GIS | Remote Sensing | LiDAR | Python | WebGIS | GeoAI

---

## Keywords

`GeoAI` `Geospatial AI Agent` `LangGraph` `LangChain`
`Qwen3.5` `Ollama` `RAG` `Chroma`
`GeoPandas` `FastAPI` `Docker` `GIS Tool Calling`
`CesiumJS` `WebGIS` `Remote Sensing` `LiDAR`
`Forest Biomass` `Land-cover Change`
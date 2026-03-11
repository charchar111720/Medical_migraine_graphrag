# 面向中文医疗指南的知识图谱构建
**Medical Knowledge Graph Construction for Chinese Clinical Guidelines based on GraphRAG**

> 基于微软 GraphRAG 框架的二次开发，专为中文医疗指南设计的知识图谱提取与构建方案，当前以偏头痛（Migraine）领域为示例。

---

## 背景 Background

微软开源的 [GraphRAG](https://github.com/microsoft/graphrag) 在通用文档上表现优异，但直接用于医疗指南时存在明显局限：默认实体类型（PERSON、ORGANIZATION 等）与医疗场景严重脱节，LLM 在提取过程中容易自由发挥，生成与诊疗无关的噪声实体，导致知识图谱质量难以满足医疗场景的精准性要求。

本项目针对上述问题对 GraphRAG 进行了系统性改造，构建了一套可快速复用、可扩展至其他疾病领域的中文医疗知识图谱方案。

---

## 核心改进 Key Features

**1. PDF 原生支持（Native PDF Support）**

GraphRAG 官方仅支持 `txt`、`csv`、`json` 三种格式，不支持 `pdf`。本项目通过二次开发，集成 [MinerU](https://github.com/opendatalab/MinerU) 对 PDF 进行解析，可完整保留文档中的表格和图片信息，并配套设计了适配 PDF 结构的自定义切分策略（Custom Chunking Strategy），彻底解除了原生格式限制。只需在 `input` 目录放入 PDF 文件，并配置好 MinerU API Key 即可。

**2. 全流程手动 Prompt 调优（Manual Prompt Tuning）**

放弃 GraphRAG 的系统自带的 Prompt 以及自动 Prompt Tuning，全程手动编写了 **12 个 Prompt**，覆盖索引（Indexing）与查询（Query）全流程。严格定义了实体类型（Entity Types）与关系类型（Relation Types），明确禁止 LLM 自行创造新类型或编造内容，确保知识图谱的可控性与一致性。

**3. 医疗专属实体与关系设计（Domain-specific Entity & Relation Schema）**

针对医疗指南的文档结构，设计了贴合临床场景的实体类型，包括：疾病（Disease）、症状（Symptom）、药物（Drug）、诊断标准（Diagnostic Criteria）、治疗方案（Treatment）、证据等级（Evidence Level）、禁忌症（Contraindication）等。关系类型同样经过专项设计，能够准确捕捉诊疗规则、用药逻辑与循证依据之间的关联。

**4. 易扩展的疾病领域适配（Easy Domain Extension）**

当前实体与关系体系以**偏头痛（Migraine）**为示例场景，但整体框架具备良好的可迁移性。如需扩展至其他疾病领域（如糖尿病、高血压、癫痫等），仅需调整 Prompt 中的具体术语与实体定义，无需修改底层代码，即可快速构建新的专科知识图谱。

---

## 使用方式 Usage

1. 将医疗指南 PDF 文件放入 `input/` 目录；
2. 设置好模型参数，包括 temperature、top_p、presence_penalty、frequency_penalty、batch_size 等；
3. 申请 MinerU API Key；
4. 修改文件类型枚举、分块策略枚举，补充工厂函数的 pdf 处理逻辑，扩展策略加载逻辑；
5. 按需修改 prompts 中的实体与关系定义（如扩展至其他疾病）；
6. 常规运行 GraphRAG 索引与查询流程：
  ```bash
  # 索引示例（--root 指向根目录，而非 input）
  python -m graphrag index --root ./
  
  # 提问示例
  python -m graphrag query --root ./ --method local --query "问题？"
  python -m graphrag query --root ./ --method global --query "问题？"
   ```

具体配置参数与运行命令请参考项目文档。


---

## 适用场景 Use Cases

- 中文临床指南、诊疗规范的结构化知识抽取
- 医疗问答系统（Medical QA）的知识库构建
- 疾病知识图谱（Disease Knowledge Graph）的自动化生成
- 基于证据等级的诊疗推理辅助

---

## 依赖 Dependencies

- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [MinerU](https://github.com/opendatalab/MinerU)（PDF 解析）

---

## 致谢 Acknowledgements

本项目基于 Microsoft GraphRAG 进行二次开发，感谢 GraphRAG 团队的开源贡献。

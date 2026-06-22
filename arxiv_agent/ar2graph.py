import arxiv
import json
import os
import openai
from pyvis.network import Network
from typing import List, Dict
import time

# =================配置区域=================
# 请在此处填写你的 OpenAI API Key，或者设置环境变量 OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "2105206ed05e3406ecbf1b873520860f.Vo6ncazYziB69s2u")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

# 配置参数
SEARCH_QUERY = "Agentic Reinforcement Learning"  # 搜索关键词
MAX_RESULTS = 5             # 爬取论文数量
OUTPUT_JSON = "articles_all_01.json"
OUTPUT_HTML = "knowledge_graph_01.html"
# =========================================

class Ar2Graph:
    def __init__(self, api_key: str, base_url: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.articles = []
        self.triplets = []

    def fetch_arxiv_papers(self, query: str, max_results: int = 10):
        """
        第一步：爬取 ArXiv 论文
        """
        print(f"🚀 正在 ArXiv 上搜索关键词: '{query}'...")
        
        # 使用 arxiv 库构建搜索
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = []
        client = arxiv.Client(delay_seconds=3.0, num_retries=3)
        
        for result in client.results(search):
            paper = {
                "id": result.entry_id,
                "title": result.title,
                "summary": result.summary.replace("\n", " "),
                "published": result.published.strftime("%Y-%m-%d"),
                "url": result.pdf_url,
                "authors": [a.name for a in result.authors]
            }
            results.append(paper)
            print(f"   - 发现论文: {paper['title'][:50]}...")

        self.articles = results
        
        # 保存原始数据
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        print(f"✅ 已保存 {len(results)} 篇论文到 {OUTPUT_JSON}")

    def extract_triplets(self):
        """
        第二步：使用 OpenAI 抽取知识三元组 (Head, Relation, Tail)
        """
        print("🧠 正在调用 AI 抽取知识三元组 (这可能需要一点时间)...")
        
        system_prompt = """
        你是一个专业的知识图谱构建专家。请从给定的论文摘要中提取核心知识三元组。
        
        规则：
        1. 输出必须是纯 JSON 格式的列表。
        2. 列表中的每个元素是一个对象，包含三个键：'head' (实体1), 'relation' (关系), 'tail' (实体2)。
        3. 实体应该是具体的概念、技术、模型名称或指标。
        4. 关系应该是动词或动词短语，简短有力。
        5. 尽量提取 3-5 个最核心的三元组。
        6. 保持语言为英文（因为大部分术语是英文）。
        
        示例输出：
        [
            {"head": "Transformer", "relation": "uses", "tail": "Self-Attention"},
            {"head": "BERT", "relation": "improves", "tail": "NLP Tasks"}
        ]
        """

        for i, article in enumerate(self.articles):
            print(f"   [{i+1}/{len(self.articles)}] 分析: {article['title'][:30]}...")
            
            user_prompt = f"Title: {article['title']}\nAbstract: {article['summary']}"
            
            try:
                response = self.client.chat.completions.create(
                    model="glm-4.6", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"} # 强制 JSON 模式
                )
                
                content = response.choices[0].message.content
                # 解析 JSON
                data = json.loads(content)
                
                # 兼容不同的返回结构（有时 AI 会把列表包在一个 key 里）
                if isinstance(data, dict):
                    # 尝试寻找可能是列表的 value
                    for key, val in data.items():
                        if isinstance(val, list):
                            extracted_list = val
                            break
                    else:
                        extracted_list = []
                elif isinstance(data, list):
                    extracted_list = data
                else:
                    extracted_list = []

                # 添加源论文信息以便可视化溯源
                for item in extracted_list:
                    item['source_title'] = article['title']
                    item['source_url'] = article['url']
                    self.triplets.append(item)
                    
            except Exception as e:
                print(f"   ❌ 抽取失败: {e}")
                time.sleep(1) # 避免速率限制

        print(f"✅ 抽取完成，共获得 {len(self.triplets)} 个三元组。")

    def generate_graph(self):
        """
        第三步：生成交互式知识图谱 HTML
        """
        print(f"🔍 正在生成知识图谱: {OUTPUT_HTML}...")
        
        # 初始化 PyVis 网络
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
        
        # 使用字典来去重节点，防止重复添加
        # 结构: node_label -> {type: 'concept' or 'paper', url: ''}
        nodes = {}
        
        for t in self.triplets:
            head = t.get('head', 'Unknown')
            tail = t.get('tail', 'Unknown')
            relation = t.get('relation', 'related to')
            source_title = t.get('source_title')
            source_url = t.get('source_url')

            # 添加三元组中的实体节点
            if head not in nodes:
                net.add_node(head, label=head, title=head, color="#97C2FC") # 浅蓝
                nodes[head] = True
            
            if tail not in nodes:
                net.add_node(tail, label=tail, title=tail, color="#FFFF00") # 黄色
                nodes[tail] = True

            # 添加边 (实体 -> 实体)
            net.add_edge(head, tail, title=relation, label=relation, color="#FFFFFF")

            # 可将源论文也作为一个中心节点连接到 head (展示来源)
            # 这里为了图谱简洁，暂时不把论文标题作为节点，而是放在边的 title (hover 显示) 或者数据中
            # 如果你想看论文节点，取消下面注释
            
            paper_node_id = f"Paper: {source_title[:20]}..."
            if paper_node_id not in nodes:
                net.add_node(paper_node_id, label="📄 " + source_title[:20]+"...", title=source_title, color="#FB7E81", shape="box") # 红色
                nodes[paper_node_id] = True
            net.add_edge(paper_node_id, head, color="rgba(255,255,255,0.3)")

        # 设置物理引擎效果 (让图谱散开)
        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 16
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": false
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            }
          }
        }
        """)

        # 保存
        net.save_graph(OUTPUT_HTML)
        print(f"✨ 任务完成！请在浏览器中打开 {OUTPUT_HTML} 查看结果。")

if __name__ == "__main__":
    # 检查 Key
    if "your-api-key-here" in OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: 请在代码中设置 OPENAI_API_KEY 或设置环境变量。")
    else:
        app = Ar2Graph(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        
        # 1. 爬取
        app.fetch_arxiv_papers(query=SEARCH_QUERY, max_results=MAX_RESULTS)
        
        # 2. 抽取
        if app.articles:
            app.extract_triplets()
            
        # 3. 绘图
        if app.triplets:
            app.generate_graph()
        else:
            print("⚠️ 没有提取到三元组，无法生成图谱。")
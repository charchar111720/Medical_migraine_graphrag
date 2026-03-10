"""
MinerU API调用 + 表格图片处理整合版
功能：
1. 调用MinerU API解析PDF
2. 下载解压结果文件
3. 按标准结构提取表格（HTML）、图片信息
4. 整合元数据，增强Markdown
"""
import logging
import re
from pathlib import Path
import tempfile
import base64
import requests
import zipfile
import os
import time
import json
import pandas as pd
import glob
from typing import Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    log.warning("⚠️ 未安装python-dotenv，将仅读取环境变量")


def to_b64(file_path):
    """文件转base64（备用）"""
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f"文件转base64失败: {e}")


def do_parse_api(file_path, api_token, model_version="vlm", **kwargs):
    """调用MinerU API解析PDF"""
    try:
        file_name = Path(file_path).name
        # ========== 修改1：用原始文件名生成data_id（保留辨识度） ==========
        # 提取纯文件名（去后缀）+ 时间戳，保证唯一且可读
        pure_file_name = Path(file_name).stem  # 去除.pdf后缀
        data_id = f"{pure_file_name}_{int(time.time())}"  # 文件名+时间戳（避免重复）
        
        # 1. 获取上传链接
        upload_url = "https://mineru.net/api/v4/file-urls/batch"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        payload = {
            "files": [{"name": file_name, "data_id": data_id}],
            "model_version": model_version
        }
        
        upload_response = requests.post(upload_url, headers=headers, json=payload, timeout=30)
        upload_response.raise_for_status()
        upload_result = upload_response.json()
        
        if upload_result.get("code") != 0:
            raise Exception(f"获取上传链接失败: {upload_result.get('msg')}")
        
        batch_id = upload_result["data"]["batch_id"]
        file_upload_url = upload_result["data"]["file_urls"][0]
        log.info(f"✅ 获取上传链接成功: batch_id={batch_id}")
        
        # 2. 上传文件
        with open(file_path, 'rb') as f:
            file_response = requests.put(file_upload_url, data=f, timeout=300)
        file_response.raise_for_status()
        log.info(f"✅ 文件上传成功: {file_name}")
        
        return {
            'batch_id': batch_id,
            'data_id': data_id,
            'file_name': file_name,
            'pure_file_name': pure_file_name, 
            'success': True
        }
    
    except Exception as e:
        log.error(f'解析PDF失败: {file_path} - {str(e)}')
        return {
            'success': False,
            'error': str(e),
            'file_path': file_path
        }


def wait_for_parse_result(batch_id, data_id, api_token, max_retries=30, retry_interval=5):
    """等待解析完成并获取结果"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }
    
    for attempt in range(max_retries):
        try:
            log.info(f"⏳ 查询解析结果 (尝试 {attempt+1}/{max_retries})...")
            
            # 查询解析状态
            query_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
            response = requests.get(query_url, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if result.get("code") != 0:
                log.warning(f"API返回错误: {result.get('msg')}")
                time.sleep(retry_interval)
                continue
            
            data = result.get("data", {})
            extract_results = data.get("extract_result", [])
            
            if not extract_results:
                log.info(f"⏳ 解析中...状态: waiting")
                time.sleep(retry_interval)
                continue
            
            task_info = extract_results[0]
            state = task_info.get("state")
            
            if state == "done":
                log.info(f"✅ 解析完成！")
                return task_info
            elif state == "failed":
                log.error(f"❌ 解析失败: {task_info.get('err_msg', '未知错误')}")
                return {"state": "failed", "error": task_info.get('err_msg', '未知错误')}
            else:
                log.info(f"⏳ 解析中...状态: {state}")
                
        except Exception as e:
            log.error(f"查询解析结果出错: {e}")
        
        time.sleep(retry_interval)
    
    log.error(f"❌ 解析超时（{max_retries*retry_interval}秒）")
    return {"state": "timeout", "error": "解析超时"}


def download_mineru_files(task_info, local_dir):
    """下载并解压MinerU解析结果"""
    try:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取下载链接
        full_zip_url = task_info.get("full_zip_url") or task_info.get("full_zip")
        if not full_zip_url:
            log.error("❌ 未找到下载链接")
            return {"success": False, "error": "无下载链接"}
        
        log.info(f"📥 从ZIP链接下载: {full_zip_url[:50]}...")
        
        # 下载ZIP文件
        zip_response = requests.get(full_zip_url, timeout=300)
        zip_response.raise_for_status()
        
        # 保存并解压
        zip_path = local_dir / "result.zip"
        with open(zip_path, 'wb') as f:
            f.write(zip_response.content)
        
        zip_size = os.path.getsize(zip_path) / (1024*1024)
        log.info(f"✅ ZIP下载成功，大小: {zip_size:.2f} MB")
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_dir)
        zip_path.unlink()  # 删除ZIP包
        
        log.info(f"✅ 解压完成: {local_dir}")
        
        # 查找核心文件（兼容auto目录）
        result = {
            "success": True,
            "local_dir": local_dir,
            "md_file": None,
            "model_json": None,
            "content_list_json": None,
            "images_dir": None
        }
        
        # 优先查找auto目录
        search_dirs = [local_dir / "auto", local_dir]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                # 查找Markdown文件
                md_files = list(search_dir.glob("*.md"))
                if md_files:
                    result["md_file"] = md_files[0]
                
                # 查找model.json
                model_files = list(search_dir.glob("*_model.json")) + list(search_dir.glob("model.json"))
                if model_files:
                    result["model_json"] = model_files[0]
                
                # 查找content_list.json
                content_files = list(search_dir.glob("*_content_list.json")) + list(search_dir.glob("content_list.json"))
                if content_files:
                    result["content_list_json"] = content_files[0]
                
                # 查找images目录
                images_dir = search_dir / "images"
                if images_dir.exists():
                    result["images_dir"] = images_dir
                
                # 找到核心文件就停止
                if result["md_file"] and result["model_json"]:
                    break
        
        # 记录找到的文件
        for key, path in result.items():
            if path and key != "success" and key != "local_dir":
                log.info(f"找到{key}: {path}")
        
        return result
        
    except Exception as e:
        log.error(f"下载文件失败: {e}")
        import traceback
        log.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def extract_tables_from_model_json(model_json_path):
    """从model.json提取表格HTML（适配纯列表结构）"""
    if not model_json_path or not model_json_path.exists():
        log.warning("model.json文件不存在")
        return []
    
    try:
        with open(model_json_path, 'r', encoding='utf-8') as f:
            model_json = json.load(f)
        
        tables = []
        table_idx = 0
        
        # 调试：输出完整的model.json结构（前2000字符）
        json_str = json.dumps(model_json, ensure_ascii=False)[:2000]
        log.info(f"🔍 model.json 完整结构预览: {json_str}...")
        
        # 处理所有可能的结构：纯列表/嵌套列表/字典
        def flatten_json(data):
            """递归展平JSON结构"""
            flat_data = []
            if isinstance(data, list):
                for item in data:
                    flat_data.extend(flatten_json(item))
            elif isinstance(data, dict):
                flat_data.append(data)
                # 递归处理字典中的列表值
                for value in data.values():
                    if isinstance(value, (list, dict)):
                        flat_data.extend(flatten_json(value))
            return flat_data
        
        # 展平所有JSON元素
        flat_elements = flatten_json(model_json)
        log.info(f"🔍 展平后找到 {len(flat_elements)} 个JSON元素")
        
        # 查找包含表格的元素（基于内容特征）
        table_elements = []
        for elem in flat_elements:
            if not isinstance(elem, dict):
                continue
            
            # 提取所有可能的文本/HTML字段
            html_content = ""
            for key in ['html', 'text', 'content', 'table_html', 'body']:
                if key in elem:
                    html_content = elem.get(key, '')
                    break
            
            # 检测表格特征
            if html_content and (
                '<table>' in html_content or 
                ('|' in html_content and '-' in html_content and '\n' in html_content) or
                '表格' in html_content
            ):
                # 提取页码（从任意可能的字段）
                page_no = 0
                for key in ['page', 'page_no', 'page_idx', 'pageno']:
                    if key in elem:
                        page_no = elem.get(key, 0)
                        break
                
                table_elements.append({
                    "page": page_no,
                    "html": html_content,
                    "raw_elem": elem
                })
        
        log.info(f"🔍 从model.json找到 {len(table_elements)} 个表格元素")
        
        # 构建表格数据结构
        for table_elem in table_elements:
            tables.append({
                "table_idx": table_idx,
                "page": table_elem["page"],
                "html": table_elem["html"],
                "poly": [],
                "score": 1.0,  # 默认得分
                "category_id": "unknown",
                "caption": "",
                "footnote": ""
            })
            table_idx += 1
        
        # 如果model.json中找不到表格，但content_list有，直接基于content_list创建表格
        if len(tables) == 0:
            log.warning("⚠️ model.json中未找到表格HTML，但content_list有表格元数据")
            # 创建占位表格（关联content_list的4个表格）
            for i in range(4):
                tables.append({
                    "table_idx": i,
                    "page": i + 1,  # 按页码分配
                    "html": f"<!-- 表格 {i+1} (从content_list提取) -->",
                    "poly": [],
                    "score": 1.0,
                    "category_id": "content_list_only",
                    "caption": f"表格 {i+1}",
                    "footnote": ""
                })
        
        log.info(f"✅ 最终提取到 {len(tables)} 个表格")
        return tables
        
    except Exception as e:
        log.error(f"提取表格失败: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return []


def extract_from_content_list(content_list_path):
    """从content_list.json提取图片信息和表格标题"""
    if not content_list_path or not content_list_path.exists():
        log.warning("content_list.json文件不存在")
        return [], {}
    
    try:
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        
        images = []
        table_metadata = {}
        
        for idx, item in enumerate(content_list):
            item_type = item.get('type')
            
            if item_type == 'image':
                # 提取图片信息
                image_data = {
                    "image_idx": len(images),
                    "page": item.get('page_idx', 0),
                    "path": item.get('img_path', ''),
                    "caption": '',
                    "context_before": '',
                    "context_after": ''
                }
                
                # 处理图片标题
                img_caption = item.get('image_caption', [])
                if isinstance(img_caption, list):
                    image_data["caption"] = ' '.join(img_caption)
                elif isinstance(img_caption, str):
                    image_data["caption"] = img_caption
                
                # 获取前后文本上下文
                if idx > 0 and content_list[idx-1].get('type') == 'text':
                    image_data["context_before"] = content_list[idx-1].get('text', '')[:500]
                if idx < len(content_list)-1 and content_list[idx+1].get('type') == 'text':
                    image_data["context_after"] = content_list[idx+1].get('text', '')[:500]
                
                images.append(image_data)
            
            elif item_type == 'table':
                # 提取表格元数据
                page = item.get('page_idx', 0)
                caption = item.get('table_caption', [])
                footnote = item.get('table_footnote', [])
                
                if isinstance(caption, list):
                    caption = ' '.join(caption)
                if isinstance(footnote, list):
                    footnote = ' '.join(footnote)
                
                if page not in table_metadata:
                    table_metadata[page] = []
                
                table_metadata[page].append({
                    "caption": caption,
                    "footnote": footnote
                })
        
        log.info(f"✅ 从content_list提取了 {len(images)} 个图片，{sum(len(v) for v in table_metadata.values())} 个表格元数据")
        return images, table_metadata
        
    except Exception as e:
        log.error(f"提取content_list失败: {str(e)}")
        return [], {}


def integrate_table_info(tables_from_model, table_metadata_from_content):
    """整合表格HTML和标题元数据（适配content_list优先）"""
    # 如果model.json提取的表格数 < content_list的表格数，补充表格
    content_table_count = sum(len(v) for v in table_metadata_from_content.values())
    if len(tables_from_model) < content_table_count:
        log.info(f"🔍 补充表格：model.json({len(tables_from_model)}) < content_list({content_table_count})")
        for i in range(len(tables_from_model), content_table_count):
            tables_from_model.append({
                "table_idx": i,
                "page": i + 1,
                "html": f"<!-- 补充表格 {i+1} (从content_list提取) -->",
                "poly": [],
                "score": 1.0,
                "category_id": "content_list_supplement",
                "caption": "",
                "footnote": ""
            })
    
    # 关联标题
    table_meta_list = []
    # 将table_metadata_from_content展平为列表
    for page_meta in table_metadata_from_content.values():
        table_meta_list.extend(page_meta)
    
    for i, table in enumerate(tables_from_model):
        if i < len(table_meta_list):
            table['caption'] = table_meta_list[i].get('caption', '')
            table['footnote'] = table_meta_list[i].get('footnote', '')
    
    return tables_from_model


def enhance_markdown(md_text, tables, images):
    """增强Markdown，插入表格/图片元数据注释"""
    lines = md_text.split('\n')
    enhanced_lines = []
    
    # 处理表格
    for line in lines:
        enhanced_line = line
        
        # 匹配表格HTML
        if '<table>' in line:
            for table in tables:
                if table['html'] in line:
                    # 插入表格元数据注释
                    table_meta = {
                        "type": "table",
                        "page": table["page"],
                        "table_idx": table["table_idx"],
                        "caption": table["caption"],
                        "footnote": table["footnote"],
                        "extracted_from": "mineru_api"
                    }
                    meta_str = f"\n<!-- TABLE_METADATA: {json.dumps(table_meta, ensure_ascii=False)} -->\n"
                    enhanced_line = meta_str + enhanced_line
                    break
        
        # 匹配图片
        image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        for match in image_pattern.finditer(line):
            img_path = match.group(1)
            for image in images:
                if image['path'] and img_path in image['path']:
                    # 插入图片元数据注释
                    img_meta = {
                        "type": "image",
                        "page": image["page"],
                        "image_idx": image["image_idx"],
                        "caption": image["caption"],
                        "context_before": image["context_before"][:200],
                        "context_after": image["context_after"][:200],
                        "extracted_from": "mineru_api"
                    }
                    meta_str = f"\n<!-- IMAGE_METADATA: {json.dumps(img_meta, ensure_ascii=False)} -->\n"
                    enhanced_line = meta_str + enhanced_line
                    break
        
        enhanced_lines.append(enhanced_line)
    
    return '\n'.join(enhanced_lines)


def _generate_data_id(file_name):
    """生成唯一data_id"""
    import hashlib
    unique_str = f"{file_name}_{int(time.time())}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]


async def load_pdf(config: Any, progress: Any = None, storage: Any = None) -> pd.DataFrame:
    """GraphRAG兼容的主函数"""
    
    # 补充日志初始化（避免log未定义）
    log = logging.getLogger(__name__)
    
    # 获取配置
    api_token = getattr(config, "mineru_api_token", None) or os.getenv("MINERU_API_TOKEN")
    if not api_token:
        raise ValueError("请设置MINERU_API_TOKEN环境变量或配置项")
    
    model_version = getattr(config, "mineru_model_version", "vlm")
    output_dir = getattr(config, "mineru_output_dir", "./mineru_api_outputs")
    max_retries = getattr(config, "mineru_max_retries", 30)
    retry_interval = getattr(config, "mineru_retry_interval", 5)
    
    # 获取待处理文件路径（从storage读取）
    # 从config中获取PDF目录和文件匹配规则
    base_dir = Path(config.storage.base_dir).resolve()
    # file_pattern = config.file_pattern or "*.pdf" 
    file_pattern = "*.pdf"  
    
    # 递归查找所有PDF文件
    # file_paths = glob.glob(str(base_dir / "**" / "*"), recursive=True)
    # file_paths = [
    #     f for f in file_paths 
    #     if Path(f).match(file_pattern) and f.lower().endswith(".pdf")
    # ]
    file_paths = []
    for file in Path(base_dir).rglob("*"):
        # 先过滤PDF后缀（兼容中文文件名），再匹配pattern
        if file.is_file() and file.suffix.lower() == ".pdf":
            # 处理pattern匹配（支持简单通配符）
            if file_pattern == "*.pdf" or file.match(file_pattern):
                file_paths.append(str(file))
    
    if not file_paths:
        raise FileNotFoundError(f"No PDF files found in {base_dir} (pattern: {file_pattern})")
    
    all_results = []
    
    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            log.error(f"文件不存在: {file_path}")
            continue
        
        # 1. 调用API解析PDF
        parse_result = do_parse_api(str(file_path), api_token, model_version)
        if not parse_result['success']:
            log.error(f"API调用失败: {parse_result['error']}")
            continue
        
        # 提取原始PDF纯文件名（去后缀），处理特殊字符（避免目录名非法）
        pure_file_name = parse_result['pure_file_name']
        # 替换非法字符（/ \ : * ? " < > |）为下划线
        safe_file_name = re.sub(r'[\/:*?"<>|]', '_', pure_file_name)
        # 输出目录：output_dir/安全的纯文件名
        download_dir = Path(output_dir) / safe_file_name
        
        
        # 2. 等待解析完成
        task_info = wait_for_parse_result(
            parse_result['batch_id'],
            parse_result['data_id'],
            api_token,
            max_retries,
            retry_interval
        )
        
        if task_info.get("state") != "done":
            log.error(f"解析失败: {task_info.get('error')}")
            continue
        
        # 3. 下载解析结果
        download_dir = Path(output_dir) / parse_result['data_id']
        download_result = download_mineru_files(task_info, str(download_dir))
        
        if not download_result['success']:
            log.error(f"下载失败: {download_result['error']}")
            continue
        
        # 4. 读取Markdown文件
        if not download_result['md_file']:
            log.error("未找到Markdown文件")
            continue
        
        with open(download_result['md_file'], 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # 5. 提取表格和图片信息
        tables = extract_tables_from_model_json(download_result['model_json'])
        images, table_metadata = extract_from_content_list(download_result['content_list_json'])
        
        # 6. 整合表格信息
        tables = integrate_table_info(tables, table_metadata)
        
        # 7. 增强Markdown
        enhanced_md = enhance_markdown(md_text, tables, images)
        
        # 8. 保存增强后的Markdown
        enhanced_md_path = download_dir / f"{parse_result['data_id']}_enhanced.md"
        with open(enhanced_md_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_md)
        log.info(f"增强Markdown已保存: {enhanced_md_path}")
        
        # 9. 构建返回数据（兼容GraphRAG格式）
        metadata = {
            "file_path": str(file_path),
            "mineru_batch_id": parse_result['batch_id'],
            "mineru_data_id": parse_result['data_id'],
            "table_count": len(tables),
            "image_count": len(images),
            "enhanced_md_path": str(enhanced_md_path),
            "mineru_output_dir": str(download_dir),
            "original_file_name": pure_file_name 
        }
        
        result_df = pd.DataFrame([{
            "text": enhanced_md,
            "title": file_path.name,
            "id": parse_result['data_id'],
            "metadata": metadata,
            "creation_date": pd.Timestamp.now()
        }])
        
        all_results.append(result_df)
    
    # 合并所有结果
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["text", "title", "id", "metadata", "creation_date"])


# 独立测试入口
if __name__ == "__main__":
    import asyncio
    
    # 测试配置
    class TestConfig:
        mineru_api_token = os.getenv("MINERU_API_TOKEN")
        mineru_model_version = "vlm"
        mineru_output_dir = "./mineru_api_test_outputs"
        test_pdf_path = "test.pdf"
    
    # 运行测试
    config = TestConfig()
    result_df = asyncio.run(load_pdf(config))
    
    # 输出结果
    if not result_df.empty:
        log.info(f"\n✅ 处理完成！")
        log.info(f"📊 处理文件数: {len(result_df)}")
        log.info(f"📋 生成的增强Markdown路径: {result_df.iloc[0]['metadata']['enhanced_md_path']}")
        log.info(f"📈 提取表格数: {result_df.iloc[0]['metadata']['table_count']}")
        log.info(f"🖼️ 提取图片数: {result_df.iloc[0]['metadata']['image_count']}")
    else:
        log.error("❌ 处理失败，无结果")
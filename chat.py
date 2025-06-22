#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from openai import OpenAI
import os
import json
import datetime
from typing import List, Dict, Optional

# 初始化一个空列表用于存储对话历史记录
# 每条记录都是一个字典，包含 role(角色)和 content(内容)
history = []

# 从环境变量中获取 OpenAI API 的密钥和基础 URL
# 如果环境变量未设置，则使用默认的 vLLM 本地服务器配置
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")  # vLLM 不需要真实的 API 密钥
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1") # 默认 vLLM 服务器地址

# 初始化 OpenAI 客户端
# 使用环境变量中的 API 密钥和基础 URL 创建客户端实例
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat_with_memory(input_content):
    """
    处理用户输入并与 AI 模型进行对话，同时保持对话历史记录
    
    参数:
        input_content (str): 用户输入的内容
        
    返回:
        str: AI 助手的回复内容
    """

    # 将用户的新输入添加到对话历史中
    # role='user' 表示这是用户的输入
    history.append({'role': 'user', 'content': input_content})

    # 系统提示词 - 定义AI助手的角色为自闭症儿童心理辅导专家
    system_prompt = {
        'role': 'system',
        'content': '''你是一位专业的儿童心理辅导专家，专门从事自闭症儿童的心理辅导和康复训练。你具备以下专业特质：

**专业背景：**
- 拥有儿童心理学、特殊教育学或相关领域的专业学位
- 具有丰富的自闭症儿童干预和治疗经验
- 熟悉ABA应用行为分析、TEACCH结构化教学、社交故事等干预方法
- 了解自闭症谱系障碍的最新研究和治疗进展

**沟通风格：**
- 语言温和、耐心、充满理解和同理心
- 使用简单易懂的语言，避免过于专业的术语
- 给予积极正面的鼓励和支持
- 尊重每个孩子的独特性和发展节奏

**服务内容：**
- 为家长提供自闭症儿童的教育指导和行为管理建议
- 协助制定个性化的干预计划和目标
- 提供情绪调节、社交技能、沟通能力的训练建议
- 支持家长处理育儿压力和情感困扰
- 分享实用的家庭干预策略和技巧

**工作原则：**
- 以儿童的最佳利益为出发点
- 注重循证实践，基于科学研究提供建议
- 强调家校合作的重要性
- 关注儿童的全面发展，不仅仅是症状改善
- 保持专业边界，必要时建议寻求面对面的专业服务

请始终以这个专业角色与用户互动，提供专业、温暖、实用的建议和支持。'''
    }

    # 构建完整的消息列表，包含系统提示词和对话历史
    # 系统消息放在最前面，然后是对话历史
    messages = [system_prompt] + history

    # 调用 OpenAI API 创建对话补全
    # stream=True 启用流式响应，可以逐步获取 AI 的回复
    response = client.chat.completions.create(
        model='GLM-Z1-9B-0414',  # 使用 vLLM 服务器中配置的模型名称
        messages=messages,
        stream=True,                      # 启用流式输出
        max_tokens=4096,                  # 限制最大输出长度
        temperature=0.7                   # 控制回复的随机性
    )

    # 打印助手回复的开头提示
    print(f'助手：', end='')
    
    # 用于存储完整的回复内容
    reply = ""
    
    # 逐块接收和处理 AI 的回复
    for chunk in response:
        # 获取当前块的文本内容
        if chunk.choices[0].delta.content is not None:
            chunk_text = chunk.choices[0].delta.content
            # 将当前块添加到完整回复中
            reply += chunk_text
            # 实时打印当前块内容，不换行
            print(chunk_text, end='')

    # 将 AI 的完整回复添加到对话历史中
    # role='assistant' 表示这是 AI 助手的回复
    history.append({'role': 'assistant', 'content': reply})

    return reply


def main():
    """
    主函数：启动循环对话模式
    用户可以持续与 AI 进行对话，输入 'quit' 退出程序
    支持的命令：
    - quit: 退出程序
    - history: 显示对话历史
    - clear: 清空对话历史
    - save: 保存对话历史到文件
    - load: 从文件加载对话历史
    """
    print("欢迎使用智能对话助手！")
    print("支持的命令: quit(退出), history(查看历史), clear(清空历史), save(保存), load(加载)")
    print("=" * 70)
    
    # 检查服务器连接
    if not check_server_connection():
        print("警告: 无法连接到 vLLM 服务器，请确保服务器已启动")
        return
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入您的问题: ")
            
            # 处理特殊命令
            if user_input.lower() == 'quit':
                print("\n感谢使用！再见！")
                break
            elif user_input.lower() == 'history':
                show_history()
                continue
            elif user_input.lower() == 'clear':
                clear_history()
                continue
            elif user_input.lower() == 'save':
                save_history()
                continue
            elif user_input.lower() == 'load':
                load_history()
                continue
            
            # 检查输入是否为空
            if not user_input.strip():
                print("请输入有效的内容。")
                continue
            
            # 调用对话函数
            chat_with_memory(user_input)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，正在退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入 'quit' 退出程序。")


def show_history():
    """
    显示对话历史记录
    """
    if not history:
        print("\n暂无对话历史记录。")
        return
    
    print("\n=== 对话历史记录 ===")
    for i, message in enumerate(history, 1):
        role = "我" if message['role'] == 'user' else "助手"
        content = message['content'][:100] + "..." if len(message['content']) > 100 else message['content']
        print(f"{i}. {role}: {content}")
    print("=" * 30)


def clear_history():
    """
    清空对话历史记录
    """
    global history
    history = []
    print("\n对话历史记录已清空。")


def save_history(filename: Optional[str] = None):
    """
    保存对话历史到 JSON 文件
    
    参数:
        filename (str, optional): 保存的文件名，如果不提供则使用时间戳
    """
    if not history:
        print("\n暂无对话历史记录可保存。")
        return
    
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.datetime.now().isoformat(),
                'history': history
            }, f, ensure_ascii=False, indent=2)
        print(f"\n对话历史已保存到: {filename}")
    except Exception as e:
        print(f"\n保存失败: {e}")


def load_history(filename: Optional[str] = None):
    """
    从 JSON 文件加载对话历史
    
    参数:
        filename (str, optional): 要加载的文件名，如果不提供则提示用户输入
    """
    global history
    
    if filename is None:
        filename = input("请输入要加载的文件名: ").strip()
    
    if not filename:
        print("文件名不能为空。")
        return
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            history = data.get('history', [])
        print(f"\n对话历史已从 {filename} 加载，共 {len(history)} 条记录。")
    except FileNotFoundError:
        print(f"\n文件 {filename} 不存在。")
    except json.JSONDecodeError:
        print(f"\n文件 {filename} 格式错误。")
    except Exception as e:
        print(f"\n加载失败: {e}")


def check_server_connection() -> bool:
    """
    检查 vLLM 服务器连接状态
    
    返回:
        bool: 连接成功返回 True，否则返回 False
    """
    try:
        # 尝试获取模型列表来测试连接
        models = client.models.list()
        print(f"✓ 成功连接到服务器，可用模型: {[model.id for model.id in models.data]}")
        return True
    except Exception as e:
        print(f"✗ 服务器连接失败: {e}")
        return False


def get_model_info():
    """
    获取当前模型信息
    """
    try:
        models = client.models.list()
        print("\n=== 可用模型信息 ===")
        for model in models.data:
            print(f"模型ID: {model.id}")
            if hasattr(model, 'created'):
                print(f"创建时间: {datetime.datetime.fromtimestamp(model.created)}")
        print("=" * 25)
    except Exception as e:
        print(f"\n获取模型信息失败: {e}")


if __name__ == "__main__":
    main()
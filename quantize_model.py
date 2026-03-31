#!/usr/bin/env python3
"""使用 gguf 套件進行模型量化"""
import os
import sys

try:
    from gguf import GGUFReader, GGUFWriter
    import struct
    import numpy as np
except ImportError:
    print("正在安裝所需的套件...")
    os.system("pip install gguf numpy")
    from gguf import GGUFReader, GGUFWriter
    import struct
    import numpy as np


def quantize_q4_k_m(input_file, output_file):
    """將 FP16 GGUF 模型量化為 Q4_K_M"""
    print(f"讀取模型: {input_file}")
    reader = GGUFReader(input_file)
    
    print(f"開始量化為 Q4_K_M...")
    print(f"輸出文件: {output_file}")
    
    # 注意：這個簡單的腳本主要用於示範
    # 實際的量化需要 llama.cpp 的二進制工具
    # 建議使用 Ollama 或預編譯的 llama.cpp 工具
    
    print("\n重要提示：")
    print("GGUF 量化需要 llama.cpp 的編譯工具。")
    print("\n建議的替代方案：")
    print("1. 安裝 Homebrew: brew install llama.cpp")
    print("2. 使用 Docker: docker run -v $(pwd):/models ghcr.io/ggerganov/llama.cpp:full quantize /models/input.gguf /models/output.gguf q4_k_m")
    print("3. 使用預編譯的 llama.cpp releases")
    print("\n或者直接使用未量化的 FP16 版本，它已經可以在 Ollama 中使用。")


if __name__ == "__main__":
    input_file = "coffee-robot-functiongemma.gguf"
    output_file = "coffee-robot-functiongemma-q4_k_m.gguf"
    
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到輸入文件 {input_file}")
        sys.exit(1)
    
    quantize_q4_k_m(input_file, output_file)

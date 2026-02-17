#!/usr/bin/env python3
"""
CUDA/PyTorch 环境诊断脚本
用于排查 'CUDA error: named symbol not found' 等问题
"""

import sys
import subprocess

def run_command(cmd):
    """运行shell命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    print("=" * 80)
    print("CUDA/PyTorch 环境诊断")
    print("=" * 80)
    print()
    
    # 1. 系统CUDA版本
    print("1️⃣  系统CUDA版本:")
    print("-" * 40)
    nvcc_version = run_command("nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not found'")
    print(f"   nvcc: {nvcc_version}")
    
    nvidia_smi = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi failed'")
    print(f"   Driver: {nvidia_smi}")
    print()
    
    # 2. PyTorch信息
    print("2️⃣  PyTorch信息:")
    print("-" * 40)
    try:
        import torch
        print(f"   版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA版本 (PyTorch): {torch.version.cuda}")
            print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print("   ❌ CUDA不可用！")
        print()
        
        # 3. 测试基本CUDA操作
        print("3️⃣  测试基本CUDA操作:")
        print("-" * 40)
        
        if torch.cuda.is_available():
            try:
                # 测试张量创建
                x = torch.randn(100, 100).cuda()
                print("   ✓ 张量创建: 成功")
                
                # 测试张量运算
                y = x @ x.T
                print("   ✓ 张量运算: 成功")
                
                # 测试多GPU操作
                if torch.cuda.device_count() > 1:
                    try:
                        import torch.distributed as dist
                        # 不实际初始化，只检查是否可以导入
                        print(f"   ✓ 分布式模块: 可用")
                    except Exception as e:
                        print(f"   ⚠ 分布式模块: {e}")
                
                del x, y
                torch.cuda.empty_cache()
                print()
                
            except Exception as e:
                print(f"   ❌ CUDA操作失败: {e}")
                print()
        
        # 4. 检查版本兼容性
        print("4️⃣  版本兼容性检查:")
        print("-" * 40)
        
        pytorch_version = torch.__version__
        pytorch_cuda = torch.version.cuda if torch.cuda.is_available() else "N/A"
        
        # 提取主版本号
        if pytorch_cuda != "N/A":
            pt_cuda_major = pytorch_cuda.split('.')[0]
            
            # 检查系统CUDA
            system_cuda_version = run_command(
                "nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+' || echo 'unknown'"
            )
            
            if system_cuda_version != "unknown":
                sys_cuda_major = system_cuda_version.split('.')[0]
                
                if pt_cuda_major == sys_cuda_major:
                    print(f"   ✓ CUDA版本匹配: PyTorch={pytorch_cuda}, 系统={system_cuda_version}")
                else:
                    print(f"   ⚠ CUDA版本可能不匹配:")
                    print(f"      PyTorch CUDA: {pytorch_cuda}")
                    print(f"      系统 CUDA:    {system_cuda_version}")
                    print(f"      建议: 重装PyTorch以匹配系统CUDA版本")
            else:
                print(f"   ℹ️  无法检测系统CUDA版本")
                print(f"      PyTorch CUDA版本: {pytorch_cuda}")
        
        print()
        
        # 5. 检查其他依赖
        print("5️⃣  相关依赖版本:")
        print("-" * 40)
        
        try:
            import accelerate
            print(f"   Accelerate: {accelerate.__version__}")
        except ImportError:
            print(f"   Accelerate: 未安装")
        
        try:
            import transformers
            print(f"   Transformers: {transformers.__version__}")
        except ImportError:
            print(f"   Transformers: 未安装")
        
        try:
            import deepspeed
            print(f"   DeepSpeed: {deepspeed.__version__}")
        except ImportError:
            print(f"   DeepSpeed: 未安装")
        
        print()
        
        # 6. 建议
        print("6️⃣  诊断结果和建议:")
        print("-" * 40)
        
        if not torch.cuda.is_available():
            print("   ❌ CUDA不可用！")
            print("      可能原因:")
            print("      1. PyTorch安装的是CPU版本")
            print("      2. CUDA驱动未正确安装")
            print("      3. 环境变量配置错误")
            print()
            print("      解决方案:")
            print("      重新安装PyTorch GPU版本:")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        elif "named symbol not found" in str(sys.argv):
            print("   ⚠ 遇到 'named symbol not found' 错误")
            print("      可能原因:")
            print("      1. PyTorch CUDA版本与系统CUDA不完全兼容")
            print("      2. FSDP初始化时的特定问题")
            print("      3. GPU驱动版本过旧")
            print()
            print("      建议尝试:")
            print("      1. 使用DDP替代FSDP (configs/ddp_config.yaml)")
            print("      2. 重装PyTorch匹配系统CUDA版本")
            print("      3. 更新GPU驱动")
        
        else:
            print("   ✓ 环境配置正常")
            print("      如果仍遇到CUDA错误，建议:")
            print("      1. 先尝试DDP配置 (configs/ddp_config.yaml)")
            print("      2. 检查是否有残留的GPU进程 (nvidia-smi)")
            print("      3. 重启机器清理GPU状态")
        
    except ImportError as e:
        print(f"❌ 无法导入PyTorch: {e}")
        print("   请先安装PyTorch:")
        print("   pip install torch torchvision torchaudio")
    
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

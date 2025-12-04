#!/usr/bin/env python3
"""
PyCharm 深度学习环境版本检查与兼容性验证工具
检查: Python, PyTorch, OpenCV, MMDetection, MMFace
作者: AI助手
"""

import sys
import warnings
import subprocess
import platform
import json
from datetime import datetime
from packaging import version

# 抑制警告信息
warnings.filterwarnings("ignore")

# 兼容性对照表
COMPATIBILITY_MATRIX = {
    "MMDetection": {
        "2.x": {
            "PyTorch": ["1.6.0", "1.13.1"],
            "Python": ["3.6", "3.9"],
            "MMCV": ["1.3.17", "1.7.1"],
            "CUDA": ["10.1", "11.3"],
            "OpenCV": [None, None],  # 无特殊要求
            "备注": "MMDetection 2.x 系列"
        },
        "3.x": {
            "PyTorch": ["1.8.0", "2.1.0"],
            "Python": ["3.7", "3.11"],
            "MMCV": ["2.0.0", None],
            "MMEngine": ["0.7.0", None],
            "CUDA": ["11.1", "11.8"],
            "备注": "MMDetection 3.x 需要 MMEngine"
        }
    },
    "MMFace": {
        "要求": {
            "PyTorch": ["1.6.0", None],
            "Python": ["3.7", "3.9"],
            "MMCV": ["1.3.17", None],
            "备注": "通常基于 MMDetection 构建"
        }
    }
}


class Color:
    """终端颜色"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    """打印标题"""
    print(f"\n{Color.BOLD}{Color.CYAN}{'=' * 60}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{text.center(60)}{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'=' * 60}{Color.END}")


def print_section(text):
    """打印章节"""
    print(f"\n{Color.BOLD}{Color.YELLOW}{text}{Color.END}")
    print(f"{Color.YELLOW}{'-' * len(text)}{Color.END}")


def print_status(package, version, status="info", notes=""):
    """打印状态信息"""
    colors = {
        "success": Color.GREEN,
        "warning": Color.YELLOW,
        "error": Color.RED,
        "info": Color.BLUE
    }

    icon = {
        "success": "✓",
        "warning": "⚠",
        "error": "✗",
        "info": "ℹ"
    }

    color = colors.get(status, Color.BLUE)
    status_icon = icon.get(status, "ℹ")

    version_str = f"{Color.BOLD}{version}{Color.END}" if version else "未安装"
    print(f"{color}{status_icon}{Color.END} {Color.BOLD}{package}:{Color.END} {version_str} {notes}")


def get_system_info():
    """获取系统信息"""
    print_header("系统环境信息")

    info = {
        "操作系统": f"{platform.system()} {platform.release()}",
        "系统架构": platform.machine(),
        "处理器": platform.processor(),
        "Python解释器": sys.executable,
        "当前时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    for key, value in info.items():
        print(f"{Color.BLUE}ℹ{Color.END} {Color.BOLD}{key}:{Color.END} {value}")

    return info


def get_python_info():
    """获取Python信息"""
    print_section("Python 环境")

    info = {}

    # Python版本
    python_version = platform.python_version()
    info["版本"] = python_version

    # 检查Python版本兼容性
    py_ver = version.parse(python_version)
    if version.parse("3.7") <= py_ver <= version.parse("3.11"):
        status = "success"
        notes = f"{Color.GREEN}(推荐版本){Color.END}"
    elif py_ver < version.parse("3.6"):
        status = "error"
        notes = f"{Color.RED}(版本太低，建议升级){Color.END}"
    else:
        status = "warning"
        notes = f"{Color.YELLOW}(较新版本，可能有不兼容){Color.END}"

    print_status("Python", python_version, status, notes)

    # 检查pip版本
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                capture_output=True, text=True, check=True)
        pip_version = result.stdout.split()[1]
        info["pip版本"] = pip_version
        print_status("pip", pip_version, "info")
    except:
        info["pip版本"] = "未知"
        print_status("pip", "未知", "warning")

    return info


def get_pytorch_info():
    """获取PyTorch信息"""
    print_section("PyTorch 环境")

    info = {}

    try:
        import torch
        torch_version = torch.__version__
        info["版本"] = torch_version
        info["CUDA可用"] = torch.cuda.is_available()

        # 检查PyTorch版本
        torch_ver = version.parse(torch_version.split('+')[0])
        if version.parse("1.8.0") <= torch_ver <= version.parse("2.1.0"):
            status = "success"
            notes = f"{Color.GREEN}(稳定版本){Color.END}"
        else:
            status = "warning"
            notes = f"{Color.YELLOW}(非主流版本){Color.END}"

        print_status("PyTorch", torch_version, status, notes)

        # CUDA信息
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            info["CUDA版本"] = cuda_version
            info["GPU数量"] = torch.cuda.device_count()
            info["当前GPU"] = torch.cuda.get_device_name(0)

            print_status("CUDA", cuda_version, "success")
            print_status("GPU设备", torch.cuda.get_device_name(0), "info")
            print_status("GPU数量", str(torch.cuda.device_count()), "info")

            # 检查CUDA兼容性
            cuda_ver = version.parse(cuda_version)
            if version.parse("10.1") <= cuda_ver <= version.parse("11.8"):
                print_status("CUDA兼容性", "兼容", "success")
            else:
                print_status("CUDA兼容性", "可能有问题", "warning")
        else:
            info["CUDA版本"] = "不可用"
            print_status("CUDA", "不可用", "warning")

    except ImportError:
        info["错误"] = "未安装"
        print_status("PyTorch", "未安装", "error")
    except Exception as e:
        info["错误"] = str(e)
        print_status("PyTorch", "检查失败", "error", f"{Color.RED}({e}){Color.END}")

    return info


def get_opencv_info():
    """获取OpenCV信息"""
    print_section("OpenCV 环境")

    info = {}

    try:
        import cv2
        opencv_version = cv2.__version__
        info["版本"] = opencv_version

        # 检查OpenCV版本
        cv_ver = version.parse(opencv_version)
        if version.parse("4.5.0") <= cv_ver <= version.parse("4.8.0"):
            status = "success"
            notes = f"{Color.GREEN}(稳定版本){Color.END}"
        elif cv_ver < version.parse("4.0.0"):
            status = "warning"
            notes = f"{Color.YELLOW}(版本较旧){Color.END}"
        else:
            status = "info"
            notes = f"{Color.BLUE}(较新版本){Color.END}"

        print_status("OpenCV", opencv_version, status, notes)

        # 检查扩展模块
        try:
            has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            info["CUDA支持"] = has_cuda
            if has_cuda:
                print_status("OpenCV CUDA", "已启用", "success")
            else:
                print_status("OpenCV CUDA", "未启用", "info")
        except:
            info["CUDA支持"] = False
            print_status("OpenCV CUDA", "不支持", "info")

    except ImportError:
        info["错误"] = "未安装"
        print_status("OpenCV", "未安装", "error")
    except Exception as e:
        info["错误"] = str(e)
        print_status("OpenCV", "检查失败", "error", f"{Color.RED}({e}){Color.END}")

    return info


def get_mmdetection_info():
    """获取MMDetection信息"""
    print_section("MMDetection 环境")

    info = {}

    try:
        import mmdet
        mmdet_version = getattr(mmdet, '__version__', '未知')
        info["版本"] = mmdet_version

        # 判断MMDetection版本系列
        if mmdet_version.startswith('2.'):
            version_series = "2.x"
            status = "success"
            notes = f"{Color.GREEN}(经典稳定版){Color.END}"
        elif mmdet_version.startswith('3.'):
            version_series = "3.x"
            status = "success"
            notes = f"{Color.GREEN}(新版，需要MMEngine){Color.END}"
        else:
            version_series = "其他"
            status = "warning"
            notes = f"{Color.YELLOW}(非常规版本){Color.END}"

        info["版本系列"] = version_series
        print_status("MMDetection", mmdet_version, status, notes)

        # 检查MMCV
        try:
            import mmcv
            mmcv_version = getattr(mmcv, '__version__', '未知')
            info["MMCV版本"] = mmcv_version

            # 检查MMCV与MMDetection兼容性
            mmcv_ver = version.parse(mmcv_version.split('+')[0])

            if version_series == "2.x":
                if mmcv_version.startswith('1.'):
                    if version.parse("1.3.0") <= mmcv_ver <= version.parse("1.7.1"):
                        cv_status = "success"
                        cv_notes = f"{Color.GREEN}(兼容){Color.END}"
                    else:
                        cv_status = "warning"
                        cv_notes = f"{Color.YELLOW}(可能不兼容){Color.END}"
                else:
                    cv_status = "error"
                    cv_notes = f"{Color.RED}(不兼容: MMDetection 2.x 需要 MMCV 1.x){Color.END}"
            elif version_series == "3.x":
                if mmcv_version.startswith('2.'):
                    cv_status = "success"
                    cv_notes = f"{Color.GREEN}(兼容){Color.END}"
                else:
                    cv_status = "warning"
                    cv_notes = f"{Color.YELLOW}(MMDetection 3.x 推荐 MMCV 2.x){Color.END}"
            else:
                cv_status = "warning"
                cv_notes = f"{Color.YELLOW}(无法判断兼容性){Color.END}"

            print_status("MMCV", mmcv_version, cv_status, cv_notes)

        except ImportError:
            info["MMCV错误"] = "未安装"
            print_status("MMCV", "未安装", "error", f"{Color.RED}(MMDetection 需要 MMCV){Color.END}")

        # 检查MMEngine (仅MMDetection 3.x需要)
        try:
            import mmengine
            mmengine_version = getattr(mmengine, '__version__', '未知')
            info["MMEngine版本"] = mmengine_version

            if version_series == "3.x":
                engine_status = "success"
                engine_notes = f"{Color.GREEN}(必需组件){Color.END}"
            else:
                engine_status = "info"
                engine_notes = f"{Color.BLUE}(2.x不需要，但已安装){Color.END}"

            print_status("MMEngine", mmengine_version, engine_status, engine_notes)

        except ImportError:
            if version_series == "3.x":
                print_status("MMEngine", "未安装", "error",
                             f"{Color.RED}(MMDetection 3.x 需要 MMEngine){Color.END}")
            else:
                print_status("MMEngine", "未安装", "info",
                             f"{Color.BLUE}(MMDetection 2.x 不需要){Color.END}")

        # 测试MMDetection API
        try:
            from mmdet.apis import init_detector
            print_status("MMDetection API", "可用", "success")
            info["API状态"] = "正常"
        except Exception as e:
            print_status("MMDetection API", "异常", "error", f"{Color.RED}({e}){Color.END}")
            info["API状态"] = f"异常: {e}"

    except ImportError:
        info["错误"] = "未安装"
        print_status("MMDetection", "未安装", "error")
    except Exception as e:
        info["错误"] = str(e)
        print_status("MMDetection", "检查失败", "error", f"{Color.RED}({e}){Color.END}")

    return info


def get_mmface_info():
    """获取MMFace信息"""
    print_section("MMFace 环境")

    info = {}

    try:
        # MMFace 可能有不同的导入方式
        try:
            import mmface
            mmface_version = getattr(mmface, '__version__', '未知')
            module_name = "mmface"
        except:
            try:
                import face
                mmface_version = getattr(face, '__version__', '未知')
                module_name = "face"
            except:
                try:
                    import mmf
                    mmface_version = getattr(mmf, '__version__', '未知')
                    module_name = "mmf"
                except:
                    raise ImportError("未找到MMFace相关模块")

        info["版本"] = mmface_version
        info["模块名"] = module_name

        print_status("MMFace", f"{mmface_version} ({module_name})", "success")

        # 检查是否基于MMDetection
        try:
            import mmdet
            mmdet_version = mmdet.__version__
            info["MMDetection版本"] = mmdet_version
            print_status("依赖 - MMDetection", mmdet_version, "info")
        except:
            print_status("依赖 - MMDetection", "未找到", "warning")

    except ImportError:
        info["错误"] = "未安装"
        print_status("MMFace", "未安装", "warning",
                     f"{Color.YELLOW}(可选组件){Color.END}")
    except Exception as e:
        info["错误"] = str(e)
        print_status("MMFace", "检查失败", "error", f"{Color.RED}({e}){Color.END}")

    return info


def check_compatibility(all_info):
    """检查整体兼容性"""
    print_header("兼容性分析报告")

    issues = []
    warnings = []
    recommendations = []

    # 提取关键信息
    python_version = all_info.get("Python", {}).get("版本", "未知")
    torch_version = all_info.get("PyTorch", {}).get("版本", "未知")
    mmdet_version = all_info.get("MMDetection", {}).get("版本", "未知")
    mmcv_version = all_info.get("MMDetection", {}).get("MMCV版本", "未知")

    # 1. 检查Python版本
    py_ver = version.parse(python_version)
    if py_ver < version.parse("3.6"):
        issues.append(f"Python版本 {python_version} 过低，建议升级到 3.7+")
    elif py_ver > version.parse("3.11"):
        warnings.append(f"Python版本 {python_version} 较新，可能存在兼容性问题")

    # 2. 检查PyTorch与CUDA
    if all_info.get("PyTorch", {}).get("CUDA可用"):
        cuda_version = all_info.get("PyTorch", {}).get("CUDA版本", "未知")
        if cuda_version != "不可用":
            cuda_ver = version.parse(cuda_version)
            if not (version.parse("10.1") <= cuda_ver <= version.parse("11.8")):
                warnings.append(f"CUDA版本 {cuda_version} 可能不是最佳选择")
    else:
        warnings.append("CUDA不可用，训练和推理速度会受影响")

    # 3. 检查MMDetection版本兼容性
    if mmdet_version != "未知" and mmdet_version != "未安装":
        if mmdet_version.startswith('2.'):
            # MMDetection 2.x
            if mmcv_version and mmcv_version.startswith('2.'):
                issues.append("MMDetection 2.x 需要 MMCV 1.x，但检测到 MMCV 2.x")
                recommendations.append("建议: pip install mmcv-full==1.7.1")

            try:
                import mmengine
                warnings.append("MMDetection 2.x 通常不需要 MMEngine")
            except:
                pass

        elif mmdet_version.startswith('3.'):
            # MMDetection 3.x
            if mmcv_version and mmcv_version.startswith('1.'):
                issues.append("MMDetection 3.x 需要 MMCV 2.x，但检测到 MMCV 1.x")
                recommendations.append("建议: pip install mmcv>=2.0.0")

            try:
                import mmengine
            except ImportError:
                issues.append("MMDetection 3.x 需要 MMEngine")
                recommendations.append("建议: pip install mmengine")

    # 打印问题
    if issues:
        print(f"\n{Color.BOLD}{Color.RED}发现的问题:{Color.END}")
        for issue in issues:
            print(f"{Color.RED}✗ {issue}{Color.END}")

    if warnings:
        print(f"\n{Color.BOLD}{Color.YELLOW}注意事项:{Color.END}")
        for warning in warnings:
            print(f"{Color.YELLOW}⚠ {warning}{Color.END}")

    if recommendations:
        print(f"\n{Color.BOLD}{Color.GREEN}建议:{Color.END}")
        for rec in recommendations:
            print(f"{Color.GREEN}✓ {rec}{Color.END}")

    if not issues and not warnings:
        print(f"\n{Color.BOLD}{Color.GREEN}✓ 未发现兼容性问题{Color.END}")

    return {
        "问题": issues,
        "警告": warnings,
        "建议": recommendations
    }


def export_report(all_info, compatibility):
    """导出报告到文件"""
    try:
        report = {
            "生成时间": datetime.now().isoformat(),
            "系统信息": all_info.get("系统信息", {}),
            "Python信息": all_info.get("Python", {}),
            "PyTorch信息": all_info.get("PyTorch", {}),
            "OpenCV信息": all_info.get("OpenCV", {}),
            "MMDetection信息": all_info.get("MMDetection", {}),
            "MMFace信息": all_info.get("MMFace", {}),
            "兼容性分析": compatibility
        }

        filename = f"environment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{Color.GREEN}✓ 报告已保存到: {filename}{Color.END}")
        return filename
    except Exception as e:
        print(f"{Color.RED}✗ 保存报告失败: {e}{Color.END}")
        return None


def main():
    """主函数"""
    print_header("PyCharm 深度学习环境检查工具")

    all_info = {}

    try:
        # 安装packaging库（如果需要）
        try:
            from packaging import version
        except ImportError:
            print(f"{Color.YELLOW}正在安装packaging库...{Color.END}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging", "-q"])
            from packaging import version

        # 收集信息
        all_info["系统信息"] = get_system_info()
        all_info["Python"] = get_python_info()
        all_info["PyTorch"] = get_pytorch_info()
        all_info["OpenCV"] = get_opencv_info()
        all_info["MMDetection"] = get_mmdetection_info()
        all_info["MMFace"] = get_mmface_info()

        # 兼容性分析
        compatibility = check_compatibility(all_info)

        # 导出报告
        export_report(all_info, compatibility)

        print_header("检查完成")
        print(f"\n{Color.GREEN}✓ 环境检查已完成{Color.END}")

        # 总结
        has_errors = any(
            "错误" in info or
            ("CUDA可用" in info and not info.get("CUDA可用", True))
            for info in all_info.values() if isinstance(info, dict)
        )

        if has_errors:
            print(f"\n{Color.YELLOW}⚠ 环境存在一些问题，请查看上面的报告{Color.END}")
        else:
            print(f"\n{Color.GREEN}✓ 环境看起来基本正常{Color.END}")

        print(f"\n{Color.BLUE}提示:{Color.END} 查看保存的JSON报告文件获取详细信息")

    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}⚠ 用户中断{Color.END}")
        return 1
    except Exception as e:
        print(f"\n{Color.RED}✗ 检查过程中发生错误: {e}{Color.END}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
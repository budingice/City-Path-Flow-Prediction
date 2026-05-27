@echo off
REM ========================================================================================
REM 快速推理脚本 - Windows 批处理文件
REM ========================================================================================
REM 
REM 功能: 快速运行单模型推理或完整的多模型对比分析
REM
REM 使用方法:
REM   1. 双击此文件运行默认推理
REM   2. 或在命令行中指定参数:
REM      quick_run.bat infer     (单模型推理)
REM      quick_run.bat compare   (多模型对比)
REM      quick_run.bat test      (环境检查)
REM
REM ========================================================================================

setlocal enabledelayedexpansion

REM 配置
set PYTHON=python
set MODEL_NAME=adaptive_semantic_base
set ADJ_TYPE=semantic

REM 获取命令
if "%1"=="" (
    set CMD=infer
) else (
    set CMD=%1
)

echo.
echo ========================================================================================
echo AST-GCN 模型推理脚本
echo ========================================================================================
echo 命令: %CMD%
echo 模型: %MODEL_NAME%
echo 邻接矩阵类型: %ADJ_TYPE%
echo ========================================================================================
echo.

if "%CMD%"=="infer" (
    echo 🚀 执行单模型推理...
    echo.
    %PYTHON% scripts/infer_single_model.py ^
        --model_name %MODEL_NAME% ^
        --adj_type %ADJ_TYPE% ^
        --batch_size 32
    
) else if "%CMD%"=="compare" (
    echo 🚀 执行多模型对比...
    echo.
    %PYTHON% scripts/verify_model.py ^
        --model_name %MODEL_NAME% ^
        --adj_type %ADJ_TYPE% ^
        --num_time_steps 500
    
) else if "%CMD%"=="test" (
    echo 🧪 执行环境检查...
    echo.
    %PYTHON% scripts/test_verify_model.py
    
) else if "%CMD%"=="all" (
    echo 🔄 执行完整流程...
    echo.
    echo [1/3] 环境检查
    %PYTHON% scripts/test_verify_model.py
    if errorlevel 1 (
        echo 环境检查失败！请检查错误信息。
        pause
        exit /b 1
    )
    
    echo.
    echo [2/3] 单模型推理
    %PYTHON% scripts/infer_single_model.py ^
        --model_name %MODEL_NAME% ^
        --adj_type %ADJ_TYPE%
    
    echo.
    echo [3/3] 多模型对比
    %PYTHON% scripts/verify_model.py ^
        --model_name %MODEL_NAME% ^
        --include_baselines
    
) else (
    echo 未知命令: %CMD%
    echo.
    echo 可用命令:
    echo   infer   - 单模型推理 (默认)
    echo   compare - 多模型对比推理
    echo   test    - 环境检查
    echo   all     - 执行完整流程 (检查 + 推理 + 对比)
    echo.
    exit /b 1
)

echo.
echo ========================================================================================
echo ✅ 完成！
echo ========================================================================================
echo.
echo 结果保存位置:
echo   - 预测结果: experiments/predictions/
echo   - 对比图表: experiments/verify_plots/
echo.
echo 您可以在浏览器中打开 experiments/verify_plots/ 查看生成的图表。
echo.
pause

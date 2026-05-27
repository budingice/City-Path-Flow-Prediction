#!/bin/bash
# ========================================================================================
# 快速推理脚本 - Linux/Mac 版本
# ========================================================================================
#
# 功能: 快速运行单模型推理或完整的多模型对比分析
#
# 使用方法:
#   chmod +x scripts/quick_run.sh
#   ./scripts/quick_run.sh infer     # 单模型推理
#   ./scripts/quick_run.sh compare   # 多模型对比
#   ./scripts/quick_run.sh test      # 环境检查
#   ./scripts/quick_run.sh all       # 完整流程
#
# ========================================================================================

# 配置
PYTHON=${PYTHON:-python}
MODEL_NAME="adaptive_semantic_base"
ADJ_TYPE="semantic"
BATCH_SIZE=32

# 获取命令
CMD="${1:-infer}"

# 颜色输出 (可选)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo ""
    echo "========================================================================================"
    echo "$1"
    echo "========================================================================================"
    echo ""
}

print_step() {
    echo -e "${BLUE}>>> $1${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    echo ""
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    echo ""
}

# 主程序
print_header "🤖 AST-GCN 模型推理脚本"

echo "命令: $CMD"
echo "模型: $MODEL_NAME"
echo "邻接矩阵类型: $ADJ_TYPE"
echo "批大小: $BATCH_SIZE"
echo ""

case "$CMD" in
    infer)
        print_step "执行单模型推理..."
        $PYTHON scripts/infer_single_model.py \
            --model_name "$MODEL_NAME" \
            --adj_type "$ADJ_TYPE" \
            --batch_size "$BATCH_SIZE"
        ;;
        
    compare)
        print_step "执行多模型对比..."
        $PYTHON scripts/verify_model.py \
            --model_name "$MODEL_NAME" \
            --adj_type "$ADJ_TYPE" \
            --num_time_steps 500
        ;;
        
    test)
        print_step "执行环境检查..."
        $PYTHON scripts/test_verify_model.py
        ;;
        
    all)
        print_step "[1/3] 环境检查"
        $PYTHON scripts/test_verify_model.py
        if [ $? -ne 0 ]; then
            print_error "环境检查失败！"
            exit 1
        fi
        
        print_step "[2/3] 单模型推理"
        $PYTHON scripts/infer_single_model.py \
            --model_name "$MODEL_NAME" \
            --adj_type "$ADJ_TYPE" \
            --batch_size "$BATCH_SIZE"
        
        print_step "[3/3] 多模型对比"
        $PYTHON scripts/verify_model.py \
            --model_name "$MODEL_NAME" \
            --include_baselines \
            --num_time_steps 500
        ;;
        
    *)
        echo "未知命令: $CMD"
        echo ""
        echo "可用命令:"
        echo "  infer   - 单模型推理 (默认)"
        echo "  compare - 多模型对比推理"
        echo "  test    - 环境检查"
        echo "  all     - 执行完整流程 (检查 + 推理 + 对比)"
        echo ""
        echo "使用示例:"
        echo "  ./scripts/quick_run.sh infer"
        echo "  ./scripts/quick_run.sh compare"
        echo "  ./scripts/quick_run.sh all"
        echo ""
        exit 1
        ;;
esac

EXIT_CODE=$?

print_header "🎉 完成！"

if [ $EXIT_CODE -eq 0 ]; then
    print_success "推理成功！"
    echo "结果保存位置:"
    echo "  📁 预测结果: experiments/predictions/"
    echo "  📊 对比图表: experiments/verify_plots/"
    echo ""
    echo "🎨 查看生成的图表:"
    echo "  📷 对比图: experiments/verify_plots/comparison_sample0_path0.png"
    echo "  📈 时序图: experiments/verify_plots/timeseries_path0_*.png"
    echo ""
else
    print_error "推理过程出现错误！请查看上面的错误信息。"
fi

exit $EXIT_CODE

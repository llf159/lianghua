#!/bin/bash
# Linux启动脚本
cd "$(dirname "$0")"

# 读取配置的虚拟环境目录（默认 venv）
VENV_NAME=$(python3 - <<'PY' 2>/dev/null
try:
    import config
    print(getattr(config, "VENV_NAME", "venv") or "venv")
except Exception:
    print("venv")
PY
)
[ -n "$VENV_NAME" ] || VENV_NAME="venv"

VENV_DIR="$PWD/$VENV_NAME"
if [ -f "$VENV_DIR/bin/activate" ]; then
    # 使用 -f 避免因权限位不是可执行而误判
    source "$VENV_DIR/bin/activate"
else
    echo "未找到虚拟环境 $VENV_NAME，使用系统环境运行。"
fi

streamlit run score_ui.py

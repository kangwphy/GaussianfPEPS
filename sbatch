#!/bin/bash

# --- SLURM Directives ---
#SBATCH -p fat6326
#SBATCH -c 32
#SBATCH --exclude=fat632610

# ===================================================================
#                       USER PARAMETERS
# ===================================================================
# --- 1. Model and Physics Parameters ---
MODEL="2D"         # '1D' or '2D'
TYPE="energy"      # 'energy' or 'fidelity'
T1=1.0
T2=0.0
MU=0.0

# --- 2. Approach and Optimizer ---
APPROACH="Ration"   # 'Ration' or 'Variation'
OPTIMIZER="Powell"   # 'Nelder-Mead', 'Powell', 'L-BFGS-B', 'SLSQP', 'COBYLA'

# --- 3. Order Parameter ---
# 从 sbatch 命令行获取第一个参数 (e.g., sbatch submit_job.sh 5)
# 如果没有提供，则默认为 1
if [ -z "$1" ]; then
  echo "No start_order provided via \$1. Defaulting to 1."
  START_ORDER=1
else
  START_ORDER=$1
  echo "Using start_order = $1"
fi
# ===================================================================

# --- 4. Dynamic Job Name ---
# (e.g., 2D-Var-Pow-5)
SHORT_APPROACH=$( [ "$APPROACH" = "Variation" ] && echo "Var" || echo "Rat" )
JOB_NAME="${MODEL}-${SHORT_APPROACH}-${OPTIMIZER:0:3}-${START_ORDER}"
#SBATCH --job-name=${JOB_NAME}

# --- 5. Dynamic Log File ---
# 确保 logs 目录存在
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

# 创建一个包含所有参数的详细日志文件名
PARAM_STR="t1_${T1}_t2_${T2}_mu_${MU}"
LOG_FILE="${LOG_DIR}/${MODEL}_${TYPE}_${APPROACH}_${OPTIMIZER}_o${START_ORDER}_${PARAM_STR}2.log"

# --- 6. Set Job Output ---
# 将 SLURM 的标准输出和错误也重定向到我们的日志文件
#SBATCH -o ${LOG_FILE}
#SBATCH -e ${LOG_FILE}


# --- 7. Execute the Job ---
echo "=============================================="
echo "Starting job:   ${JOB_NAME}"
echo "Model:          ${MODEL}, t1=${T1}, t2=${T2}, mu=${MU}"
echo "Approach:       ${APPROACH}, Optimizer: ${OPTIMIZER}"
echo "Order:          ${START_ORDER}"
echo "Logging to:     ${LOG_FILE}"
echo "=============================================="
echo "" # 添加一个空行以分隔 sbatch 输出和 python 输出

# 运行 Python 脚本
# srun 会自动阻塞直到命令完成，不需要 & 和 wait
srun python main.py \
  --model ${MODEL} \
  --error_type ${TYPE} \
  --t1 ${T1} \
  --t2 ${T2} \
  --mu ${MU} \
  --approach ${APPROACH} \
  --optimizer ${OPTIMIZER} \
  --start_order ${START_ORDER}

echo ""
echo "=============================================="
echo "Job ${JOB_NAME} finished."
echo "=============================================="
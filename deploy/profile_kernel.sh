#!/bin/zsh

export TMAC_KCFG_FILE=./tuned/kcfg.ini 
export TMAC_KERNELS_LIBRARY=./build/kernels.dll 

# ./build/benchmark 3200 3200 1 5 10 1 1024

# 要调试的程序
PROGRAM="./your_program"
# 程序的参数
ARGS="arg1 arg2"

# 存放断点的文件，每行是 "file:line"
BREAKPOINT_FILE="./breakpoints.txt"

# 初始化 gdb 命令
GDB_CMD="gdb"

# 读取断点文件并生成 -ex "break ..." 命令
while IFS= read -r line
do
    GDB_CMD="$GDB_CMD -ex \"break $line\""
done < "$BREAKPOINT_FILE"

# 添加运行命令
GDB_CMD="$GDB_CMD -ex \"run\" --args $PROGRAM $ARGS"

# 执行 gdb 命令
eval $GDB_CMD



gdb --args ./build/benchmark 3200 3200 1 5 10 1 1024


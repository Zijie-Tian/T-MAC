import logging
import sys

import numpy as np
import tvm
from tvm import te
import tvm.testing

# 模块名叫 `autotvm`
from tvm import autotvm

@autotvm.template("matmul_cuda")
def matmul_cuda(M, N, K):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    # 创建调度
    sch = te.create_schedule(C.op)

    # 将 i 轴按块进行分割，block_dim=16
    i, j = C.op.axis
    
    cfg = autotvm.get_config()
    cfg.define_knob("tile_size", [16, 32, 64, 128])
    
    bx, tx = sch[C].split(i, factor=cfg["tile_size"].val)

    # 绑定到 CUDA 的 block 和 thread
    sch[C].bind(bx, te.thread_axis("blockIdx.x"))
    sch[C].bind(tx, te.thread_axis("threadIdx.x"))
    
    return sch, [A, B, C]

if __name__ == "__main__":
    # sch, [A, B, C] = matmul_cuda(128, 128, 128)
    task = autotvm.task.create("matmul_cuda", args=(128, 128, 128), target="cuda")
    print(task.config_space)
    
    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

    # 用 RandomTuner 开始调优, 日志记录到 `matmul.log` 文件中
    # 可用 XGBTuner 来替代.
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(
        n_trial=10,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("matmul.log")],
    )
    
    # 从日志文件中应用历史最佳
    with autotvm.apply_history_best("matmul.log"):
        with tvm.target.Target("cuda"):
            s, arg_bufs = matmul_cuda(128, 128, 128)
            func = tvm.build(s, arg_bufs)
                
    # 验证正确性
    a_np = np.random.uniform(size=(128, 128)).astype(np.float32)
    b_np = np.random.uniform(size=(128, 128)).astype(np.float32)
    c_np = a_np.dot(b_np)
    
    ctx = tvm.cuda(0)

    c_tvm = tvm.nd.empty(c_np.shape, device=tvm.runtime.Device(tvm.runtime.Device.kDLCUDA, 0))
    func(tvm.nd.array(a_np, device=tvm.runtime.Device(tvm.runtime.Device.kDLCUDA, 0)), 
         tvm.nd.array(b_np, device=tvm.runtime.Device(tvm.runtime.Device.kDLCUDA, 0)), c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)
    
    # 打印 func 编译生成的 C++ 代码
    cuda_source = func.imported_modules[0].get_source()
    print("Generated C++ code:")
    print(cuda_source)
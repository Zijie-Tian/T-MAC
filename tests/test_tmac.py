from t_mac.ops.gemm import GeMMCodegen

if __name__ == "__main__":
    gemm = GeMMCodegen(
        dtype="float32", 
        target="llvm", 
        name="gemm", 
        tune=True, 
        reuse_tuned=False, 
        verify=True, 
        save_dir="./test_tmac/", 
        target_host=None, 
        remote_kwargs=None, 
        cc=None, cc_opts=None, num_threads=4)
    
    gemm.compile(1024, 1024, 1024, thread_affinity=1, return_type="llvm")
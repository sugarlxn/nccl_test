run:
    mpirun --allow-run-as-root -np 4 \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x NCCL_TOPO_DUMP_FILE=/home/stone/lxn/nccl_tests/nccl_topo.xml \
    -x CUDA_VISIBLE_DEVICES=0,1,3,4 \
    /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 128M -e 1G -f 2 -g 1 -c 0 -n 50 -d int32 -w 10


run_with_nsys:
    nsys profile  \
        --trace-fork-before-exec=true \
        --output=/home/stone/lxn/nccl_tests/nsys_report \
        --trace=cuda,nvtx,mpi \
        --mpi-impl=openmpi \
        --force-overwrite=true \
        mpirun --allow-run-as-root -np 4 \
        -x NCCL_DEBUG=TRACE \
        -x NCCL_NET_GDR_LEVEL=3 \
        -x NCCL_GDR_READ=1 \
        -x NCCL_DEBUG_SUBSYS=ALL \
        -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
        -x NCCL_TOPO_DUMP_FILE=/home/stone/lxn/nccl_tests/nccl_topo.xml \
        -x CUDA_VISIBLE_DEVICES=4,5,6,7 \
        /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 512M -e 512M -f 2 -g 1 -c 0 -n 50 -d int32 -w 10

run_with_shimlib:
    mpirun --allow-run-as-root -np 2 \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x NCCL_ALGO=Ring \
    -x NCCL_PROTO=Simple \
    -x LD_PRELOAD=/home/stone/lxn/mTSS/build/libmtss_shim.so \
    /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 512K -e 512K -f 2 -g 1 -c 0 -n 20 -d int32

run_msccl:
    mpirun --allow-run-as-root -np 1 \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x NCCL_ALGO=MSCCL,Ring \
    -x NCCL_PROTO=Simple \
    -x MSCCL_XML_FILES=/home/stone/lxn/msccl-tools/config/test.xml \
    /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 512K -e 512K -f 2 -g 8 -c 0 -n 20 -d int32


run_with_nccl_profiler:
    mpirun --allow-run-as-root -np 4 \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x NCCL_PROFILER_PLUGIN=/home/stone/lxn/nccl/ext-profiler/example/libnccl-profiler.so \
    -x NCCL_PROFILE_DUMP_FILE=/home/stone/lxn/nccl_tests/nccl_profiler/nccl_profile.json \
    -x CUDA_VISIBLE_DEVICES=4,5,6,7 \
    /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 512M -e 512M -f 2 -g 1 -c 0 -n 50 -d int32 -w 10

run_with_nccl_tuner:
    mpirun --allow-run-as-root -np 4 \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x NCCL_TUNER_PLUGIN=/home/stone/lxn/NCCL-v2.28.9-1/nccl-2.28.9-1/ext-tuner/example/libnccl-tuner-example.so \
    -x NCCL_TUNER_CONFIG_FILE=/home/stone/lxn/NCCL-v2.28.9-1/nccl-2.28.9-1/ext-tuner/example/nccl_tuner_1node4dev.conf \
    -x CUDA_VISIBLE_DEVICES=0,1,3,4 \
    /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 128M -e 1G -f 2 -g 1 -c 0 -n 50 -d int32 -w 10

run_with_inspector_tuner:
    mpirun --allow-run-as-root -np 4 \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x CUDA_VISIBLE_DEVICES=1,2,3,4 \
    -x NCCL_PROFILER_PLUGIN=/home/stone/lxn/NCCL-v2.28.9-1/nccl-2.28.9-1/ext-profiler/inspector/libnccl-profiler-inspector.so \
    -x NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500 \
    -x NCCL_INSPECTOR_DUMP_DIR=/home/stone/lxn/nccl_tests/baseline/nccl_tuner/all_reduce_perf_1node_4dev_4090 \
    -x NCCL_INSPECTOR_ENABLE=1 \
    -x NCCL_TENANT_ID=1001 \
    -x NCCL_TUNER_PLUGIN=/home/stone/lxn/NCCL-v2.28.9-1/nccl-2.28.9-1/ext-tuner/example/libnccl-tuner-example.so \
    -x NCCL_TUNER_CONFIG_FILE=/home/stone/lxn/NCCL-v2.28.9-1/nccl-2.28.9-1/ext-tuner/example/nccl_tuner_1node4dev.conf \
    /home/stone/lxn/nccl_tests/build/all_reduce_perf -b 4K -e 4G -f 2 -g 1 -c 0 -n 50 -d int32 -w 10

run_with_inspector:
    mpirun --allow-run-as-root -np 4 \
    -x NCCL_DEBUG=TRACE \
    -x NCCL_NET_GDR_LEVEL=3 \
    -x NCCL_GDR_READ=1 \
    -x NCCL_DEBUG_SUBSYS=ALL \
    -x NCCL_DEBUG_FILE=/home/stone/lxn/nccl_tests/nccl_trace.log \
    -x CUDA_VISIBLE_DEVICES=1,2,3,4 \
    -x NCCL_PROFILER_PLUGIN=/home/stone/lxn/NCCL-v2.28.9-1/nccl-2.28.9-1/ext-profiler/inspector/libnccl-profiler-inspector.so \
    -x NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500 \
    -x NCCL_INSPECTOR_DUMP_DIR=/home/stone/lxn/nccl_tests/baseline/nccl_default/alltoall_perf_1node_4dev_4090 \
    -x NCCL_INSPECTOR_ENABLE=1 \
    -x NCCL_TENANT_ID=1001 \
    /home/stone/lxn/nccl_tests/build/alltoall_perf -b 4k -e 4G -f 2 -g 1 -c 0 -n 50 -d int32 -w 10


make:
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-12.4/lib64:/usr/local/openmpi/lib
    make -j 48 MPI=1 MPI_HOME=/usr/local/openmpi CUDA_HOME=/usr/local/cuda-12.4


remake:
    make clean
    just make

make_msccl:
    make clean
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/openmpi/lib
    make -j 48 MPI=1 MPI_HOME=/usr/local/openmpi CUDA_HOME=/usr/local/cuda-12.4

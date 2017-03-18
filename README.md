imageFilteringGpu
-----
フィルタ処理のCUDA実装

# 確認環境
筆者は以下の環境にて動作確認を行いました。

* Ubuntu 16.04.1 LTS
* CUDA Toolkit v8.0
* NVCC 8.0.26
* GCC 5.4.0
* OpenCV 3.1(<code>WITH_CUDA</code>=ON)
* GeForce GTX 1080

# 備考
## OpenCV関係
CMake実行時、<code>OpenCV_DIR</code>にOpenCVのインストールパスを設定する必要があります。

## CUDA関係
同梱のCMakeLists.txtではコード生成を以下のように設定しています。

```
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61")
```

動作GPUに応じて適切なCompute Capabilityを指定する必要があります。<br>
※Compute Capabilityは[NVIDIAサイト](https://developer.nvidia.com/cuda-gpus)で確認することができます。

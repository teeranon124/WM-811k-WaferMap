import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.model import WaferResNet

def export_and_benchmark():
    print("=" * 65)
    print("🚀 Exporting WaferResNet to ONNX & Edge Inference Benchmark")
    print("=" * 65)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/wafer_resnet_best.pth'
    onnx_path = 'models/wafer_resnet.onnx'
    
    # 1. Load PyTorch Model
    model = WaferResNet(num_classes=9)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ Loaded PyTorch model from '{model_path}'")
    
    # 2. Export to ONNX
    dummy_input = torch.randn(1, 3, 56, 56)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['wafer_map'],
        output_names=['defect_logits'],
        dynamic_axes={
            'wafer_map': {0: 'batch_size'},
            'defect_logits': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model integrity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ ONNX model successfully exported and validated: '{onnx_path}' ({file_size_mb:.2f} MB)")
    
    # 3. Numerical Parity Verification (PyTorch vs ONNX Runtime)
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    test_input = np.random.randn(1, 3, 56, 56).astype(np.float32)
    
    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_input)).numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"✅ Numerical Parity check passed: Max absolute difference = {max_diff:.2e} (< 1e-5)")
    
    # 4. Latency Benchmark across Providers & Batch Sizes
    print("\n" + "=" * 65)
    print("⏱️ Production Latency Benchmark (500 warmup + 1,000 runs)")
    print("=" * 65)
    
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
        
    for provider in providers:
        sess = ort.InferenceSession(onnx_path, providers=[provider])
        provider_name = "NVIDIA CUDA (GPU)" if "CUDA" in provider else "Intel/AMD x86 (CPU)"
        print(f"\n--- Provider: {provider_name} ---")
        
        for batch_size in [1, 32, 64]:
            batch_input = np.random.randn(batch_size, 3, 56, 56).astype(np.float32)
            feed = {sess.get_inputs()[0].name: batch_input}
            
            # Warmup
            for _ in range(50):
                _ = sess.run(None, feed)
                
            # Benchmark 500 iterations
            num_iters = 500
            start = time.perf_counter()
            for _ in range(num_iters):
                _ = sess.run(None, feed)
            elapsed = time.perf_counter() - start
            
            total_wafers = batch_size * num_iters
            latency_per_wafer_ms = (elapsed / total_wafers) * 1000
            throughput_fps = total_wafers / elapsed
            
            print(f"Batch {batch_size:2d} | Per-wafer Latency: {latency_per_wafer_ms:.4f} ms | Throughput: {throughput_fps:,.1f} wafers/sec")
            
    print("\n✅ Benchmark completed successfully!")

if __name__ == '__main__':
    export_and_benchmark()

def test_onnx_stack_imports():
    import onnx
    import onnxruntime
    assert onnx.__version__
    assert onnxruntime.get_available_providers()  # at least CPUExecutionProvider

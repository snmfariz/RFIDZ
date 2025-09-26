import numpy as np
import onnxruntime as ort

class ONNX_API:
    """Run ONNX denoising with batched inference."""
    def __init__(self, path_to_onnx: str):
        self.path_to_model = path_to_onnx
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.path_to_model, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def denoise(self, obs: np.ndarray, batch_size: int = 16):
        """Denoise and return arrays shaped like the inputs."""
        if obs.ndim == 2:
            obs_bchw = obs[None, None, :, :]
        elif obs.ndim == 3:
            obs_bchw = obs[:, None, :, :]
        elif obs.ndim == 4 and obs.shape[1] == 1:
            obs_bchw = obs
        else:
            raise ValueError("obs must be (H,W), (N,H,W) or (N,1,H,W)")
        obs_bchw = obs_bchw.astype(np.float32, copy=False)
        N = obs_bchw.shape[0]
        outputs = []
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            x = obs_bchw[s:e]
            y = self.sess.run([self.output_name], {self.input_name: x})[0]
            outputs.append(y)
        y = np.concatenate(outputs, axis=0)
        y = y[:, 0]
        return y[0] if y.shape[0] == 1 else y

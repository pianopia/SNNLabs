let gpuState = {
  initialized: false,
  available: false,
  device: null,
  pipeline: null,
  bindGroupLayout: null,
  reason: 'not-initialized',
};

const shaderCode = `
struct Params {
  salience: f32,
  rewardAbs: f32,
  count: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> inputV: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputV: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= params.count) {
    return;
  }
  outputV[i] = inputV[i] * 0.72 + 0.48 + params.salience * 0.24 + params.rewardAbs * 0.18;
}
`;

export const getWebGpuStatus = () => ({
  available: gpuState.available,
  initialized: gpuState.initialized,
  reason: gpuState.reason,
});

export const initWebGpuLearner = async () => {
  if (gpuState.initialized) return getWebGpuStatus();
  gpuState.initialized = true;

  if (!globalThis.navigator?.gpu) {
    gpuState.reason = 'navigator.gpu unavailable';
    return getWebGpuStatus();
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      gpuState.reason = 'adapter unavailable';
      return getWebGpuStatus();
    }
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: shaderCode });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });
    gpuState = {
      initialized: true,
      available: true,
      device,
      pipeline,
      bindGroupLayout,
      reason: 'available',
    };
  } catch (error) {
    gpuState.reason = String(error?.message || error);
  }
  return getWebGpuStatus();
};

export const computeVoltagesOnGpu = async ({ voltages, salience, reward }) => {
  await initWebGpuLearner();
  if (!gpuState.available || voltages.length === 0) {
    return { backend: 'cpu', voltages: null, status: getWebGpuStatus() };
  }

  const device = gpuState.device;
  const input = new Float32Array(voltages);
  const byteLength = Math.max(4, input.byteLength);
  const inputBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const params = new ArrayBuffer(16);
  const paramsView = new DataView(params);
  paramsView.setFloat32(0, salience, true);
  paramsView.setFloat32(4, Math.abs(reward), true);
  paramsView.setUint32(8, input.length, true);
  paramsView.setUint32(12, 0, true);
  device.queue.writeBuffer(inputBuffer, 0, input);
  device.queue.writeBuffer(paramsBuffer, 0, params);

  const bindGroup = device.createBindGroup({
    layout: gpuState.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gpuState.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(input.length / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const result = Array.from(new Float32Array(readBuffer.getMappedRange()).slice(0, input.length));
  readBuffer.unmap();
  return { backend: 'webgpu', voltages: result, status: getWebGpuStatus() };
};

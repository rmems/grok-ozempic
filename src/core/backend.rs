use half::f16;

use crate::core::quantizer::{self, QuantizedTensor};

/// A deployable kernel backend that performs tensor quantization operations.
///
/// `grok-ozempic` calls through this trait rather than invoking quantizer
/// functions directly. The two implementations are:
///
/// - [`LocalBackend`] — delegates to the existing CPU quantizer in `quantizer.rs`.
/// - [`MyelinBackend`] — stub for the `myelin-accelerator` CUDA FFI bridge.
pub trait BackendKernel {
    /// Pack a slice of ternary floats into 2-bit representation (4 values/byte).
    fn pack_ternary(&self, ternary: &[f32]) -> Vec<u8>;

    /// Full ternary quantization of FP32 weights: RMS saliency, GIF threshold,
    /// ternary gate, then 2-bit packing.
    fn quantize_f32(&self, weights: &[f32], gif_threshold: f32) -> QuantizedTensor;

    /// Full ternary quantization of FP16 weights (converts to F32 internally).
    fn quantize_f16(&self, weights: &[f16], gif_threshold: f32) -> QuantizedTensor;

    /// Re-encode FP16 weights as raw little-endian FP16 bytes (passthrough).
    fn passthrough_f16(&self, weights: &[f16]) -> Vec<u8>;

    /// Convert FP32 weights to FP16 bytes for mixed-precision passthrough.
    fn convert_f32_to_f16_bytes(&self, weights: &[f32]) -> Vec<u8>;
}

// ---------------------------------------------------------------------------
// LocalBackend — delegates to the existing CPU quantizer
// ---------------------------------------------------------------------------

/// Backend that runs all kernel operations on CPU via the existing
/// `quantizer.rs` implementations.
///
/// This is the default backend. It produces identical results to the
/// pre-refactoring pipeline — the same ternary packing, the same GIF
/// threshold logic, the same FP16 encoding.
pub struct LocalBackend;

impl LocalBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LocalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendKernel for LocalBackend {
    fn pack_ternary(&self, ternary: &[f32]) -> Vec<u8> {
        quantizer::pack_trits(ternary)
    }

    fn quantize_f32(&self, weights: &[f32], gif_threshold: f32) -> QuantizedTensor {
        quantizer::quantize_f32(weights, gif_threshold)
    }

    fn quantize_f16(&self, weights: &[f16], gif_threshold: f32) -> QuantizedTensor {
        quantizer::quantize_f16(weights, gif_threshold)
    }

    fn passthrough_f16(&self, weights: &[f16]) -> Vec<u8> {
        quantizer::passthrough_f16(weights)
    }

    fn convert_f32_to_f16_bytes(&self, weights: &[f32]) -> Vec<u8> {
        quantizer::convert_f32_to_f16_bytes(weights)
    }
}

// ---------------------------------------------------------------------------
// MyelinBackend — stub for myelin-accelerator FFI
// ---------------------------------------------------------------------------

/// Stub backend that will delegate kernel operations to `myelin-accelerator`
/// via Rust/CUDA FFI once the dependency is linked.
///
/// Every method currently returns an error. This establishes the integration
/// point so callers can be written against the `MyelinBackend` type before the
/// actual CUDA library is available.
pub struct MyelinBackend;

impl MyelinBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MyelinBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendKernel for MyelinBackend {
    fn pack_ternary(&self, _ternary: &[f32]) -> Vec<u8> {
        unimplemented!("myelin-accelerator FFI not yet linked; use LocalBackend for CPU fallback")
    }

    fn quantize_f32(&self, _weights: &[f32], _gif_threshold: f32) -> QuantizedTensor {
        unimplemented!("myelin-accelerator FFI not yet linked; use LocalBackend for CPU fallback")
    }

    fn quantize_f16(&self, _weights: &[f16], _gif_threshold: f32) -> QuantizedTensor {
        unimplemented!("myelin-accelerator FFI not yet linked; use LocalBackend for CPU fallback")
    }

    fn passthrough_f16(&self, _weights: &[f16]) -> Vec<u8> {
        unimplemented!("myelin-accelerator FFI not yet linked; use LocalBackend for CPU fallback")
    }

    fn convert_f32_to_f16_bytes(&self, _weights: &[f32]) -> Vec<u8> {
        unimplemented!("myelin-accelerator FFI not yet linked; use LocalBackend for CPU fallback")
    }
}

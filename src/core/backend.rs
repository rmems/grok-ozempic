use half::f16;

use crate::core::quantizer::{self, QuantizedTensor};
use crate::error::{GrokOzempicError, Result};

#[cfg(feature = "myelin")]
const MYELIN_PACK_CHUNK_TRITS: usize = 16 * 1024;

/// A deployable kernel backend that performs tensor quantization operations.
///
/// `grok-ozempic` calls through this trait rather than invoking quantizer
/// functions directly. The two implementations are:
///
/// - [`LocalBackend`] — delegates to the existing CPU quantizer in `quantizer.rs`.
/// - [`MyelinBackend`] — feature-gated host packing via `myelin-accelerator`;
///   the remaining CUDA bridge is not yet integrated.
pub trait BackendKernel {
    /// Pack a slice of ternary floats into 2-bit representation (4 values/byte).
    fn pack_ternary(&self, ternary: &[f32]) -> Result<Vec<u8>>;

    /// Full ternary quantization of FP32 weights: RMS saliency, GIF threshold,
    /// ternary gate, then 2-bit packing.
    fn quantize_f32(&self, weights: &[f32], gif_threshold: f32) -> Result<QuantizedTensor>;

    /// Full ternary quantization of FP16 weights (converts to F32 internally).
    fn quantize_f16(&self, weights: &[f16], gif_threshold: f32) -> Result<QuantizedTensor>;

    /// Re-encode FP16 weights as raw little-endian FP16 bytes (passthrough).
    fn passthrough_f16(&self, weights: &[f16]) -> Result<Vec<u8>>;

    /// Convert FP32 weights to FP16 bytes for mixed-precision passthrough.
    fn convert_f32_to_f16_bytes(&self, weights: &[f32]) -> Result<Vec<u8>>;
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
    fn pack_ternary(&self, ternary: &[f32]) -> Result<Vec<u8>> {
        Ok(quantizer::pack_trits(ternary))
    }

    fn quantize_f32(&self, weights: &[f32], gif_threshold: f32) -> Result<QuantizedTensor> {
        Ok(quantizer::quantize_f32(weights, gif_threshold))
    }

    fn quantize_f16(&self, weights: &[f16], gif_threshold: f32) -> Result<QuantizedTensor> {
        Ok(quantizer::quantize_f16(weights, gif_threshold))
    }

    fn passthrough_f16(&self, weights: &[f16]) -> Result<Vec<u8>> {
        Ok(quantizer::passthrough_f16(weights))
    }

    fn convert_f32_to_f16_bytes(&self, weights: &[f32]) -> Result<Vec<u8>> {
        Ok(quantizer::convert_f32_to_f16_bytes(weights))
    }
}

// ---------------------------------------------------------------------------
// MyelinBackend — partial myelin-accelerator integration
// ---------------------------------------------------------------------------

/// Backend integration point for `myelin-accelerator`.
///
/// With the `myelin` feature, [`BackendKernel::pack_ternary`] uses myelin's
/// host-only GOZ1-compatible byte packer. The remaining methods return
/// [`GrokOzempicError::BackendNotAvailable`] until the CUDA adapter is complete.
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
    fn pack_ternary(&self, ternary: &[f32]) -> Result<Vec<u8>> {
        #[cfg(feature = "myelin")]
        {
            let mut packed = Vec::with_capacity(ternary.len().div_ceil(4));
            for chunk in ternary.chunks(MYELIN_PACK_CHUNK_TRITS) {
                let trits: Vec<i8> = chunk
                    .iter()
                    .map(|&value| {
                        if value > 0.0 {
                            1
                        } else if value < 0.0 {
                            -1
                        } else {
                            0
                        }
                    })
                    .collect();
                packed.extend(myelin_accelerator::bitpacking::pack_ternary_bytes(&trits));
            }
            Ok(packed)
        }

        #[cfg(not(feature = "myelin"))]
        {
            let _ = ternary;
            Err(GrokOzempicError::BackendNotAvailable(
                "myelin feature is not enabled; use LocalBackend for CPU fallback".into(),
            ))
        }
    }

    fn quantize_f32(&self, _weights: &[f32], _gif_threshold: f32) -> Result<QuantizedTensor> {
        Err(GrokOzempicError::BackendNotAvailable(
            "myelin quantization is not yet integrated; use LocalBackend for CPU fallback".into(),
        ))
    }

    fn quantize_f16(&self, _weights: &[f16], _gif_threshold: f32) -> Result<QuantizedTensor> {
        Err(GrokOzempicError::BackendNotAvailable(
            "myelin quantization is not yet integrated; use LocalBackend for CPU fallback".into(),
        ))
    }

    fn passthrough_f16(&self, _weights: &[f16]) -> Result<Vec<u8>> {
        Err(GrokOzempicError::BackendNotAvailable(
            "myelin FP16 passthrough is not yet integrated; use LocalBackend for CPU fallback"
                .into(),
        ))
    }

    fn convert_f32_to_f16_bytes(&self, _weights: &[f32]) -> Result<Vec<u8>> {
        Err(GrokOzempicError::BackendNotAvailable(
            "myelin FP16 conversion is not yet integrated; use LocalBackend for CPU fallback"
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::quantizer;
    use half::f16;

    #[test]
    fn local_pack_ternary_matches_direct_call() {
        let backend = LocalBackend::new();
        let ternary = [1.0f32, -1.0, 0.0, 1.0, -1.0];
        let result = backend.pack_ternary(&ternary).unwrap();
        let expected = quantizer::pack_trits(&ternary);
        assert_eq!(result, expected);
    }

    #[test]
    fn local_pack_ternary_empty_slice() {
        let backend = LocalBackend::new();
        let result = backend.pack_ternary(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn local_quantize_f32_matches_direct_call() {
        let backend = LocalBackend::new();
        let weights = vec![0.5f32, -0.3, 1.2, -0.8, 0.01, 2.0];
        let gif_threshold = 0.5;
        let result = backend.quantize_f32(&weights, gif_threshold).unwrap();
        let expected = quantizer::quantize_f32(&weights, gif_threshold);
        assert_eq!(result.packed, expected.packed);
        assert_eq!(result.num_elements, expected.num_elements);
        assert_eq!(result.rms, expected.rms);
        assert_eq!(result.threshold, expected.threshold);
        assert_eq!(result.sparsity, expected.sparsity);
    }

    #[test]
    fn local_quantize_f32_empty_slice() {
        let backend = LocalBackend::new();
        let result = backend.quantize_f32(&[], 0.5).unwrap();
        assert_eq!(result.num_elements, 0);
        assert!(result.packed.is_empty());
    }

    #[test]
    fn local_quantize_f16_matches_direct_call() {
        let backend = LocalBackend::new();
        let weights: Vec<f16> = vec![
            f16::from_f32(0.5),
            f16::from_f32(-0.3),
            f16::from_f32(1.2),
            f16::from_f32(-0.8),
        ];
        let gif_threshold = 0.5;
        let result = backend.quantize_f16(&weights, gif_threshold).unwrap();
        let expected = quantizer::quantize_f16(&weights, gif_threshold);
        assert_eq!(result.packed, expected.packed);
        assert_eq!(result.num_elements, expected.num_elements);
        assert_eq!(result.rms, expected.rms);
        assert_eq!(result.threshold, expected.threshold);
        assert_eq!(result.sparsity, expected.sparsity);
    }

    #[test]
    fn local_passthrough_f16_matches_direct_call() {
        let backend = LocalBackend::new();
        let weights = vec![f16::from_f32(1.5), f16::from_f32(-2.5), f16::from_f32(0.0)];
        let result = backend.passthrough_f16(&weights).unwrap();
        let expected = quantizer::passthrough_f16(&weights);
        assert_eq!(result, expected);
    }

    #[test]
    fn local_convert_f32_to_f16_matches_direct_call() {
        let backend = LocalBackend::new();
        let weights = vec![1.5f32, -2.5, 0.0, 3.15];
        let result = backend.convert_f32_to_f16_bytes(&weights).unwrap();
        let expected = quantizer::convert_f32_to_f16_bytes(&weights);
        assert_eq!(result, expected);
    }

    #[cfg(not(feature = "myelin"))]
    #[test]
    fn myelin_pack_ternary_returns_error() {
        let backend = MyelinBackend::new();
        let result = backend.pack_ternary(&[1.0, -1.0, 0.0]);
        assert!(matches!(
            result,
            Err(GrokOzempicError::BackendNotAvailable(_))
        ));
    }

    #[cfg(feature = "myelin")]
    #[test]
    fn myelin_pack_ternary_uses_goz1_byte_layout() {
        let backend = MyelinBackend::new();
        let packed = backend.pack_ternary(&[1.0, -1.0, 0.0, 1.0, -1.0]).unwrap();

        assert_eq!(packed, vec![0b0100_1001, 0b0000_0010]);
    }

    #[cfg(feature = "myelin")]
    #[test]
    fn myelin_pack_ternary_preserves_sign_semantics() {
        let backend = MyelinBackend::new();
        let packed = backend
            .pack_ternary(&[f32::MIN_POSITIVE, -f32::MIN_POSITIVE, -0.0, f32::NAN])
            .unwrap();

        assert_eq!(packed, vec![0b0000_1001]);
    }

    #[cfg(feature = "myelin")]
    #[test]
    fn myelin_pack_ternary_preserves_layout_across_chunks() {
        let backend = MyelinBackend::new();
        let ternary: Vec<f32> = (0..MYELIN_PACK_CHUNK_TRITS + 5)
            .map(|index| match index % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();

        let packed = backend.pack_ternary(&ternary).unwrap();

        assert_eq!(packed, quantizer::pack_trits(&ternary));
    }

    #[test]
    fn myelin_quantize_f32_returns_error() {
        let backend = MyelinBackend::new();
        let result = backend.quantize_f32(&[1.0, 2.0], 0.5);
        assert!(matches!(
            result,
            Err(GrokOzempicError::BackendNotAvailable(_))
        ));
    }

    #[test]
    fn myelin_quantize_f16_returns_error() {
        let backend = MyelinBackend::new();
        let result = backend.quantize_f16(&[f16::from_f32(1.0)], 0.5);
        assert!(matches!(
            result,
            Err(GrokOzempicError::BackendNotAvailable(_))
        ));
    }

    #[test]
    fn myelin_passthrough_f16_returns_error() {
        let backend = MyelinBackend::new();
        let result = backend.passthrough_f16(&[f16::from_f32(1.0)]);
        assert!(matches!(
            result,
            Err(GrokOzempicError::BackendNotAvailable(_))
        ));
    }

    #[test]
    fn myelin_convert_f32_to_f16_returns_error() {
        let backend = MyelinBackend::new();
        let result = backend.convert_f32_to_f16_bytes(&[1.0, 2.0]);
        assert!(matches!(
            result,
            Err(GrokOzempicError::BackendNotAvailable(_))
        ));
    }
}

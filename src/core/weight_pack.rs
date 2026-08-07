//! **GOZ1** — grok-ozempic packed checkpoint format (not GGUF).
//!
//! Little-endian container: metadata key/value pairs, tensor table with
//! placeholder offsets, alignment padding, then streamed tensor blobs.
//!
//! # Streaming
//! [`PackStreamWriter`] writes tensor info with placeholder offsets, then each
//! payload once via [`PackStreamWriter::write_tensor_data`], and patches offsets
//! in [`PackStreamWriter::finalize`].
//!
//! # Versions
//!
//! | Version | Tensor row |
//! |---------|------------|
//! | 1 | `name, ndim, shape[], tensor_type, data_offset` |
//! | 2 | …then `scale: f32`, `sentinel: u32` |
//! | 3 | …then `scale: f32`, `gif_threshold: f32`, `threshold_abs: f32`, `sentinel: u32` |
//!
//! Version 2 (GH #65) appended the per-tensor reconstruction scale so a pack can
//! be dequantized **from its own contents**. A v1 pack stores only trits, which
//! carry sign but no magnitude, so `w ≈ α·t` had no recoverable `α` and every
//! consumer had to derive one from the original checkpoint — see
//! [`crate::core::quantizer::QuantizedTensor::scale`].
//!
//! Version 3 (GH #66) adds the **applied threshold**, closing the companion
//! provenance gap: τ resolves per tensor
//! (`ternary_candidates[].gif_threshold` > `defaults` > CLI), but the pack-level
//! `oz.gif_threshold` metadata key records only `defaults || config`, so it
//! misreports every tensor that carried an override (GH #58). Both senses of "τ"
//! are stored, because this crate uses the symbol for both and they are different
//! numbers: `gif_threshold` is the multiplier, `threshold_abs = gif_threshold ×
//! rms` is the cut actually compared against `|w|`. Storing both also recovers
//! `rms = threshold_abs / gif_threshold` for free.
//!
//! Each versioned row is a strict *append*, so a reader parses the common prefix
//! and reads later fields only when the version says they are there.
//! Reconstruction is uniform across payload kinds: `value = scale × payload`,
//! where the payload is a trit for [`TENSOR_TERNARY`] and the stored half itself
//! for [`TENSOR_F16`] (whose scale is therefore `1.0`, with both thresholds `0.0`
//! since no GIF gate ran).

use std::{
    collections::BTreeMap,
    io::{self, Seek, SeekFrom, Write},
};

use crate::error::{GrokOzempicError, Result};

/// File magic: ASCII `GOZ1` (grok-ozempic container), little-endian.
///
/// The magic is the *format family* and stays `GOZ1`; the layout version is the
/// `u32` that follows it.
const OZ1_MAGIC: u32 = u32::from_le_bytes(*b"GOZ1");
/// Current layout version written by [`PackStreamWriter`]: 3, adding the applied
/// per-tensor threshold (GH #66) on top of v2's scale (GH #65). Readers still
/// accept 1 and 2; see [`crate::core::weight_pack_read`].
///
/// v2 was **not** redefined in place to carry the threshold, even though it had
/// not been released: it is reachable on `main`, so two different row widths
/// would both claim version 2. The version field's whole job is to gate the
/// layout.
pub const OZ1_VERSION: u32 = 3;

/// Sentinel closing every scale-bearing tensor row (v2 and v3), so a
/// writer/reader row-width disagreement fails immediately instead of misparsing
/// the following row (GH #65).
pub const OZ1_ROW_SENTINEL: u32 = 0x5CA1E021;

/// Reconstruction/provenance values stored in one tensor row.
///
/// A struct rather than three more `f32` parameters on
/// [`PackStreamWriter::write_tensor_data`]: they are mutually transposable at a
/// call site, and transposing `scale` with a threshold produces a pack that
/// verifies but dequantizes to nonsense.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorRowStats {
    /// `α*` for ternary; `1.0` for fp16.
    pub scale: f32,
    /// The applied `gif_threshold` **multiplier**; `0.0` for fp16.
    pub gif_threshold: f32,
    /// The absolute firing cut `gif_threshold × rms`; `0.0` for fp16.
    pub threshold_abs: f32,
}

impl TensorRowStats {
    /// Row values for an fp16 payload: identity scale, no GIF gate.
    pub const fn fp16() -> Self {
        Self {
            scale: 1.0,
            gif_threshold: 0.0,
            threshold_abs: 0.0,
        }
    }

    /// Row values for a ternary payload.
    pub const fn ternary(scale: f32, gif_threshold: f32, threshold_abs: f32) -> Self {
        Self {
            scale,
            gif_threshold,
            threshold_abs,
        }
    }
}

/// Tensor blob alignment in bytes.
pub const DATA_ALIGNMENT: u64 = 32;

const META_U32: u32 = 0;
const META_STR: u32 = 1;

/// Packed tensor payload kinds (GOZ1 tensor table).
pub const TENSOR_F16: u32 = 0;
/// 2-bit ternary {-1,0,+1}, four values per byte (same packing as quantizer).
pub const TENSOR_TERNARY: u32 = 1;

/// Metadata value in the GOZ1 header.
#[derive(Debug, Clone)]
pub enum PackMetaValue {
    U32(u32),
    Str(String),
}

/// One row in the tensor table (payload written separately).
#[derive(Clone, Debug)]
pub struct PackTensorHeader {
    pub name: String,
    /// Row-major shape, slowest index first.
    pub shape: Vec<u64>,
    pub tensor_type: u32,
}

/// Streams a GOZ1 file without buffering all tensor payloads in RAM.
pub struct PackStreamWriter<'a, W: Write + Seek> {
    writer: &'a mut W,
    tensor_count: usize,
    tensors_written: usize,
    tensor_types: Vec<u32>,
    offset_field_positions: Vec<u64>,
    real_offsets: Vec<u64>,
    /// Scale/threshold fields are patched exactly like offsets, and for the same
    /// reason: the whole tensor table is written before any payload is quantized,
    /// so none of them is known when its row is laid down. One position per row,
    /// pointing at the first of the three consecutive `f32`s.
    scale_field_positions: Vec<u64>,
    real_stats: Vec<TensorRowStats>,
    data_section_start: u64,
}

impl<'a, W: Write + Seek> PackStreamWriter<'a, W> {
    pub fn begin(
        writer: &'a mut W,
        metadata: &BTreeMap<String, PackMetaValue>,
        tensor_headers: &[PackTensorHeader],
    ) -> Result<Self> {
        write_u32(writer, OZ1_MAGIC)?;
        write_u32(writer, OZ1_VERSION)?;
        write_u64(writer, tensor_headers.len() as u64)?;
        write_u64(writer, metadata.len() as u64)?;

        for (key, value) in metadata {
            write_str(writer, key)?;
            match value {
                PackMetaValue::U32(v) => {
                    write_u32(writer, META_U32)?;
                    write_u32(writer, *v)?;
                }
                PackMetaValue::Str(s) => {
                    write_u32(writer, META_STR)?;
                    write_str(writer, s)?;
                }
            }
        }

        let mut offset_field_positions: Vec<u64> = Vec::with_capacity(tensor_headers.len());
        let mut scale_field_positions: Vec<u64> = Vec::with_capacity(tensor_headers.len());
        let mut tensor_types: Vec<u32> = Vec::with_capacity(tensor_headers.len());
        for entry in tensor_headers {
            write_str(writer, &entry.name)?;
            write_u32(writer, entry.shape.len() as u32)?;
            for &dim in &entry.shape {
                write_u64(writer, dim)?;
            }
            write_u32(writer, entry.tensor_type)?;
            tensor_types.push(entry.tensor_type);
            offset_field_positions.push(writer.stream_position().map_err(GrokOzempicError::Io)?);
            write_u64(writer, 0u64)?;
            // Scale + threshold placeholders. NaN rather than 0.0 so a writer that
            // somehow finalizes without supplying real values produces a pack that
            // `verify_pack_file` rejects, instead of one that silently
            // reconstructs every ternary weight as zero (or claims τ = 0, which
            // would read as "nothing was silenced").
            scale_field_positions.push(writer.stream_position().map_err(GrokOzempicError::Io)?);
            write_f32(writer, f32::NAN)?; // scale
            write_f32(writer, f32::NAN)?; // gif_threshold
            write_f32(writer, f32::NAN)?; // threshold_abs
            write_u32(writer, OZ1_ROW_SENTINEL)?;
        }

        let header_end = writer.stream_position().map_err(GrokOzempicError::Io)?;
        let padding_needed = (DATA_ALIGNMENT - (header_end % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
        writer
            .write_all(&vec![0u8; padding_needed as usize])
            .map_err(GrokOzempicError::Io)?;

        let data_section_start = writer.stream_position().map_err(GrokOzempicError::Io)?;

        Ok(Self {
            writer,
            tensor_count: tensor_headers.len(),
            tensors_written: 0,
            tensor_types,
            offset_field_positions,
            real_offsets: Vec::with_capacity(tensor_headers.len()),
            scale_field_positions,
            real_stats: Vec::with_capacity(tensor_headers.len()),
            data_section_start,
        })
    }

    fn validate_stats_f16(idx: usize, stats: &TensorRowStats) -> Result<()> {
        // The halves *are* the values, so anything but the identity means the
        // row and the payload disagree about what the tensor holds. Thresholds
        // must be zero because no GIF gate ran: a nonzero one here would read
        // as a silencing cut that never happened.
        if !stats.scale.is_finite() || stats.scale != 1.0 {
            return Err(GrokOzempicError::PackWrite(format!(
                "write_tensor_data: tensor {idx} is fp16 and must have scale 1.0, got {}",
                stats.scale
            )));
        }
        if stats.gif_threshold != 0.0 || stats.threshold_abs != 0.0 {
            return Err(GrokOzempicError::PackWrite(format!(
                "write_tensor_data: tensor {idx} is fp16 and must have zero thresholds, got \
                 gif_threshold={} threshold_abs={}",
                stats.gif_threshold, stats.threshold_abs
            )));
        }
        Ok(())
    }

    fn validate_stats_ternary(idx: usize, stats: &TensorRowStats) -> Result<()> {
        // alpha is a magnitude -- sign lives in the trit -- so a negative
        // alpha would invert every weight in the tensor.
        if !stats.scale.is_finite() || stats.scale < 0.0 {
            return Err(GrokOzempicError::PackWrite(format!(
                "write_tensor_data: tensor {idx} has non-finite or negative scale {}; a pack \
                 must be dequantizable from its own contents",
                stats.scale
            )));
        }
        // Both thresholds are magnitudes compared against |w|; negative is
        // meaningless and would describe a gate that silenced nothing.
        if !stats.gif_threshold.is_finite() || stats.gif_threshold < 0.0 {
            return Err(GrokOzempicError::PackWrite(format!(
                "write_tensor_data: tensor {idx} has non-finite or negative gif_threshold {}",
                stats.gif_threshold
            )));
        }
        if !stats.threshold_abs.is_finite() || stats.threshold_abs < 0.0 {
            return Err(GrokOzempicError::PackWrite(format!(
                "write_tensor_data: tensor {idx} has non-finite or negative threshold_abs {}",
                stats.threshold_abs
            )));
        }
        // RM-252: threshold_abs = gif_threshold × rms. A zero multiplier with a
        // non-zero absolute cut is impossible; reject at write so packs cannot
        // pass the writer and fail verify_pack_file. The explicit (0, 0) pair is
        // allowed (dense ternary, no GIF sparsification).
        if stats.gif_threshold == 0.0 && stats.threshold_abs != 0.0 {
            return Err(GrokOzempicError::PackWrite(format!(
                "write_tensor_data: tensor {idx} has inconsistent thresholds \
                 (gif_threshold={}, threshold_abs={}); non-zero absolute cut with \
                 zero multiplier per RM-252",
                stats.gif_threshold, stats.threshold_abs
            )));
        }
        Ok(())
    }

    fn validate_stats(t_type: u32, idx: usize, stats: &TensorRowStats) -> Result<()> {
        match t_type {
            TENSOR_F16 => Self::validate_stats_f16(idx, stats),
            TENSOR_TERNARY => Self::validate_stats_ternary(idx, stats),
            _ => Ok(()),
        }
    }

    /// Write one tensor payload and record its reconstruction scale.
    ///
    /// `stats` carries the per-tensor reconstruction scale and the applied
    /// thresholds. `scale` is `α*` from
    /// [`crate::core::quantizer::QuantizedTensor::scale`] for a ternary payload,
    /// and `1.0` for an fp16 payload, so that `value = scale × payload` holds
    /// uniformly. It must be finite: `verify_pack_file` rejects a non-finite
    /// scale on a ternary tensor, and catching it at write time names the tensor
    /// while we still know it.
    pub fn write_tensor_data(&mut self, data: &[u8], stats: TensorRowStats) -> Result<()> {
        if self.tensors_written >= self.tensor_count {
            return Err(GrokOzempicError::PackWrite(
                "write_tensor_data: more blobs than tensor headers".into(),
            ));
        }
        Self::validate_stats(
            self.tensor_types[self.tensors_written],
            self.tensors_written,
            &stats,
        )?;
        self.real_stats.push(stats);
        let pos = self
            .writer
            .stream_position()
            .map_err(GrokOzempicError::Io)?;
        self.real_offsets.push(pos - self.data_section_start);
        self.writer.write_all(data).map_err(GrokOzempicError::Io)?;
        let cur = self
            .writer
            .stream_position()
            .map_err(GrokOzempicError::Io)?;
        let pad = (DATA_ALIGNMENT - (cur % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
        self.writer
            .write_all(&vec![0u8; pad as usize])
            .map_err(GrokOzempicError::Io)?;
        self.tensors_written += 1;
        Ok(())
    }

    pub fn finalize(self) -> Result<()> {
        if self.tensors_written != self.tensor_count {
            return Err(GrokOzempicError::PackWrite(format!(
                "finalize: expected {} tensor blobs, got {}",
                self.tensor_count, self.tensors_written
            )));
        }
        if self.real_offsets.len() != self.offset_field_positions.len()
            || self.real_stats.len() != self.scale_field_positions.len()
        {
            return Err(GrokOzempicError::PackWrite(
                "internal: offset/scale bookkeeping mismatch".into(),
            ));
        }
        for (offset_pos, real_offset) in self
            .offset_field_positions
            .iter()
            .zip(self.real_offsets.iter())
        {
            self.writer
                .seek(SeekFrom::Start(*offset_pos))
                .map_err(GrokOzempicError::Io)?;
            write_u64(self.writer, *real_offset)?;
        }
        // The three f32s are consecutive, so one seek per row patches all of
        // them; the sentinel that follows is already correct and untouched.
        for (scale_pos, stats) in self
            .scale_field_positions
            .iter()
            .zip(self.real_stats.iter())
        {
            self.writer
                .seek(SeekFrom::Start(*scale_pos))
                .map_err(GrokOzempicError::Io)?;
            write_f32(self.writer, stats.scale)?;
            write_f32(self.writer, stats.gif_threshold)?;
            write_f32(self.writer, stats.threshold_abs)?;
        }
        self.writer
            .seek(SeekFrom::End(0))
            .map_err(GrokOzempicError::Io)?;
        Ok(())
    }
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_metadata() -> BTreeMap<String, PackMetaValue> {
        let mut m = BTreeMap::new();
        m.insert(
            "oz.name".into(),
            PackMetaValue::Str("grok-ozempic-test".into()),
        );
        m.insert("oz.quantization_version".into(), PackMetaValue::U32(1));
        m
    }

    #[test]
    fn stream_writer_magic_and_version() {
        let headers = vec![PackTensorHeader {
            name: "blk.0.ffn_gate.weight".into(),
            shape: vec![64, 32],
            tensor_type: TENSOR_TERNARY,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[0xAB; 64], TensorRowStats::ternary(0.25, 0.6, 0.15))
                .unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        assert_eq!(&bytes[0..4], b"GOZ1");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, OZ1_VERSION);
    }

    #[test]
    fn stream_writer_rejects_ternary_zero_gif_nonzero_abs() {
        // RM-252: writer must refuse the pair verify_pack_file already rejects.
        let headers = vec![PackTensorHeader {
            name: "t".into(),
            shape: vec![1],
            tensor_type: TENSOR_TERNARY,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
        let err = w
            .write_tensor_data(&[0u8; 1], TensorRowStats::ternary(0.25, 0.0, 0.15))
            .expect_err("must reject gif_threshold=0 with threshold_abs!=0");
        let msg = err.to_string();
        assert!(
            msg.contains("inconsistent thresholds") || msg.contains("RM-252"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn stream_writer_allows_ternary_zero_gif_zero_abs() {
        let headers = vec![PackTensorHeader {
            name: "t".into(),
            shape: vec![1],
            tensor_type: TENSOR_TERNARY,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[0u8; 1], TensorRowStats::ternary(0.25, 0.0, 0.0))
                .expect("dense ternary (0,0) is allowed");
            w.finalize().unwrap();
        }
        assert_eq!(&buf.into_inner()[0..4], b"GOZ1");
    }

    #[test]
    fn stream_writer_tensor_count() {
        let headers = vec![PackTensorHeader {
            name: "t".into(),
            shape: vec![1],
            tensor_type: TENSOR_F16,
        }];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[0u8; 2], TensorRowStats::fp16())
                .unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 1);
    }

    #[test]
    fn stream_writer_two_tensors() {
        let headers = vec![
            PackTensorHeader {
                name: "a".into(),
                shape: vec![2],
                tensor_type: TENSOR_TERNARY,
            },
            PackTensorHeader {
                name: "b".into(),
                shape: vec![4],
                tensor_type: TENSOR_F16,
            },
        ];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.write_tensor_data(&[1, 2], TensorRowStats::ternary(0.5, 0.4, 0.2))
                .unwrap();
            w.write_tensor_data(&[0u8; 8], TensorRowStats::fp16())
                .unwrap();
            w.finalize().unwrap();
        }
        let len = buf.into_inner().len() as u64;
        assert_eq!(len % DATA_ALIGNMENT, 0);
    }

    #[test]
    fn stream_writer_empty_tensors() {
        let headers: Vec<PackTensorHeader> = vec![];
        let meta = sample_metadata();
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
            w.finalize().unwrap();
        }
        let bytes = buf.into_inner();
        assert_eq!(&bytes[0..4], b"GOZ1");
        let tc = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(tc, 0);
    }
}

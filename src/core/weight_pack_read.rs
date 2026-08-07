//! Verifier for **GOZ1** packed checkpoints (see [`crate::core::weight_pack`]).
//!
//! Parses header + tensor table and checks blob layout against file size.
//!
//! # Version policy (GH #65)
//!
//! | Version | Accepted | Scale |
//! |---------|----------|-------|
//! | 1 | yes, read-only legacy | absent — consumers must fall back to an oracle α derived from the source weights, and must record that fallback in provenance |
//! | 2 | yes, written by this crate | present per tensor; required finite on every ternary tensor |
//! | other | **rejected** | — |
//!
//! Unknown versions are refused rather than parsed on the assumption that the
//! prefix is compatible: a future layout could reorder the tensor row, and
//! silently misreading offsets would produce a plausible report for a pack we do
//! not understand.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::{
    core::weight_pack::{DATA_ALIGNMENT, OZ1_VERSION, TENSOR_F16, TENSOR_TERNARY},
    error::{GrokOzempicError, Result},
};

const OZ1_MAGIC: u32 = u32::from_le_bytes(*b"GOZ1");

/// First version carrying a per-tensor scale in the tensor row.
pub const OZ1_VERSION_SCALED: u32 = 2;

const META_U32: u32 = 0;
const META_STR: u32 = 1;

#[derive(Debug, Clone)]
pub struct PackVerifyReport {
    pub version: u32,
    pub tensor_count: usize,
    pub metadata_keys: Vec<String>,
    pub tensor_names: Vec<String>,
    pub file_size: u64,
    /// Absolute file offset where the payload section begins.
    ///
    /// A tensor's payload lives at `data_section_start + data_offset`. Without
    /// this a caller cannot locate a blob from the report alone, which is
    /// exactly what "dequantizable from its own contents" requires.
    pub data_section_start: u64,
    /// Per-tensor payload offsets relative to [`Self::data_section_start`], in
    /// tensor-table order.
    pub data_offsets: Vec<u64>,
    /// Per-tensor reconstruction scales, in tensor-table order.
    ///
    /// `None` for a v1 pack, which carries no scales at all — distinct from
    /// `Some(vec![])` for a v2 pack with no tensors, so a consumer can tell
    /// "this format has no scales" from "this pack has no tensors".
    pub scales: Option<Vec<f32>>,
}

pub fn verify_pack_file(path: &Path) -> Result<PackVerifyReport> {
    let mut f = File::open(path).map_err(GrokOzempicError::Io)?;
    let file_size = f.metadata().map_err(GrokOzempicError::Io)?.len();

    let magic = read_u32(&mut f)?;
    if magic != OZ1_MAGIC {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "GOZ1 verify: bad magic (expected GOZ1), got {magic:#x}"
        )));
    }
    let version = read_u32(&mut f)?;
    if version == 0 || version > OZ1_VERSION {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "GOZ1 verify: unsupported container version {version} (this build understands 1..={OZ1_VERSION})"
        )));
    }
    let has_scales = version >= OZ1_VERSION_SCALED;
    let tensor_count = read_u64(&mut f)? as usize;
    let meta_count = read_u64(&mut f)? as usize;

    let mut metadata_keys = Vec::with_capacity(meta_count);
    for _ in 0..meta_count {
        let key = read_str(&mut f)?;
        metadata_keys.push(key.clone());
        let vtype = read_u32(&mut f)?;
        skip_meta_value(&mut f, vtype)?;
    }

    let mut tensor_names = Vec::with_capacity(tensor_count);
    let mut infos: Vec<TensorInfoParsed> = Vec::with_capacity(tensor_count);

    for _ in 0..tensor_count {
        let name = read_str(&mut f)?;
        tensor_names.push(name.clone());
        let ndim = read_u32(&mut f)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(&mut f)?);
        }
        let tensor_type = read_u32(&mut f)?;
        let offset = read_u64(&mut f)?;
        // The v2 row appends the scale, so v1 parsing is the same code minus
        // this read; nothing else about the row moved.
        let scale = if has_scales {
            Some(read_f32(&mut f)?)
        } else {
            None
        };
        infos.push(TensorInfoParsed {
            name,
            shape,
            tensor_type,
            data_offset: offset,
            scale,
        });
    }

    let header_end = f.stream_position().map_err(GrokOzempicError::Io)?;
    let padding_needed = (DATA_ALIGNMENT - (header_end % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
    f.seek(SeekFrom::Current(padding_needed as i64))
        .map_err(GrokOzempicError::Io)?;
    let data_section_start = f.stream_position().map_err(GrokOzempicError::Io)?;

    let mut expected_rel: u64 = 0;
    for (i, info) in infos.iter().enumerate() {
        if info.data_offset != expected_rel {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "GOZ1 verify: tensor {} ({}) data_offset {} != expected cumulative {}",
                i, info.name, info.data_offset, expected_rel
            )));
        }
        // A ternary payload is sign-only, so without a finite scale the tensor
        // cannot be reconstructed at all. Checked for ternary specifically: an
        // fp16 payload is self-describing and survives a junk scale, but a
        // ternary one silently dequantizes to garbage (or to all-zero).
        if let Some(scale) = info.scale
            && info.tensor_type == TENSOR_TERNARY
            && (!scale.is_finite() || *scale < 0.0)
        {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "GOZ1 verify: tensor {} ({}) is ternary with non-finite or negative scale {}; the pack \
                 cannot be dequantized from its own contents",
                i, info.name, scale
            )));
        }
        let nbytes = tensor_nbytes(info)?;
        let abs = data_section_start + info.data_offset;
        let end = abs.saturating_add(nbytes);
        if end > file_size {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "GOZ1 verify: tensor {} ({}) blob end {} exceeds file size {}",
                i, info.name, end, file_size
            )));
        }
        expected_rel = expected_rel.saturating_add(align_up(nbytes, DATA_ALIGNMENT));
    }

    if data_section_start + expected_rel > file_size {
        return Err(GrokOzempicError::InvalidConfig(format!(
            "GOZ1 verify: declared tensor payloads need {} bytes past data section start, file has {}",
            expected_rel,
            file_size.saturating_sub(data_section_start)
        )));
    }

    let scales = has_scales.then(|| infos.iter().filter_map(|i| i.scale).collect());

    Ok(PackVerifyReport {
        version,
        tensor_count,
        metadata_keys,
        tensor_names,
        file_size,
        data_section_start,
        data_offsets: infos.iter().map(|i| i.data_offset).collect(),
        scales,
    })
}

struct TensorInfoParsed {
    name: String,
    shape: Vec<u64>,
    tensor_type: u32,
    data_offset: u64,
    /// `None` on a v1 pack, which has no scale field.
    scale: Option<f32>,
}

fn tensor_nbytes(info: &TensorInfoParsed) -> Result<u64> {
    let n: u64 = info.shape.iter().product();
    match info.tensor_type {
        TENSOR_F16 => n.checked_mul(2).ok_or_else(|| {
            GrokOzempicError::InvalidConfig("GOZ1 verify: shape overflow (f16)".into())
        }),
        TENSOR_TERNARY => Ok(n.div_ceil(4)),
        other => Err(GrokOzempicError::InvalidConfig(format!(
            "GOZ1 verify: unknown tensor type {other} (cannot compute size)"
        ))),
    }
}

fn align_up(n: u64, align: u64) -> u64 {
    n.div_ceil(align) * align
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(GrokOzempicError::Io)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(GrokOzempicError::Io)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(GrokOzempicError::Io)?;
    Ok(f32::from_le_bytes(b))
}

fn read_str<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(GrokOzempicError::Io)?;
    String::from_utf8(buf)
        .map_err(|e| GrokOzempicError::InvalidConfig(format!("GOZ1: invalid utf8: {e}")))
}

fn skip_meta_value<R: Read>(r: &mut R, vtype: u32) -> Result<()> {
    match vtype {
        META_U32 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(GrokOzempicError::Io)?;
        }
        META_STR => {
            let _s = read_str(r)?;
        }
        _ => {
            return Err(GrokOzempicError::InvalidConfig(format!(
                "GOZ1 verify: unsupported metadata value type {vtype}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::{BufWriter, Write};

    use crate::core::weight_pack::{
        PackMetaValue, PackStreamWriter, PackTensorHeader, TENSOR_TERNARY,
    };

    #[test]
    fn verify_round_trip_stream_writer() {
        let dir = std::env::temp_dir();
        let path = dir.join("grok_ozempic_verify_test.goz1");
        let _ = std::fs::remove_file(&path);

        let mut meta = BTreeMap::new();
        meta.insert("oz.name".into(), PackMetaValue::Str("t".into()));
        let headers = vec![PackTensorHeader {
            name: "w".into(),
            shape: vec![8],
            tensor_type: TENSOR_TERNARY,
        }];
        let f = File::create(&path).unwrap();
        let mut bw = BufWriter::new(f);
        {
            let mut w = PackStreamWriter::begin(&mut bw, &meta, &headers).unwrap();
            w.write_tensor_data(&[0xFF; 2], 0.75).unwrap();
            w.finalize().unwrap();
        }
        bw.flush().unwrap();
        drop(bw);

        let report = verify_pack_file(&path).unwrap();
        assert_eq!(report.tensor_count, 1);
        assert_eq!(report.version, OZ1_VERSION);
        assert!(report.metadata_keys.contains(&"oz.name".to_string()));
        assert_eq!(report.scales.as_deref(), Some(&[0.75f32][..]));
        let _ = std::fs::remove_file(&path);
    }

    use crate::core::quantizer::{decode_trit, quantize_f32};

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("grok_ozempic_{tag}_{}.goz1", std::process::id()))
    }

    fn write_pack(path: &Path, headers: &[PackTensorHeader], blobs: &[(&[u8], f32)]) {
        let mut meta = BTreeMap::new();
        meta.insert("oz.name".into(), PackMetaValue::Str("t".into()));
        let mut bw = BufWriter::new(File::create(path).unwrap());
        {
            let mut w = PackStreamWriter::begin(&mut bw, &meta, headers).unwrap();
            for (data, scale) in blobs {
                w.write_tensor_data(data, *scale).unwrap();
            }
            w.finalize().unwrap();
        }
        bw.flush().unwrap();
    }

    /// The #65 acceptance criterion: reconstruct with **no access to the source
    /// weights**, using only bytes that came out of the file.
    #[test]
    fn pack_alone_reconstructs_without_the_source_weights() {
        let weights: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.21).cos() * 2.0).collect();
        let qt = quantize_f32(&weights, 0.6);
        let in_memory: Vec<f32> = (0..qt.num_elements)
            .map(|i| qt.scale * decode_trit((qt.packed[i / 4] >> ((i % 4) * 2)) & 0b11))
            .collect();

        let path = temp_path("packonly");
        let _ = std::fs::remove_file(&path);
        let headers = vec![PackTensorHeader {
            name: "w".into(),
            shape: vec![64],
            tensor_type: TENSOR_TERNARY,
        }];
        write_pack(&path, &headers, &[(&qt.packed, qt.scale)]);

        // Everything below comes from the file only.
        let report = verify_pack_file(&path).unwrap();
        let scale = report.scales.as_ref().expect("v2 pack must carry scales")[0];
        let bytes = std::fs::read(&path).unwrap();
        let at = (report.data_section_start + report.data_offsets[0]) as usize;
        let payload = &bytes[at..at + qt.packed.len()];
        let from_pack: Vec<f32> = (0..weights.len())
            .map(|i| scale * decode_trit((payload[i / 4] >> ((i % 4) * 2)) & 0b11))
            .collect();

        assert_eq!(from_pack, in_memory, "pack-only reconstruction diverged");
        let _ = std::fs::remove_file(&path);
    }

    /// Strip the v2 scale fields to synthesize a legacy pack, proving the reader
    /// still parses one and reports the absence of scales rather than inventing
    /// them. Built by surgery on a real pack so the rest of the layout is exact.
    /// Build a v1 pack byte by byte.
    ///
    /// Deliberately *not* derived from the current writer: this is the
    /// compatibility test, so it must describe the legacy layout independently.
    /// (Deriving it by excising the v2 scale field also breaks — the header
    /// shrinks, the alignment padding shifts, and the payload no longer lands
    /// where the recomputed data section starts.)
    fn make_v1_pack(path: &Path) {
        let mut b: Vec<u8> = Vec::new();
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        b.extend_from_slice(b"GOZ1");
        b.extend_from_slice(&1u32.to_le_bytes()); // version 1
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&1u64.to_le_bytes()); // meta_count

        put_str(&mut b, "oz.name");
        b.extend_from_slice(&META_STR.to_le_bytes());
        put_str(&mut b, "t");

        // v1 tensor row: name, ndim, shape[], tensor_type, data_offset. No scale.
        put_str(&mut b, "w");
        b.extend_from_slice(&1u32.to_le_bytes()); // ndim
        b.extend_from_slice(&8u64.to_le_bytes()); // shape[0]
        b.extend_from_slice(&TENSOR_TERNARY.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // data_offset

        let pad = (DATA_ALIGNMENT - (b.len() as u64 % DATA_ALIGNMENT)) % DATA_ALIGNMENT;
        b.extend(std::iter::repeat_n(0u8, pad as usize));

        // 8 trits = 2 bytes, then padded to the alignment the verifier expects.
        b.extend_from_slice(&[0xE4u8; 2]);
        b.extend(std::iter::repeat_n(0u8, (DATA_ALIGNMENT - 2) as usize));

        std::fs::write(path, &b).unwrap();
    }

    #[test]
    fn version_1_pack_still_parses_and_reports_no_scales() {
        let path = temp_path("v1");
        let _ = std::fs::remove_file(&path);
        make_v1_pack(&path);
        let report = verify_pack_file(&path).unwrap();
        assert_eq!(report.version, 1);
        assert_eq!(report.tensor_count, 1);
        assert!(
            report.scales.is_none(),
            "a v1 pack has no scales; None is what tells a consumer to use the \
             legacy oracle path and tag its provenance"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_version_is_rejected_rather_than_parsed_as_a_prefix() {
        let path = temp_path("vfuture");
        let _ = std::fs::remove_file(&path);
        let headers = vec![PackTensorHeader {
            name: "w".into(),
            shape: vec![8],
            tensor_type: TENSOR_TERNARY,
        }];
        write_pack(&path, &headers, &[(&[0xE4u8; 2], 0.5)]);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&(OZ1_VERSION + 1).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let err = verify_pack_file(&path).unwrap_err().to_string();
        assert!(err.contains("unsupported container version"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn version_zero_is_rejected() {
        let path = temp_path("vzero");
        let _ = std::fs::remove_file(&path);
        let headers = vec![PackTensorHeader {
            name: "w".into(),
            shape: vec![8],
            tensor_type: TENSOR_TERNARY,
        }];
        write_pack(&path, &headers, &[(&[0xE4u8; 2], 0.5)]);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(verify_pack_file(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn non_finite_scale_on_a_ternary_tensor_fails_verification() {
        // The writer refuses this, so the bytes are corrupted directly -- the
        // point is that a pack arriving from anywhere else cannot pass verify
        // while being undequantizable.
        let path = temp_path("nanscale");
        let _ = std::fs::remove_file(&path);
        let headers = vec![PackTensorHeader {
            name: "w".into(),
            shape: vec![8],
            tensor_type: TENSOR_TERNARY,
        }];
        write_pack(&path, &headers, &[(&[0xE4u8; 2], 0.5)]);
        let mut bytes = std::fs::read(&path).unwrap();
        let scale_at = bytes
            .windows(4)
            .position(|w| w == 0.5f32.to_le_bytes())
            .expect("scale field present in a v2 pack");
        bytes[scale_at..scale_at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let err = verify_pack_file(&path).unwrap_err().to_string();
        assert!(err.contains("non-finite scale"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_refuses_a_non_finite_scale() {
        let mut meta = BTreeMap::new();
        meta.insert("oz.name".into(), PackMetaValue::Str("t".into()));
        let headers = vec![PackTensorHeader {
            name: "w".into(),
            shape: vec![8],
            tensor_type: TENSOR_TERNARY,
        }];
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let mut w = PackStreamWriter::begin(&mut buf, &meta, &headers).unwrap();
        assert!(w.write_tensor_data(&[0xE4u8; 2], f32::INFINITY).is_err());
        assert!(w.write_tensor_data(&[0xE4u8; 2], f32::NAN).is_err());
    }
}

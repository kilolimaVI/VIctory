//! Regression for warming and reusing the same pixel shader under different host capabilities.

#![cfg(windows)]

use victory_gpu::shader_cache::{host_shader, use_shader_cache};
use victory_shader::{Container, DxilProgram, FourCC, emit};

fn signature(name: &str, system_value: u32, read_mask: u32) -> Vec<u8> {
    // One row at register 0. The name follows the eight-word element.
    let words: [u32; 10] = [1, 8, 0, 40, 0, system_value, 3, 0, 15 | read_mask << 8, 0];
    let mut bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    bytes.extend(name.as_bytes());
    bytes.push(0);
    bytes
}

fn shader_container(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let mut bytes = vec![0; 32 + 4 * parts.len()];
    bytes[..4].copy_from_slice(b"DXBC");
    bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
    bytes[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
    for (index, (tag, data)) in parts.iter().enumerate() {
        let slot = 32 + 4 * index;
        let offset = bytes.len() as u32;
        bytes[slot..slot + 4].copy_from_slice(&offset.to_le_bytes());
        bytes.extend(tag.0);
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
    }
    let size = bytes.len() as u32;
    bytes[24..28].copy_from_slice(&size.to_le_bytes());
    bytes
}

fn pixel_shader() -> Vec<u8> {
    // A straight-line pixel program that broadcasts the first lane's
    // position.x and writes it as a color. The code is generated here.
    let words: [u32; 5] = [
        0x7e02_0500, // v_readfirstlane_b32 s1, v0
        0x7e08_0201, // v_mov_b32 v4, s1
        0xf800_180f,
        0x0404_0404, // exp mrt0 v4, v4, v4, v4 done vm
        0xbf81_0000, // s_endpgm
    ];
    let code: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();

    let mut payload = vec![0u8; 0x68]; // FSSM stage-offset table
    payload[..8].copy_from_slice(&6u64.to_le_bytes());
    for offset in [0x08, 0x10, 0x18, 0x20, 0x28, 0x48, 0x60] {
        payload[offset..offset + 8].copy_from_slice(&0xdead_beef_dead_beefu64.to_le_bytes());
    }
    let record = payload.len();
    payload[0x30..0x38].copy_from_slice(&(record as u64).to_le_bytes()); // pixel slot
    let stage = record + 0x54;
    let patch_table = stage + 0x14c;
    let code_offset = patch_table + 8 * 404;
    payload.resize(code_offset, 0);
    payload[record..record + 8].copy_from_slice(&(code_offset as u64).to_le_bytes());
    payload[record + 8..record + 16].copy_from_slice(&(code.len() as u64).to_le_bytes());
    payload[record + 0x50..record + 0x54].copy_from_slice(&(28u32 | (31 << 20)).to_le_bytes());
    payload[stage + 0x134..stage + 0x138].copy_from_slice(&0x100u32.to_le_bytes());
    for mrt in 0..8 {
        let offset = patch_table + mrt * 404;
        payload[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    }
    payload.extend(code);

    let mut fssm = b"FSSM".to_vec();
    fssm.extend(0x101u32.to_le_bytes());
    fssm.extend((payload.len() as u64).to_le_bytes());
    fssm.extend(payload);
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    xbch.extend(3u32.to_le_bytes());
    xbch.extend((compressed.len() as u64).to_le_bytes());
    xbch.extend((fssm.len() as u64).to_le_bytes());
    xbch.extend(compressed);

    let mut dxil = vec![0; 0x18];
    dxil.extend(b"XDAT");
    for word in [1u32, 0x24, 0x0d, 0x30, xbch.len() as u32] {
        dxil.extend(word.to_le_bytes());
    }
    dxil.extend(xbch);
    shader_container(&[
        (FourCC::DXIL, &dxil),
        (FourCC::ISG1, &signature("SV_Position", 1, 15)),
        (FourCC::OSG1, &signature("SV_Target", 64, 0)),
    ])
}

/// This integration-test process has its own cache and helper-lane flag.
#[test]
fn cached_pixel_shaders_use_the_matching_helper_lane_setting() {
    let dir = tempfile::tempdir().unwrap();
    use_shader_cache(dir.path().to_path_buf()).expect("open shader cache");
    let input = pixel_shader();
    for (helper_lanes, expected_hit) in [(false, false), (true, false), (false, true), (true, true)]
    {
        emit::set_helper_lane_waves(helper_lanes);
        let expected = emit::translate_with(&input, emit::Options { view_marks: true }).unwrap();
        let (cached, hit) = host_shader(&input);
        assert_eq!(hit, expected_hit, "helper lanes: {helper_lanes}");
        let cached = cached.unwrap();
        let container = Container::parse(&cached.dxil).unwrap();
        let program = DxilProgram::parse(container.part(FourCC::DXIL).unwrap().data).unwrap();
        let expected_model = if helper_lanes { (6, 7) } else { (6, 6) };
        assert_eq!(
            program.shader_model, expected_model,
            "helper lanes: {helper_lanes}"
        );
        assert_eq!(cached, expected, "helper lanes: {helper_lanes}");
    }
}

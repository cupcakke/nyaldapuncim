const std = @import("std");

pub const Error = error{
    Truncated,
    ChecksumMismatch,
    MagicMismatch,
    VersionMismatch,
    EndiannessMismatch,
};

pub fn validate(data: []const u8, expected_magic: [8]u8, expected_version: u32) Error![]const u8 {
    if (data.len < expected_magic.len + @sizeOf(u32) * 3) return Error.Truncated;
    const payload = data[0 .. data.len - @sizeOf(u32)];
    const stored_checksum = std.mem.readInt(u32, data[data.len - 4 ..][0..4], .little);
    var checksum = std.hash.Crc32.init();
    checksum.update(payload);
    if (checksum.final() != stored_checksum) return Error.ChecksumMismatch;
    if (!std.mem.eql(u8, payload[0..expected_magic.len], expected_magic[0..])) return Error.MagicMismatch;
    const version_offset = expected_magic.len;
    if (std.mem.readInt(u32, payload[version_offset..][0..4], .little) != expected_version) return Error.VersionMismatch;
    if (std.mem.readInt(u32, payload[version_offset + 4 ..][0..4], .little) != 0x01020304) return Error.EndiannessMismatch;
    return payload;
}

fn makeEnvelope(allocator: std.mem.Allocator, magic: [8]u8, version: u32, body: []const u8) ![]u8 {
    var bytes = std.ArrayList(u8).init(allocator);
    errdefer bytes.deinit();
    try bytes.appendSlice(magic[0..]);
    try bytes.writer().writeInt(u32, version, .little);
    try bytes.writer().writeInt(u32, 0x01020304, .little);
    try bytes.appendSlice(body);
    var checksum = std.hash.Crc32.init();
    checksum.update(bytes.items);
    try bytes.writer().writeInt(u32, checksum.final(), .little);
    return bytes.toOwnedSlice();
}

test "checkpoint envelope validates checksum version and endianness" {
    const allocator = std.testing.allocator;
    const magic = [8]u8{ 'J', 'A', 'I', 'D', 'E', 'C', 'K', 'P' };
    const envelope = try makeEnvelope(allocator, magic, 7, &.{ 1, 2, 3, 4 });
    defer allocator.free(envelope);
    const payload = try validate(envelope, magic, 7);
    try std.testing.expectEqual(envelope.len - 4, payload.len);
}

test "checkpoint envelope rejects corruption and truncation" {
    const allocator = std.testing.allocator;
    const magic = [8]u8{ 'J', 'A', 'I', 'D', 'E', 'C', 'K', 'P' };
    const envelope = try makeEnvelope(allocator, magic, 7, &.{ 9, 8, 7 });
    defer allocator.free(envelope);
    envelope[envelope.len - 5] ^= 1;
    try std.testing.expectError(Error.ChecksumMismatch, validate(envelope, magic, 7));
    try std.testing.expectError(Error.Truncated, validate(envelope[0..12], magic, 7));
}

test "checkpoint envelope rejects incompatible metadata" {
    const allocator = std.testing.allocator;
    const magic = [8]u8{ 'J', 'A', 'I', 'D', 'E', 'C', 'K', 'P' };
    const envelope = try makeEnvelope(allocator, magic, 7, &.{});
    defer allocator.free(envelope);
    try std.testing.expectError(Error.VersionMismatch, validate(envelope, magic, 6));
    var wrong_magic = magic;
    wrong_magic[0] = 'X';
    try std.testing.expectError(Error.MagicMismatch, validate(envelope, wrong_magic, 7));
}

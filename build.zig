const std = @import("std");
const path = std.fs.path;
const Builder = std.build.Builder;

pub fn build(b: *Builder) !void {
    const exe = b.addExecutable("triangle", "src/main.zig");

    exe.setBuildMode(b.standardReleaseOptions());

    exe.linkSystemLibrary("c");
    exe.linkSystemLibrary("glfw");
    exe.linkSystemLibrary("vulkan");

    exe.install();

    // Shaders
    try b.makePath(try path.join(b.allocator, &[_][]const u8{ b.cache_root, "s" }));
    const opt = switch (exe.build_mode) {
        .Debug => "-O0",
        .ReleaseSafe, .ReleaseFast => "-O",
        .ReleaseSmall => "-Os",
    };

    try buildShader(b, opt, "triangle.vert");
    try buildShader(b, opt, "triangle.frag");
}

fn buildShader(b: *Builder, optim_lvl: []const u8, comptime shader: []const u8) !void {
    const output = try path.join(b.allocator, &[_][]const u8{
        b.cache_root, "s", shader ++ ".spv",
    });

    // TODO Recompile on modification only
    const cmd = b.addSystemCommand(&[_][]const u8{
        "glslc", optim_lvl, "-o", output, "shaders/" ++ shader,
    });

    b.default_step.dependOn(&cmd.step);
    b.installBinFile(output, "shaders/" ++ shader ++ ".spv");
}

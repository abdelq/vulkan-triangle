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

    try b.makePath(try path.join(b.allocator, &[_][]const u8{ b.cache_root, "s" }));
    try buildShader(b, "triangle.vert");
    try buildShader(b, "triangle.frag");
}

fn buildShader(b: *Builder, comptime shader: []const u8) !void {
    const output = try path.join(b.allocator, &[_][]const u8{
        b.cache_root, "s", shader ++ ".spv",
    });
    // TODO Evaluate only once with exe.build_mode
    const optim_lvl = switch (b.release_mode.?) {
        .Debug => "-O0",
        .ReleaseSafe, .ReleaseFast => "-O",
        .ReleaseSmall => "-Os",
    };

    // TODO Recompile on modification only
    const cmd = b.addSystemCommand(&[_][]const u8{
        "glslc", optim_lvl, "-o", output, "shaders/" ++ shader,
    });

    b.default_step.dependOn(&cmd.step);
    b.installBinFile(output, "shaders/" ++ shader ++ ".spv");
}

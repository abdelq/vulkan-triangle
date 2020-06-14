const std = @import("std");
const glfw = @import("glfw.zig");
const vk = @import("vulkan.zig");

const width = 1920;
const height = 1080;

fn errorCallback(error_code: c_int, description: [*c]const u8) callconv(.C) void {
    std.debug.warn("GLFW: {s}\n", .{description});
}

fn resizeCallback(window: ?*glfw.Window, w: c_int, h: c_int) callconv(.C) void {
    vk.framebuffer_resized = true;
}

pub fn main() !void {
    if (std.builtin.mode == .Debug) {
        _ = glfw.setErrorCallback(errorCallback);
    }

    if (!glfw.init())
        return error.GlfwInitFailed;
    defer glfw.terminate();

    if (!glfw.vulkanSupported())
        return error.vkNotSupported;

    glfw.windowHint(glfw.client_api, glfw.no_api);

    const window = glfw.createWindow(width, height, "Triangle", null, null) orelse
        return error.GlfwCreateWindowFailed;
    defer glfw.destroyWindow(window);

    _ = glfw.setFramebufferSizeCallback(window, resizeCallback);

    try vk.init(std.heap.c_allocator, window);
    defer vk.cleanup();

    while (!glfw.windowShouldClose(window)) {
        glfw.pollEvents();
        try vk.drawFrame();
    }
}

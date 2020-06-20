usingnamespace @import("c.zig");

// Constants
pub const client_api = GLFW_CLIENT_API;
pub const no_api = GLFW_NO_API;

// Structures
pub const Window = GLFWwindow;

// Functions
pub const createWindow = glfwCreateWindow;
pub const destroyWindow = glfwDestroyWindow;
pub const getFramebufferSize = glfwGetFramebufferSize;
pub const getRequiredInstanceExtensions = glfwGetRequiredInstanceExtensions;
pub const pollEvents = glfwPollEvents;
pub const setErrorCallback = glfwSetErrorCallback;
pub const setFramebufferSizeCallback = glfwSetFramebufferSizeCallback;
pub const terminate = glfwTerminate;
pub const waitEvents = glfwWaitEvents;
pub const windowHint = glfwWindowHint;

pub inline fn init() bool {
    return glfwInit() != GLFW_FALSE;
}

pub inline fn vulkanSupported() bool {
    return glfwVulkanSupported() != GLFW_FALSE;
}

pub inline fn windowShouldClose(window: *Window) bool {
    return glfwWindowShouldClose(window) != GLFW_FALSE;
}

pub inline fn createWindowSurface(
    instance: VkInstance,
    window: *Window,
    allocator: ?*const VkAllocationCallbacks,
    surface: *VkSurfaceKHR,
) !void {
    return switch (glfwCreateWindowSurface(instance, window, allocator, surface)) {
        .VK_SUCCESS => {},
        .VK_ERROR_EXTENSION_NOT_PRESENT => error.ExtensionNotPresent,
        .VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        .VK_ERROR_NATIVE_WINDOW_IN_USE_KHR => error.NativeWindowInUse,
        else => unreachable,
    };
}

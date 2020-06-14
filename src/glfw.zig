const c = @import("c.zig");

pub const Window = c.GLFWwindow;

pub const client_api = c.GLFW_CLIENT_API;
pub const no_api = c.GLFW_NO_API;

pub const setErrorCallback = c.glfwSetErrorCallback;
pub const setFramebufferSizeCallback = c.glfwSetFramebufferSizeCallback;

pub fn init() bool {
    return c.glfwInit() != c.GLFW_FALSE;
}
pub const terminate = c.glfwTerminate;

pub fn vulkanSupported() bool {
    return c.glfwVulkanSupported() != c.GLFW_FALSE;
}

pub const windowHint = c.glfwWindowHint;

pub const createWindow = c.glfwCreateWindow;
pub const destroyWindow = c.glfwDestroyWindow;

pub fn windowShouldClose(window: *c.GLFWwindow) bool {
    return c.glfwWindowShouldClose(window) != c.GLFW_FALSE;
}

pub const pollEvents = c.glfwPollEvents;

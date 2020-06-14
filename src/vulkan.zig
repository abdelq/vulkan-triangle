// TODO Naming Convention + variable names
// TODO View might be actually flipped (e.g. negative coords as positive coords
// & positive coords as negative coords)
const std = @import("std");
const builtin = @import("builtin");
const c = @import("c.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const WIDTH = 1920;
const HEIGHT = 1080;

const max_frames_in_flight = 2;
var currentFrame: u32 = 0;
pub var framebuffer_resized = false;

var instance: c.VkInstance = undefined;
var surface: c.VkSurfaceKHR = undefined;
var callback: c.VkDebugReportCallbackEXT = undefined;
var physicalDevice: c.VkPhysicalDevice = undefined;
var device: c.VkDevice = undefined;
var graphicsQueue: c.VkQueue = undefined;
var presentQueue: c.VkQueue = undefined;
var swapChainImages: []c.VkImage = undefined;
var swapChain: c.VkSwapchainKHR = undefined;
var swapChainImageFormat: c.VkFormat = undefined;
var swapChainExtent: c.VkExtent2D = undefined;
var imageViews: []c.VkImageView = undefined;
var renderPass: c.VkRenderPass = undefined;
var pipelineLayout: c.VkPipelineLayout = undefined;
var pipeline: c.VkPipeline = undefined;
var framebuffers: []c.VkFramebuffer = undefined;
var commandPool: c.VkCommandPool = undefined;
var commandBuffers: []c.VkCommandBuffer = undefined;
var debugMessenger: c.VkDebugUtilsMessengerEXT = undefined; // FIXME

var imageAvailableSemaphores: [max_frames_in_flight]c.VkSemaphore = undefined;
var renderFinishedSemaphores: [max_frames_in_flight]c.VkSemaphore = undefined;
var inFlightFences: [max_frames_in_flight]c.VkFence = undefined;

const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
const device_extensions = [_][*:0]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};

const QueueFamilyIndices = struct {
    graphicsFamily: ?u32,
    presentFamily: ?u32,

    fn init() QueueFamilyIndices {
        return QueueFamilyIndices{
            .graphicsFamily = null,
            .presentFamily = null,
        };
    }

    fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphicsFamily != null and self.presentFamily != null;
    }
};

const SwapChainSupportDetails = struct {
    capabilities: c.VkSurfaceCapabilitiesKHR,
    formats: ArrayList(c.VkSurfaceFormatKHR),
    presentModes: ArrayList(c.VkPresentModeKHR),

    fn init(allocator: *Allocator) SwapChainSupportDetails {
        var result = SwapChainSupportDetails{
            .capabilities = std.mem.zeroes(c.VkSurfaceCapabilitiesKHR),
            .formats = ArrayList(c.VkSurfaceFormatKHR).init(allocator),
            .presentModes = ArrayList(c.VkPresentModeKHR).init(allocator),
        };
        // const slice = @sliceToBytes((*[1]c.VkSurfaceCapabilitiesKHR)(&result.capabilities)[0..1]);
        // std.mem.set(u8, slice, 0);
        return result;
    }

    fn deinit(self: *SwapChainSupportDetails) void {
        self.formats.deinit();
        self.presentModes.deinit();
    }
};

const VulkanError = error{
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
    DeviceLost,
    MemoryMapFailed,
    LayerNotPresent,
    ExtensionNotPresent,
    FeatureNotPresent,
    IncompatibleDriver,
    TooManyObjects,
    FormatNotSupported,
    FragmentedPool,
    Unknown,
    OutOfPoolMemory,
    InvalidExternalHandle,
    Fragmentation,
    InvalidOpaqueCaptureAddress,
    SurfaceLost,
    NativeWindowInUse,
    OutOfDate,
    IncompatibleDisplay,
    ValidationFailed,
    InvalidShader,
    IncompatibleVersion,
    InvalidDRMFormatModifierPlaneLayout,
    NotPermitted,
    FullScreenExclusiveModeLost,
};

fn checkResult(result: c.VkResult) VulkanError!void {
    return switch (result) {
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        .VK_ERROR_MEMORY_MAP_FAILED => error.MemoryMapFailed,
        .VK_ERROR_LAYER_NOT_PRESENT => error.LayerNotPresent,
        .VK_ERROR_EXTENSION_NOT_PRESENT => error.ExtensionNotPresent,
        .VK_ERROR_FEATURE_NOT_PRESENT => error.FeatureNotPresent,
        .VK_ERROR_INCOMPATIBLE_DRIVER => error.IncompatibleDriver,
        .VK_ERROR_TOO_MANY_OBJECTS => error.TooManyObjects,
        .VK_ERROR_FORMAT_NOT_SUPPORTED => error.FormatNotSupported,
        .VK_ERROR_FRAGMENTED_POOL => error.FragmentedPool,
        .VK_ERROR_UNKNOWN => error.Unknown,
        .VK_ERROR_OUT_OF_POOL_MEMORY => error.OutOfPoolMemory,
        .VK_ERROR_INVALID_EXTERNAL_HANDLE => error.InvalidExternalHandle,
        .VK_ERROR_FRAGMENTATION => error.Fragmentation,
        .VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS => error.InvalidOpaqueCaptureAddress,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        .VK_ERROR_NATIVE_WINDOW_IN_USE_KHR => error.NativeWindowInUse,
        .VK_ERROR_OUT_OF_DATE_KHR => error.OutOfDate,
        .VK_ERROR_INCOMPATIBLE_DISPLAY_KHR => error.IncompatibleDisplay,
        .VK_ERROR_VALIDATION_FAILED_EXT => error.ValidationFailed,
        .VK_ERROR_INVALID_SHADER_NV => error.InvalidShader,
        .VK_ERROR_INCOMPATIBLE_VERSION_KHR => error.IncompatibleVersion,
        .VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT => error.InvalidDRMFormatModifierPlaneLayout,
        .VK_ERROR_NOT_PERMITTED_EXT => error.NotPermitted,
        .VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => error.FullScreenExclusiveModeLost,
        else => {}, // XXX
    };
}

fn debugCallback(messageSeverity: c.VkDebugUtilsMessageSeverityFlagBitsEXT, messageTypes: c.VkDebugUtilsMessageTypeFlagsEXT, pCallbackData: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT, pUserData: ?*c_void) callconv(.C) c.VkBool32 {
    // TODO Show message severity and type
    std.debug.warn("{s}\n", .{pCallbackData.*.pMessage}); // XXX
    return c.VK_FALSE;
}

// fn isDeviceSuitable(device: VkPhysicalDevice) bool {
// QueueFamilyIndices indices = findQueueFamilies(device);

// bool extensionsSupported = checkDeviceExtensionSupport(device);

// bool swapChainAdequate = false;
// if (extensionsSupported) {
//     SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
//     swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
// }

// return indices.isComplete() && extensionsSupported && swapChainAdequate;
// }

// fn pickPhysicalDevice(allocator: *Allocator) !void {
//     var deviceCount: u32 = 0;
//     try checkResult(c.vkEnumeratePhysicalDevices(instance, &deviceCount, null));

//     if (deviceCount == 0)
//         return error.NoPhysicalDevicesFound; // XXX

//     var devices = try ArrayList([]const c.VkPhysicalDevice).initCapacity(allocator, deviceCount);
//     errdefer devices.deinit();

//     // try checkResult(c.vkEnumeratePhysicalDevices(instance, &deviceCount, devices.ptr));

//     // for (const auto& device : devices) {
//     //     if (isDeviceSuitable(device)) {
//     //         physicalDevice = device;
//     //         break;
//     //     }
//     // }

//     // if (physicalDevice == VK_NULL_HANDLE) {
//     //     throw std::runtime_error("failed to find a suitable GPU!");
//     // }
// }

pub fn drawFrame() !void {
    try checkResult(c.vkWaitForFences(device, 1, &inFlightFences[currentFrame], c.VK_TRUE, std.math.maxInt(u64)));
    try checkResult(c.vkResetFences(device, 1, &inFlightFences[currentFrame]));

    var imageIndex: u32 = undefined;
    try checkResult(c.vkAcquireNextImageKHR(device, swapChain, std.math.maxInt(u64), imageAvailableSemaphores[currentFrame], null, &imageIndex));

    var waitSemaphores = [_]c.VkSemaphore{imageAvailableSemaphores[currentFrame]};
    var waitStages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    const signalSemaphores = [_]c.VkSemaphore{renderFinishedSemaphores[currentFrame]};

    var submitInfoReal = c.VkSubmitInfo{
        .sType = .VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = null,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &waitSemaphores,
        .pWaitDstStageMask = &waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = commandBuffers.ptr + imageIndex,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signalSemaphores,
    };

    var submitInfo = [_]c.VkSubmitInfo{submitInfoReal};

    try checkResult(c.vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]));

    const swapChains = [_]c.VkSwapchainKHR{swapChain};
    var presentInfo = c.VkPresentInfoKHR{
        .sType = .VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = null,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signalSemaphores,
        .swapchainCount = 1,
        .pSwapchains = &swapChains,
        .pImageIndices = &imageIndex,
        .pResults = null,
    };

    try checkResult(c.vkQueuePresentKHR(presentQueue, &presentInfo));

    currentFrame = (currentFrame + 1) % max_frames_in_flight;
}

fn hash_cstr(a: [*:0]const u8) u32 {
    // FNV 32-bit hash
    var h: u32 = 2166136261;
    var i: usize = 0;
    while (a[i] != 0) : (i += 1) {
        h ^= a[i];
        h *%= 16777619;
    }
    return h;
}

fn eql_cstr(a: [*:0]const u8, b: [*:0]const u8) bool {
    return std.cstr.cmp(a, b) == 0;
}

fn checkDeviceExtensionSupport(allocator: *Allocator, phys_device: c.VkPhysicalDevice) !bool {
    var extensionCount: u32 = undefined;
    try checkResult(c.vkEnumerateDeviceExtensionProperties(phys_device, null, &extensionCount, null));

    const availableExtensions = try allocator.alloc(c.VkExtensionProperties, extensionCount);
    defer allocator.free(availableExtensions);
    try checkResult(c.vkEnumerateDeviceExtensionProperties(phys_device, null, &extensionCount, availableExtensions.ptr));

    var requiredExtensions = std.HashMap([*:0]const u8, void, hash_cstr, eql_cstr).init(allocator);
    defer requiredExtensions.deinit();
    for (device_extensions) |device_ext| {
        _ = try requiredExtensions.put(device_ext, {});
    }

    for (availableExtensions) |extension| {
        const name: [*:0]const u8 = @ptrCast([*:0]const u8, &extension.extensionName);
        _ = requiredExtensions.remove(name);
    }

    return requiredExtensions.count() == 0;
}

fn isDeviceSuitable(allocator: *Allocator, phys_device: c.VkPhysicalDevice) !bool {
    const indices = try findQueueFamilies(allocator, phys_device);

    const extensionsSupported = try checkDeviceExtensionSupport(allocator, phys_device);

    var swapChainAdequate = false;
    if (extensionsSupported) {
        var swapChainSupport = try querySwapChainSupport(allocator, phys_device);
        defer swapChainSupport.deinit();
        swapChainAdequate = swapChainSupport.formats.items.len != 0 and swapChainSupport.presentModes.items.len != 0;
    }

    return indices.isComplete() and extensionsSupported and swapChainAdequate;
}

fn pickPhysicalDevice(allocator: *Allocator) !void {
    var deviceCount: u32 = 0;
    try checkResult(c.vkEnumeratePhysicalDevices(instance, &deviceCount, null));

    if (deviceCount == 0) {
        return error.FailedToFindGPUsWithVulkanSupport; // TODO TODO
    }

    const devices = try allocator.alloc(c.VkPhysicalDevice, deviceCount);
    defer allocator.free(devices);
    try checkResult(c.vkEnumeratePhysicalDevices(instance, &deviceCount, devices.ptr));

    physicalDevice = for (devices) |dev| {
        if (try isDeviceSuitable(allocator, dev)) { // TODO TODO
            break dev;
        }
    } else return error.FailedToFindSuitableGPU; // TODO TODO
}

fn createWindowSurface(window: *c.GLFWwindow) !void { // XXX Bouger ailleurs?
    try checkResult(c.glfwCreateWindowSurface(instance, window, null, &surface));
}

fn getRequiredInstanceExtensions(allocator: *Allocator) ![][*c]const u8 {
    var glfwExtensionCount: u32 = 0;
    const glfwExtensions = c.glfwGetRequiredInstanceExtensions(&glfwExtensionCount) orelse
        return error.GlfwGetRequiredInstanceExtensionsFailed;

    var extensions = try ArrayList([*c]const u8).initCapacity(allocator, glfwExtensionCount);
    errdefer extensions.deinit();

    extensions.appendSliceAssumeCapacity(glfwExtensions[0..glfwExtensionCount]);
    if (builtin.mode == .Debug) {
        try extensions.append(c.VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions.toOwnedSlice();
}

fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layerCount: u32 = undefined;
    try checkResult(c.vkEnumerateInstanceLayerProperties(&layerCount, null));

    var availableLayers = try allocator.alloc(c.VkLayerProperties, layerCount);
    defer allocator.free(availableLayers);

    try checkResult(c.vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.ptr));

    for (validation_layers) |layerName| {
        var layerFound = false;
        for (availableLayers) |*layerProperties| {
            const name: [*:0]const u8 = @ptrCast([*:0]const u8, &layerProperties.layerName);
            if (std.cstr.cmp(layerName, name) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }

    return true;
}

fn setupDebugMessenger() !void {
    if (builtin.mode != .Debug) return;

    var createInfo = c.VkDebugUtilsMessengerCreateInfoEXT{
        .sType = .VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = null,
        .flags = 0,
        .messageSeverity = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = null, // XXX
    };

    const vkCreateDebugUtilsMessengerEXT = @ptrCast(c.PFN_vkCreateDebugUtilsMessengerEXT, c.vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")).?;
    try checkResult(vkCreateDebugUtilsMessengerEXT(instance, &createInfo, null, &debugMessenger));
}

fn CreateDebugReportCallbackEXT(
    pCreateInfo: *const c.VkDebugReportCallbackCreateInfoEXT,
    pAllocator: ?*const c.VkAllocationCallbacks,
    pCallback: *c.VkDebugReportCallbackEXT,
) c.VkResult {
    const func = @ptrCast(c.PFN_vkCreateDebugReportCallbackEXT, c.vkGetInstanceProcAddr(
        instance,
        "vkCreateDebugReportCallbackEXT",
    )) orelse return @intToEnum(c.VkResult, c.VK_ERROR_EXTENSION_NOT_PRESENT);
    return func(instance, pCreateInfo, pAllocator, pCallback);
}

fn createInstance(allocator: *Allocator) !void {
    const extensions = try getRequiredInstanceExtensions(allocator);
    defer allocator.free(extensions); // XXX

    var debugCreateInfo: c.VkDebugUtilsMessengerCreateInfoEXT = undefined;
    if (builtin.mode == .Debug) {
        // TODO checkValidationLayerSupport()
        debugCreateInfo = c.VkDebugUtilsMessengerCreateInfoEXT{
            .sType = .VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = null,
            .flags = 0,
            .messageSeverity = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            .pUserData = null, // XXX
        };
    }

    const appInfo = c.VkApplicationInfo{
        .sType = .VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "Triangle",
        .applicationVersion = 0, //c.VK_MAKE_VERSION(1, 0, 0), // XXX
        .pEngineName = null,
        .engineVersion = 0, //c.VK_MAKE_VERSION(1, 0, 0), // XXX
        .apiVersion = 0,
    };

    const createInfo = c.VkInstanceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = if (builtin.mode == .Debug) &debugCreateInfo else null,
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = if (builtin.mode == .Debug) validation_layers.len else 0, // XXX
        .ppEnabledLayerNames = if (builtin.mode == .Debug) &validation_layers else null,
        .enabledExtensionCount = @intCast(u32, extensions.len),
        .ppEnabledExtensionNames = extensions.ptr, // XXX
    };

    try checkResult(c.vkCreateInstance(&createInfo, null, &instance));
}

fn createShaderModule(code: []align(@alignOf(u32)) const u8) !c.VkShaderModule {
    var createInfo = c.VkShaderModuleCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .codeSize = code.len,
        .pCode = std.mem.bytesAsSlice(u32, code).ptr,
    };

    var shaderModule: c.VkShaderModule = undefined;
    try checkResult(c.vkCreateShaderModule(device, &createInfo, null, &shaderModule));

    return shaderModule;
}

fn createGraphicsPipeline(allocator: *Allocator) !void {
    const vertShaderCode = try std.fs.cwd().readFileAllocOptions(allocator, "shaders/triangle.vert.spv", 6666, @alignOf(u32), null);
    defer allocator.free(vertShaderCode);

    const fragShaderCode = try std.fs.cwd().readFileAllocOptions(allocator, "shaders/triangle.frag.spv", 6666, @alignOf(u32), null);
    defer allocator.free(fragShaderCode);

    const vertShaderModule = try createShaderModule(vertShaderCode);
    const fragShaderModule = try createShaderModule(fragShaderCode);

    var vertShaderStageInfo = c.VkPipelineShaderStageCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = .VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertShaderModule,
        .pName = "main",
        .pSpecializationInfo = null,
    };

    var fragShaderStageInfo = c.VkPipelineShaderStageCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = .VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragShaderModule,
        .pName = "main",
        .pSpecializationInfo = null,
    };

    const shaderStages = [_]c.VkPipelineShaderStageCreateInfo{ vertShaderStageInfo, fragShaderStageInfo };

    const vertexInputInfo = c.VkPipelineVertexInputStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = null,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = null,
    };

    var inputAssembly = c.VkPipelineInputAssemblyStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .topology = .VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = c.VK_FALSE,
    };

    const viewport = [_]c.VkViewport{c.VkViewport{
        .x = 0.0,
        .y = 0.0,
        .width = @intToFloat(f32, swapChainExtent.width),
        .height = @intToFloat(f32, swapChainExtent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    }};

    const scissor = [_]c.VkRect2D{c.VkRect2D{
        .offset = c.VkOffset2D{ .x = 0, .y = 0 },
        .extent = swapChainExtent,
    }};

    var viewportState = c.VkPipelineViewportStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    var rasterizer = c.VkPipelineRasterizationStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .depthClampEnable = c.VK_FALSE,
        .rasterizerDiscardEnable = c.VK_FALSE,
        .polygonMode = .VK_POLYGON_MODE_FILL,
        .cullMode = c.VK_CULL_MODE_BACK_BIT,
        .frontFace = .VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = c.VK_FALSE,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp = 0,
        .depthBiasSlopeFactor = 0,
        .lineWidth = 1.0,
    };

    var multisampling = c.VkPipelineMultisampleStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .rasterizationSamples = .VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = c.VK_FALSE,
        .minSampleShading = 0,
        .pSampleMask = null,
        .alphaToCoverageEnable = c.VK_FALSE,
        .alphaToOneEnable = c.VK_FALSE,
    };

    const colorBlendAttachment = c.VkPipelineColorBlendAttachmentState{
        .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
        .blendEnable = c.VK_FALSE,
        .srcColorBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = .VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = .VK_BLEND_OP_ADD,
    };

    var colorBlending = c.VkPipelineColorBlendStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .logicOpEnable = c.VK_FALSE,
        .logicOp = .VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
        .blendConstants = [_]f32{ 0, 0, 0, 0 },
    };

    const pipelineLayoutInfo = c.VkPipelineLayoutCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = null,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = null,
    };
    try checkResult(c.vkCreatePipelineLayout(device, &pipelineLayoutInfo, null, &pipelineLayout));

    var pipelineInfoReal = c.VkGraphicsPipelineCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stageCount = @intCast(u32, shaderStages.len),
        .pStages = &shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pTessellationState = null,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = null,
        .pColorBlendState = &colorBlending,
        .pDynamicState = null,
        .layout = pipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
        .basePipelineHandle = null,
        .basePipelineIndex = 0,
    };

    const pipelineInfo = [_]c.VkGraphicsPipelineCreateInfo{pipelineInfoReal};

    try checkResult(c.vkCreateGraphicsPipelines(
        device,
        null,
        @intCast(u32, pipelineInfo.len),
        &pipelineInfo,
        null,
        &pipeline,
    ));

    c.vkDestroyShaderModule(device, fragShaderModule, null);
    c.vkDestroyShaderModule(device, vertShaderModule, null);
}

fn createImageViews(allocator: *Allocator) !void {
    imageViews = try allocator.alloc(c.VkImageView, swapChainImages.len);
    errdefer allocator.free(imageViews);

    for (swapChainImages) |swap_chain_image, i| {
        var createInfo = c.VkImageViewCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .image = swap_chain_image,
            .viewType = .VK_IMAGE_VIEW_TYPE_2D,
            .format = swapChainImageFormat,
            .components = c.VkComponentMapping{
                .r = .VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = .VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = .VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = .VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = c.VkImageSubresourceRange{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        try checkResult(c.vkCreateImageView(device, &createInfo, null, &imageViews[i]));
    }
}

fn chooseSwapSurfaceFormat(availableFormats: []c.VkSurfaceFormatKHR) c.VkSurfaceFormatKHR {
    if (availableFormats.len == 1 and availableFormats[0].format == .VK_FORMAT_UNDEFINED) {
        return c.VkSurfaceFormatKHR{
            .format = .VK_FORMAT_B8G8R8A8_UNORM,
            .colorSpace = .VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        };
    }

    for (availableFormats) |availableFormat| {
        if (availableFormat.format == .VK_FORMAT_B8G8R8A8_UNORM and
            availableFormat.colorSpace == .VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

fn chooseSwapPresentMode(availablePresentModes: []c.VkPresentModeKHR) c.VkPresentModeKHR {
    var bestMode: c.VkPresentModeKHR = .VK_PRESENT_MODE_FIFO_KHR;

    for (availablePresentModes) |availablePresentMode| {
        if (availablePresentMode == .VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        } else if (availablePresentMode == .VK_PRESENT_MODE_IMMEDIATE_KHR) {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

fn chooseSwapExtent(capabilities: c.VkSurfaceCapabilitiesKHR) c.VkExtent2D {
    if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        var actualExtent = c.VkExtent2D{
            .width = WIDTH,
            .height = HEIGHT,
        };

        actualExtent.width = std.math.max(capabilities.minImageExtent.width, std.math.min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std.math.max(capabilities.minImageExtent.height, std.math.min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

fn createSwapChain(allocator: *Allocator) !void {
    var swapChainSupport = try querySwapChainSupport(allocator, physicalDevice);
    defer swapChainSupport.deinit();

    const surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats.items);
    const presentMode = chooseSwapPresentMode(swapChainSupport.presentModes.items);
    const extent = chooseSwapExtent(swapChainSupport.capabilities);

    var imageCount: u32 = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 and
        imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    const indices = try findQueueFamilies(allocator, physicalDevice);
    const queueFamilyIndices = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };

    const different_families = indices.graphicsFamily.? != indices.presentFamily.?;

    var createInfo = c.VkSwapchainCreateInfoKHR{
        .sType = .VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = if (different_families) .VK_SHARING_MODE_CONCURRENT else .VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = if (different_families) 2 else 0,
        .pQueueFamilyIndices = if (different_families) &queueFamilyIndices else &([_]u32{ 0, 0 }),
        .preTransform = swapChainSupport.capabilities.currentTransform,
        .compositeAlpha = .VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = c.VK_TRUE,
        .oldSwapchain = null,
    };

    try checkResult(c.vkCreateSwapchainKHR(device, &createInfo, null, &swapChain));

    try checkResult(c.vkGetSwapchainImagesKHR(device, swapChain, &imageCount, null));
    swapChainImages = try allocator.alloc(c.VkImage, imageCount);
    try checkResult(c.vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.ptr));

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

fn querySwapChainSupport(allocator: *Allocator, phys_device: c.VkPhysicalDevice) !SwapChainSupportDetails {
    var details = SwapChainSupportDetails.init(allocator);

    try checkResult(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_device, surface, &details.capabilities));

    var formatCount: u32 = undefined;
    try checkResult(c.vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &formatCount, null));

    if (formatCount != 0) {
        try details.formats.resize(formatCount);
        try checkResult(c.vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &formatCount, details.formats.items.ptr));
    }

    var presentModeCount: u32 = undefined;
    try checkResult(c.vkGetPhysicalDeviceSurfacePresentModesKHR(phys_device, surface, &presentModeCount, null));

    if (presentModeCount != 0) {
        try details.presentModes.resize(presentModeCount);
        try checkResult(c.vkGetPhysicalDeviceSurfacePresentModesKHR(phys_device, surface, &presentModeCount, details.presentModes.items.ptr));
    }

    return details;
}

fn findQueueFamilies(allocator: *Allocator, physDevice: c.VkPhysicalDevice) !QueueFamilyIndices {
    var indices = QueueFamilyIndices.init();

    var queueFamilyCount: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, null);

    const queueFamilies = try allocator.alloc(c.VkQueueFamilyProperties, queueFamilyCount);
    defer allocator.free(queueFamilies);
    c.vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, queueFamilies.ptr);

    var i: u32 = 0;
    for (queueFamilies) |queueFamily| {
        if (queueFamily.queueCount > 0 and
            queueFamily.queueFlags & @intCast(u32, c.VK_QUEUE_GRAPHICS_BIT) != 0)
        {
            indices.graphicsFamily = i;
        }

        var presentSupport: c.VkBool32 = 0;
        try checkResult(c.vkGetPhysicalDeviceSurfaceSupportKHR(physDevice, i, surface, &presentSupport));

        if (queueFamily.queueCount > 0 and presentSupport != 0) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i += 1;
    }

    return indices;
}

fn createLogicalDevice(allocator: *Allocator) !void {
    const indices = try findQueueFamilies(allocator, physicalDevice);

    var queueCreateInfos = ArrayList(c.VkDeviceQueueCreateInfo).init(allocator);
    defer queueCreateInfos.deinit();
    const all_queue_families = [_]u32{ indices.graphicsFamily.?, indices.presentFamily.? };
    const uniqueQueueFamilies = if (indices.graphicsFamily.? == indices.presentFamily.?)
        all_queue_families[0..1]
    else
        all_queue_families[0..2];

    var queuePriority: f32 = 1.0;
    for (uniqueQueueFamilies) |queueFamily| {
        var queueCreateInfo = c.VkDeviceQueueCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        };
        try queueCreateInfos.append(queueCreateInfo);
    }

    const deviceFeatures = c.VkPhysicalDeviceFeatures{
        .robustBufferAccess = 0,
        .fullDrawIndexUint32 = 0,
        .imageCubeArray = 0,
        .independentBlend = 0,
        .geometryShader = 0,
        .tessellationShader = 0,
        .sampleRateShading = 0,
        .dualSrcBlend = 0,
        .logicOp = 0,
        .multiDrawIndirect = 0,
        .drawIndirectFirstInstance = 0,
        .depthClamp = 0,
        .depthBiasClamp = 0,
        .fillModeNonSolid = 0,
        .depthBounds = 0,
        .wideLines = 0,
        .largePoints = 0,
        .alphaToOne = 0,
        .multiViewport = 0,
        .samplerAnisotropy = 0,
        .textureCompressionETC2 = 0,
        .textureCompressionASTC_LDR = 0,
        .textureCompressionBC = 0,
        .occlusionQueryPrecise = 0,
        .pipelineStatisticsQuery = 0,
        .vertexPipelineStoresAndAtomics = 0,
        .fragmentStoresAndAtomics = 0,
        .shaderTessellationAndGeometryPointSize = 0,
        .shaderImageGatherExtended = 0,
        .shaderStorageImageExtendedFormats = 0,
        .shaderStorageImageMultisample = 0,
        .shaderStorageImageReadWithoutFormat = 0,
        .shaderStorageImageWriteWithoutFormat = 0,
        .shaderUniformBufferArrayDynamicIndexing = 0,
        .shaderSampledImageArrayDynamicIndexing = 0,
        .shaderStorageBufferArrayDynamicIndexing = 0,
        .shaderStorageImageArrayDynamicIndexing = 0,
        .shaderClipDistance = 0,
        .shaderCullDistance = 0,
        .shaderFloat64 = 0,
        .shaderInt64 = 0,
        .shaderInt16 = 0,
        .shaderResourceResidency = 0,
        .shaderResourceMinLod = 0,
        .sparseBinding = 0,
        .sparseResidencyBuffer = 0,
        .sparseResidencyImage2D = 0,
        .sparseResidencyImage3D = 0,
        .sparseResidency2Samples = 0,
        .sparseResidency4Samples = 0,
        .sparseResidency8Samples = 0,
        .sparseResidency16Samples = 0,
        .sparseResidencyAliased = 0,
        .variableMultisampleRate = 0,
        .inheritedQueries = 0,
    };

    var createInfo = c.VkDeviceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueCreateInfoCount = @intCast(u32, queueCreateInfos.items.len),
        .pQueueCreateInfos = queueCreateInfos.items.ptr,
        .enabledLayerCount = if (builtin.mode == .Debug) @intCast(u32, validation_layers.len) else 0,
        .ppEnabledLayerNames = if (builtin.mode == .Debug) &validation_layers else null,
        .enabledExtensionCount = @intCast(u32, device_extensions.len),
        .ppEnabledExtensionNames = &device_extensions,
        .pEnabledFeatures = &deviceFeatures,
    };

    try checkResult(c.vkCreateDevice(physicalDevice, &createInfo, null, &device));

    c.vkGetDeviceQueue(device, indices.graphicsFamily.?, 0, &graphicsQueue);
    c.vkGetDeviceQueue(device, indices.presentFamily.?, 0, &presentQueue);
}

pub fn init(allocator: *Allocator, window: *c.GLFWwindow) !void {
    try createInstance(allocator);
    //try setupDebugMessenger();
    try createWindowSurface(window);
    try pickPhysicalDevice(allocator);
    try createLogicalDevice(allocator);
    try createSwapChain(allocator);
    try createImageViews(allocator);
    try createRenderPass();
    try createGraphicsPipeline(allocator);
    try createFramebuffers(allocator);
    try createCommandPool(allocator);
    try createCommandBuffers(allocator);
    try createSyncObjects();
}

pub fn cleanup() void { // XXX Naming
    //checkResult(c.vkDeviceWaitIdle(device)); // XXX Maybe back to main
    var i: usize = 0;
    while (i < max_frames_in_flight) : (i += 1) {
        c.vkDestroySemaphore(device, renderFinishedSemaphores[i], null);
        c.vkDestroySemaphore(device, imageAvailableSemaphores[i], null);
        c.vkDestroyFence(device, inFlightFences[i], null);
    }

    c.vkDestroyCommandPool(device, commandPool, null);

    for (framebuffers) |framebuffer| {
        c.vkDestroyFramebuffer(device, framebuffer, null);
    }

    c.vkDestroyPipeline(device, pipeline, null);
    c.vkDestroyPipelineLayout(device, pipelineLayout, null);
    c.vkDestroyRenderPass(device, renderPass, null);

    for (imageViews) |imageView| {
        c.vkDestroyImageView(device, imageView, null);
    }

    c.vkDestroySwapchainKHR(device, swapChain, null);
    c.vkDestroyDevice(device, null);

    if (builtin.mode == .Debug) {
        const vkDestroyDebugUtilsMessengerEXT = @ptrCast(c.PFN_vkDestroyDebugUtilsMessengerEXT, c.vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT")).?;
        vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, null);
    }

    c.vkDestroySurfaceKHR(instance, surface, null);
    c.vkDestroyInstance(instance, null);
}

fn createRenderPass() !void {
    const colorAttachment = c.VkAttachmentDescription{
        .flags = 0,
        .format = swapChainImageFormat,
        .samples = .VK_SAMPLE_COUNT_1_BIT,
        .loadOp = .VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = .VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = .VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = .VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = .VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = .VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    const colorAttachmentRef = c.VkAttachmentReference{
        .attachment = 0,
        .layout = .VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const subpass = c.VkSubpassDescription{
        .flags = 0,
        .pipelineBindPoint = .VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = null,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pResolveAttachments = null,
        .pDepthStencilAttachment = null,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = null,
    };

    const dependency = c.VkSubpassDependency{
        .srcSubpass = c.VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };

    var renderPassInfo = c.VkRenderPassCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    try checkResult(c.vkCreateRenderPass(device, &renderPassInfo, null, &renderPass));
}

fn createFramebuffers(allocator: *Allocator) !void {
    framebuffers = try allocator.alloc(c.VkFramebuffer, imageViews.len);

    for (imageViews) |image_view, i| {
        const attachments = [_]c.VkImageView{image_view};

        const framebufferInfo = c.VkFramebufferCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .renderPass = renderPass,
            .attachmentCount = 1,
            .pAttachments = &attachments,
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .layers = 1,
        };

        try checkResult(c.vkCreateFramebuffer(device, &framebufferInfo, null, &framebuffers[i]));
    }
}

fn createCommandPool(allocator: *Allocator) !void {
    const queueFamilyIndices = try findQueueFamilies(allocator, physicalDevice); // TODO

    const poolInfo = c.VkCommandPoolCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.?, // XXX
    };

    try checkResult(c.vkCreateCommandPool(device, &poolInfo, null, &commandPool));
}

fn createCommandBuffers(allocator: *Allocator) !void {
    commandBuffers = try allocator.alloc(c.VkCommandBuffer, framebuffers.len); // XXX free?
    // errdefer allocator.free(commandBuffer); // XXX Free

    const allocInfo = c.VkCommandBufferAllocateInfo{
        .sType = .VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .commandPool = commandPool,
        .level = .VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = @intCast(u32, commandBuffers.len),
    };

    try checkResult(c.vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.ptr));

    for (commandBuffers) |command_buffer, i| {
        const beginInfo = c.VkCommandBufferBeginInfo{
            .sType = .VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        };

        try checkResult(c.vkBeginCommandBuffer(command_buffer, &beginInfo));

        const clearColor = c.VkClearValue{ .color = c.VkClearColorValue{ .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 } } };
        const renderPassInfo = c.VkRenderPassBeginInfo{
            .sType = .VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = null,
            .renderPass = renderPass,
            .framebuffer = framebuffers[i],
            .renderArea = c.VkRect2D{
                .offset = c.VkOffset2D{ .x = 0, .y = 0 },
                .extent = swapChainExtent,
            },
            .clearValueCount = 1,
            .pClearValues = &clearColor,
        };

        c.vkCmdBeginRenderPass(command_buffer, &renderPassInfo, .VK_SUBPASS_CONTENTS_INLINE);
        {
            c.vkCmdBindPipeline(command_buffer, .VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            c.vkCmdDraw(command_buffer, 3, 1, 0, 0); // XXX
        }
        c.vkCmdEndRenderPass(command_buffer);

        try checkResult(c.vkEndCommandBuffer(command_buffer));
    }
}

fn createSyncObjects() !void {
    // TODO imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
    //
    const semaphoreInfo = c.VkSemaphoreCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
    };

    const fenceInfo = c.VkFenceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = null,
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };

    var i: usize = 0;
    while (i < max_frames_in_flight) : (i += 1) {
        try checkResult(c.vkCreateSemaphore(device, &semaphoreInfo, null, &imageAvailableSemaphores[i]));
        try checkResult(c.vkCreateSemaphore(device, &semaphoreInfo, null, &renderFinishedSemaphores[i]));
        try checkResult(c.vkCreateFence(device, &fenceInfo, null, &inFlightFences[i]));
    }
}

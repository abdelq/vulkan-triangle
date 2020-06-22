const std = @import("std");
const glfw = @import("glfw.zig");
const vk = @import("vulkan.zig");
const builtin = std.builtin;
const fs = std.fs;
const math = std.math;
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const BufSet = std.BufSet;

const max_frames_in_flight = 2;

const device_extensions = [_][]const u8{vk.swapchain_extension_name};
const validation_layers = [_][]const u8{vk.layer_khronos_validation};

const QueueFamilyIndices = struct {
    graphics_family: ?u32,
    present_family: ?u32,

    fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

const SwapChainSupportDetails = struct {
    capabilities: vk.SurfaceCapabilities,
    formats: ArrayList(vk.SurfaceFormat),
    present_modes: ArrayList(vk.PresentMode),

    fn init(allocator: *Allocator) SwapChainSupportDetails {
        return SwapChainSupportDetails{
            .capabilities = undefined,
            .formats = ArrayList(vk.SurfaceFormat).init(allocator),
            .present_modes = ArrayList(vk.PresentMode).init(allocator),
        };
    }

    fn deinit(self: *SwapChainSupportDetails) void {
        self.formats.deinit();
        self.present_modes.deinit();
    }
};

var window: *glfw.Window = undefined;

var instance: vk.Instance = undefined;
var debug_messenger: vk.DebugUtilsMessenger = undefined;
var surface: vk.Surface = undefined;

var physical_device: vk.PhysicalDevice = undefined;
var device: vk.Device = undefined;

var graphics_queue: vk.Queue = undefined;
var present_queue: vk.Queue = undefined;

var swap_chain: vk.Swapchain = undefined;
var swap_chain_images: []vk.Image = undefined;
var swap_chain_image_format: vk.Format = undefined;
var swap_chain_extent: vk.Extent2D = undefined;
var swap_chain_image_views: []vk.ImageView = undefined;
var swap_chain_framebuffers: []vk.Framebuffer = undefined;

var render_pass: vk.RenderPass = undefined;
var pipeline_layout: vk.PipelineLayout = undefined;
var graphics_pipeline: vk.Pipeline = undefined;

var command_pool: vk.CommandPool = undefined;
var command_buffers: []vk.CommandBuffer = undefined;

var image_available_semaphores: [max_frames_in_flight]vk.Semaphore = undefined;
var render_finished_semaphores: [max_frames_in_flight]vk.Semaphore = undefined;
var in_flight_fences: [max_frames_in_flight]vk.Fence = undefined;
var images_in_flight: []vk.Fence = undefined;
var current_frame: usize = 0;

var framebuffer_resized = false;

fn resizeCallback(win: ?*glfw.Window, width: c_int, height: c_int) callconv(.C) void {
    framebuffer_resized = true;
}

fn errorCallback(error_code: c_int, description: [*c]const u8) callconv(.C) void {
    std.debug.warn("GLFW: {s}\n", .{description});
}

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagBits,
    message_types: vk.DebugUtilsMessageTypeFlags,
    callback_data: [*c]const vk.DebugUtilsMessengerCallbackData,
    user_data: ?*c_void,
) callconv(.C) vk.Bool32 {
    std.debug.warn("Vulkan: {s}\n", .{callback_data.*.pMessage});
    return @boolToInt(false);
}

pub fn main() !void {
    if (builtin.mode == .Debug) {
        _ = glfw.setErrorCallback(errorCallback);
    }

    if (!glfw.init())
        return error.GlfwInitFailed;
    defer glfw.terminate();

    if (!glfw.vulkanSupported())
        return error.VulkanNotSupported;

    glfw.windowHint(glfw.client_api, glfw.no_api);

    window = glfw.createWindow(1920, 1080, "Triangle", null, null) orelse
        return error.GlfwCreateWindowFailed;
    defer glfw.destroyWindow(window);

    _ = glfw.setFramebufferSizeCallback(window, resizeCallback);

    try init();
    defer deinit();

    while (!glfw.windowShouldClose(window)) {
        glfw.pollEvents();
        try drawFrame();
    }

    try vk.deviceWaitIdle(device);
}

fn init() !void {
    const allocator = std.heap.c_allocator;
    try createInstance(allocator);
    try setupDebugMessenger();
    try createSurface();
    try pickPhysicalDevice(allocator);
    try createLogicalDevice(allocator);
    try createSwapChain(allocator);
    try createImageViews(allocator);
    try createRenderPass();
    try createGraphicsPipeline(allocator);
    try createFramebuffers(allocator);
    try createCommandPool(allocator);
    try createCommandBuffers(allocator);
    try createSyncObjects(allocator);
}

fn deinit() void {
    cleanupSwapChain();

    comptime var i = 0;
    inline while (i < max_frames_in_flight) : (i += 1) {
        vk.destroySemaphore(device, render_finished_semaphores[i], null);
        vk.destroySemaphore(device, image_available_semaphores[i], null);
        vk.destroyFence(device, in_flight_fences[i], null);
    }

    vk.destroyCommandPool(device, command_pool, null);

    vk.destroyDevice(device, null);

    vk.destroySurface(instance, surface, null);

    if (builtin.mode == .Debug) {
        vk.destroyDebugUtilsMessenger(instance, debug_messenger, null);
    }

    vk.destroyInstance(instance, null);
}

fn cleanupSwapChain() void {
    for (swap_chain_framebuffers) |framebuffer| {
        vk.destroyFramebuffer(device, framebuffer, null);
    }

    vk.freeCommandBuffers(device, command_pool, command_buffers);

    vk.destroyPipeline(device, graphics_pipeline, null);

    vk.destroyPipelineLayout(device, pipeline_layout, null);

    vk.destroyRenderPass(device, render_pass, null);

    for (swap_chain_image_views) |image_view| {
        vk.destroyImageView(device, image_view, null);
    }

    vk.destroySwapchain(device, swap_chain, null);
}

fn recreateSwapChain() !void {
    var width: i32 = 0;
    var height: i32 = 0;
    glfw.getFramebufferSize(window, &width, &height);
    while (width == 0 or height == 0) {
        glfw.getFramebufferSize(window, &width, &height);
        glfw.waitEvents();
    }

    try vk.deviceWaitIdle(device);

    cleanupSwapChain();

    const allocator = std.heap.c_allocator;
    try createSwapChain(allocator);
    try createImageViews(allocator);
    try createRenderPass();
    try createGraphicsPipeline(allocator);
    try createFramebuffers(allocator);
    try createCommandBuffers(allocator);
}

fn setupDebugMessenger() !void {
    if (builtin.mode != .Debug)
        return;

    const create_info = vk.DebugUtilsMessengerCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = null,
        .flags = 0,
        .messageSeverity = vk.debug_utils_message_severity_warning_bit |
            vk.debug_utils_message_severity_error_bit,
        .messageType = vk.debug_utils_message_type_general_bit |
            vk.debug_utils_message_type_validation_bit |
            vk.debug_utils_message_type_performance_bit,
        .pfnUserCallback = debugCallback,
        .pUserData = null,
    };

    try vk.createDebugUtilsMessenger(instance, &create_info, null, &debug_messenger);
}

fn createSurface() !void {
    try glfw.createWindowSurface(instance, window, null, &surface);
}

fn findQueueFamilies(allocator: *Allocator, phys_dev: vk.PhysicalDevice) !QueueFamilyIndices {
    var indices = QueueFamilyIndices{
        .graphics_family = null,
        .present_family = null,
    };

    var queue_family_count: u32 = 0;
    vk.getPhysicalDeviceQueueFamilyProperties(phys_dev, &queue_family_count, null);

    const queue_families = try allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    vk.getPhysicalDeviceQueueFamilyProperties(phys_dev, &queue_family_count, queue_families);

    for (queue_families) |queue_family, i| {
        if (queue_family.queueFlags & @intCast(u32, vk.queue_graphics_bit) != 0) {
            indices.graphics_family = @intCast(u32, i);
        }

        var present_support align(@alignOf(vk.Bool32)) = false;
        try vk.getPhysicalDeviceSurfaceSupport(phys_dev, i, surface, &present_support);
        if (present_support) {
            indices.present_family = @intCast(u32, i);
        }

        if (indices.isComplete()) {
            break;
        }
    }

    return indices;
}

fn checkDeviceExtensionSupport(allocator: *Allocator, phys_dev: vk.PhysicalDevice) !bool {
    var extension_count: u32 = 0;
    _ = try vk.enumerateDeviceExtensionProperties(phys_dev, null, &extension_count, null);

    const available_extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
    defer allocator.free(available_extensions);
    _ = try vk.enumerateDeviceExtensionProperties(phys_dev, null, &extension_count, available_extensions);

    var required_extensions = BufSet.init(allocator);
    defer required_extensions.deinit();

    for (device_extensions) |extension| {
        try required_extensions.put(extension);
    }

    for (available_extensions) |extension| {
        required_extensions.delete(mem.spanZ(@ptrCast([*:0]const u8, &extension.extensionName)));
    }

    return required_extensions.count() == 0;
}

fn querySwapChainSupport(allocator: *Allocator, phys_dev: vk.PhysicalDevice) !SwapChainSupportDetails {
    var details = SwapChainSupportDetails.init(allocator);
    errdefer details.deinit();

    try vk.getPhysicalDeviceSurfaceCapabilities(phys_dev, surface, &details.capabilities);

    var format_count: u32 = 0;
    _ = try vk.getPhysicalDeviceSurfaceFormats(phys_dev, surface, &format_count, null);
    if (format_count > 0) {
        try details.formats.resize(format_count);
        _ = try vk.getPhysicalDeviceSurfaceFormats(phys_dev, surface, &format_count, details.formats.items);
    }

    var present_mode_count: u32 = 0;
    _ = try vk.getPhysicalDeviceSurfacePresentModes(phys_dev, surface, &present_mode_count, null);
    if (present_mode_count > 0) {
        try details.present_modes.resize(present_mode_count);
        _ = try vk.getPhysicalDeviceSurfacePresentModes(phys_dev, surface, &present_mode_count, details.present_modes.items);
    }

    return details;
}

fn isDeviceSuitable(allocator: *Allocator, phys_dev: vk.PhysicalDevice) !bool {
    const indices = try findQueueFamilies(allocator, phys_dev);
    if (!indices.isComplete())
        return false;

    var swap_chain_adequate = false;
    if (try checkDeviceExtensionSupport(allocator, phys_dev)) {
        var swap_chain_support = try querySwapChainSupport(allocator, phys_dev);
        defer swap_chain_support.deinit();

        swap_chain_adequate =
            swap_chain_support.formats.items.len > 0 and
            swap_chain_support.present_modes.items.len > 0;
    }
    return swap_chain_adequate;
}

fn pickPhysicalDevice(allocator: *Allocator) !void {
    var device_count: u32 = 0;
    _ = try vk.enumeratePhysicalDevices(instance, &device_count, null);

    if (device_count == 0)
        return error.FailedToFindGPUsWithVulkanSupport;

    const devices = try allocator.alloc(vk.PhysicalDevice, device_count);
    defer allocator.free(devices);
    _ = try vk.enumeratePhysicalDevices(instance, &device_count, devices);

    physical_device = for (devices) |dev| {
        if (try isDeviceSuitable(allocator, dev)) {
            break dev;
        }
    } else return error.FailedToFindSuitableGPU;
}

fn createLogicalDevice(allocator: *Allocator) !void {
    const indices = try findQueueFamilies(allocator, physical_device);

    const all_queue_families = [_]u32{ indices.graphics_family.?, indices.present_family.? };
    const unique_queue_families = if (indices.graphics_family.? == indices.present_family.?)
        all_queue_families[0..1]
    else
        all_queue_families[0..2];

    const queue_create_infos = try allocator.alloc(vk.DeviceQueueCreateInfo, unique_queue_families.len);
    defer allocator.free(queue_create_infos);

    const queue_priority: f32 = 1.0;
    for (unique_queue_families) |queue_family, i| {
        queue_create_infos[i] = vk.DeviceQueueCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
    }

    const device_features = mem.zeroes(vk.PhysicalDeviceFeatures);
    var create_info = vk.DeviceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueCreateInfoCount = @intCast(u32, queue_create_infos.len),
        .pQueueCreateInfos = queue_create_infos.ptr,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = @intCast(u32, device_extensions.len),
        .ppEnabledExtensionNames = @ptrCast([*]const [*]const u8, &device_extensions),
        .pEnabledFeatures = &device_features,
    };

    if (builtin.mode == .Debug) {
        create_info.enabledLayerCount = @intCast(u32, validation_layers.len);
        create_info.ppEnabledLayerNames = @ptrCast([*]const [*]const u8, &validation_layers);
    }

    try vk.createDevice(physical_device, &create_info, null, &device);

    vk.getDeviceQueue(device, indices.graphics_family.?, 0, &graphics_queue);
    vk.getDeviceQueue(device, indices.present_family.?, 0, &present_queue);
}

fn chooseSwapSurfaceFormat(formats: []vk.SurfaceFormat) vk.SurfaceFormat {
    return for (formats) |format| {
        if (format.format == .VK_FORMAT_B8G8R8A8_SRGB and
            format.colorSpace == .VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            break format;
        }
    } else formats[0];
}

fn chooseSwapPresentMode(present_modes: []vk.PresentMode) vk.PresentMode {
    return for (present_modes) |mode| {
        if (mode == .VK_PRESENT_MODE_MAILBOX_KHR) {
            break mode;
        }
    } else .VK_PRESENT_MODE_FIFO_KHR;
}

fn chooseSwapExtent(capabilities: *vk.SurfaceCapabilities) vk.Extent2D {
    if (capabilities.currentExtent.width < math.maxInt(u32))
        return capabilities.currentExtent;

    var width: i32 = 0;
    var height: i32 = 0;
    glfw.getFramebufferSize(window, &width, &height);

    return vk.Extent2D{
        .width = math.max(
            capabilities.minImageExtent.width,
            math.min(capabilities.maxImageExtent.width, @intCast(u32, width)),
        ),
        .height = math.max(
            capabilities.minImageExtent.height,
            math.min(capabilities.maxImageExtent.height, @intCast(u32, height)),
        ),
    };
}

fn createSwapChain(allocator: *Allocator) !void {
    var swap_chain_support = try querySwapChainSupport(allocator, physical_device);
    defer swap_chain_support.deinit();

    const surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats.items);
    const present_mode = chooseSwapPresentMode(swap_chain_support.present_modes.items);
    const extent = chooseSwapExtent(&swap_chain_support.capabilities);

    var image_count = swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 and
        swap_chain_support.capabilities.maxImageCount < image_count)
    {
        image_count = swap_chain_support.capabilities.maxImageCount;
    }

    var create_info = vk.SwapchainCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = null,
        .flags = 0,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk.image_usage_color_attachment_bit,
        .imageSharingMode = .VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = null,
        .preTransform = swap_chain_support.capabilities.currentTransform,
        .compositeAlpha = .VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = @boolToInt(true),
        .oldSwapchain = null,
    };

    const indices = try findQueueFamilies(allocator, physical_device);
    if (indices.graphics_family.? != indices.present_family.?) {
        const queue_families = [_]u32{ indices.graphics_family.?, indices.present_family.? };

        create_info.imageSharingMode = .VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = queue_families.len;
        create_info.pQueueFamilyIndices = &queue_families;
    }

    try vk.createSwapchain(device, &create_info, null, &swap_chain);

    _ = try vk.getSwapchainImages(device, swap_chain, &image_count, null);
    swap_chain_images = try allocator.alloc(vk.Image, image_count);
    errdefer allocator.free(swap_chain_images);
    _ = try vk.getSwapchainImages(device, swap_chain, &image_count, swap_chain_images);

    swap_chain_image_format = surface_format.format;
    swap_chain_extent = extent;
}

fn createImageViews(allocator: *Allocator) !void {
    swap_chain_image_views = try allocator.alloc(vk.ImageView, swap_chain_images.len);
    errdefer allocator.free(swap_chain_image_views);

    for (swap_chain_images) |image, i| {
        const create_info = vk.ImageViewCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .image = image,
            .viewType = .VK_IMAGE_VIEW_TYPE_2D,
            .format = swap_chain_image_format,
            .components = vk.ComponentMapping{
                .r = .VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = .VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = .VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = .VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = vk.ImageSubresourceRange{
                .aspectMask = vk.image_aspect_color_bit,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        try vk.createImageView(device, &create_info, null, &swap_chain_image_views[i]);
    }
}

fn createRenderPass() !void {
    const color_attachment = vk.AttachmentDescription{
        .flags = 0,
        .format = swap_chain_image_format,
        .samples = .VK_SAMPLE_COUNT_1_BIT,
        .loadOp = .VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = .VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = .VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = .VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = .VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = .VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    const subpass = vk.SubpassDescription{
        .flags = 0,
        .pipelineBindPoint = .VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = null,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = null,
        .pDepthStencilAttachment = null,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = null,
    };

    const dependency = vk.SubpassDependency{
        .srcSubpass = vk.subpass_external,
        .dstSubpass = 0,
        .srcStageMask = vk.pipeline_stage_color_attachment_output_bit,
        .dstStageMask = vk.pipeline_stage_color_attachment_output_bit,
        .srcAccessMask = 0,
        .dstAccessMask = vk.access_color_attachment_write_bit,
        .dependencyFlags = 0,
    };

    const render_pass_info = vk.RenderPassCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    try vk.createRenderPass(device, &render_pass_info, null, &render_pass);
}

fn createFramebuffers(allocator: *Allocator) !void {
    swap_chain_framebuffers = try allocator.alloc(vk.Framebuffer, swap_chain_image_views.len);
    errdefer allocator.free(swap_chain_framebuffers);

    for (swap_chain_image_views) |*image_view, i| {
        const framebuffer_info = vk.FramebufferCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = @ptrCast([*]const vk.ImageView, image_view),
            .width = swap_chain_extent.width,
            .height = swap_chain_extent.height,
            .layers = 1,
        };

        try vk.createFramebuffer(device, &framebuffer_info, null, &swap_chain_framebuffers[i]);
    }
}

fn createCommandPool(allocator: *Allocator) !void {
    const indices = try findQueueFamilies(allocator, physical_device);

    const pool_info = vk.CommandPoolCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = indices.graphics_family.?,
    };

    try vk.createCommandPool(device, &pool_info, null, &command_pool);
}

fn createSyncObjects(allocator: *Allocator) !void {
    images_in_flight = try allocator.alloc(vk.Fence, swap_chain_images.len);
    errdefer allocator.free(images_in_flight);
    mem.set(vk.Fence, images_in_flight, null);

    const semaphore_info = vk.SemaphoreCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
    };

    const fence_info = vk.FenceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = null,
        .flags = vk.fence_create_signaled_bit,
    };

    comptime var i = 0;
    inline while (i < max_frames_in_flight) : (i += 1) {
        try vk.createSemaphore(device, &semaphore_info, null, &image_available_semaphores[i]);
        try vk.createSemaphore(device, &semaphore_info, null, &render_finished_semaphores[i]);
        try vk.createFence(device, &fence_info, null, &in_flight_fences[i]);
    }
}

fn createShaderModule(code: []const u8) !vk.ShaderModule {
    const create_info = vk.ShaderModuleCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .codeSize = code.len,
        .pCode = mem.bytesAsSlice(u32, @alignCast(@alignOf(u32), code)).ptr, // XXX
    };

    var shader_module: vk.ShaderModule = undefined;
    try vk.createShaderModule(device, &create_info, null, &shader_module);
    return shader_module;
}

fn createGraphicsPipeline(allocator: *Allocator) !void {
    const vert_shader_code = try fs.cwd().readFileAlloc(allocator, "shaders/triangle.vert.spv", 2048);
    defer allocator.free(vert_shader_code);

    const frag_shader_code = try fs.cwd().readFileAlloc(allocator, "shaders/triangle.frag.spv", 2048);
    defer allocator.free(frag_shader_code);

    const vert_shader_module = try createShaderModule(vert_shader_code);
    defer vk.destroyShaderModule(device, vert_shader_module, null);

    const frag_shader_module = try createShaderModule(frag_shader_code);
    defer vk.destroyShaderModule(device, frag_shader_module, null);

    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
        vk.PipelineShaderStageCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = .VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_shader_module,
            .pName = "main",
            .pSpecializationInfo = null,
        },
        vk.PipelineShaderStageCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = .VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_shader_module,
            .pName = "main",
            .pSpecializationInfo = null,
        },
    };

    const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = null,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = null,
    };

    const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .topology = .VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = @boolToInt(false),
    };

    const viewport = vk.Viewport{
        .x = 0.0,
        .y = 0.0,
        .width = @intToFloat(f32, swap_chain_extent.width),
        .height = @intToFloat(f32, swap_chain_extent.height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };

    const scissor = vk.Rect2D{
        .offset = vk.Offset2D{ .x = 0, .y = 0 },
        .extent = swap_chain_extent,
    };

    const viewport_state = vk.PipelineViewportStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const rasterizer = vk.PipelineRasterizationStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .depthClampEnable = @boolToInt(false),
        .rasterizerDiscardEnable = @boolToInt(false),
        .polygonMode = .VK_POLYGON_MODE_FILL,
        .cullMode = vk.cull_mode_back_bit,
        .frontFace = .VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = @boolToInt(false),
        .depthBiasConstantFactor = 0,
        .depthBiasClamp = 0,
        .depthBiasSlopeFactor = 0,
        .lineWidth = 1.0,
    };

    const multisampling = vk.PipelineMultisampleStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .rasterizationSamples = .VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = @boolToInt(false),
        .minSampleShading = 0,
        .pSampleMask = null,
        .alphaToCoverageEnable = @boolToInt(false),
        .alphaToOneEnable = @boolToInt(false),
    };

    const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
        .colorWriteMask = vk.color_component_r_bit |
            vk.color_component_g_bit |
            vk.color_component_b_bit |
            vk.color_component_a_bit,
        .blendEnable = @boolToInt(false),
        .srcColorBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = .VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = .VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = .VK_BLEND_OP_ADD,
    };

    const color_blending = vk.PipelineColorBlendStateCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .logicOpEnable = @boolToInt(false),
        .logicOp = .VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants = [_]f32{ 0, 0, 0, 0 },
    };

    const pipeline_layout_info = vk.PipelineLayoutCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = null,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = null,
    };

    try vk.createPipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout);

    const pipeline_info = [_]vk.GraphicsPipelineCreateInfo{vk.GraphicsPipelineCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stageCount = shader_stages.len,
        .pStages = &shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = null,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = null,
        .pColorBlendState = &color_blending,
        .pDynamicState = null,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = null,
        .basePipelineIndex = 0,
    }};

    _ = try vk.createGraphicsPipelines(device, null, &pipeline_info, null, @as(*[1]vk.Pipeline, &graphics_pipeline));
}

fn drawFrame() !void {
    _ = try vk.waitForFences(device, @as(*[1]vk.Fence, &in_flight_fences[current_frame]), true, math.maxInt(u64));

    var image_index: u32 = 0;
    if (vk.acquireNextImage(device, swap_chain, math.maxInt(u64), image_available_semaphores[current_frame], null, &image_index)) |result| {
        if (result != .Success and result != .Suboptimal)
            return error.FailedToAcquireImage;
    } else |err| switch (err) {
        error.OutOfDate => {
            try recreateSwapChain();
            return;
        },
        else => return err,
    }

    if (images_in_flight[image_index] != null) {
        _ = try vk.waitForFences(device, @as(*[1]vk.Fence, &images_in_flight[image_index]), true, math.maxInt(u64));
    }
    images_in_flight[image_index] = in_flight_fences[current_frame];

    const signal_semaphores = [_]vk.Semaphore{render_finished_semaphores[current_frame]}; // XXX

    const submit_info = [_]vk.SubmitInfo{vk.SubmitInfo{ // XXX
        .sType = .VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = null,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &[_]vk.Semaphore{image_available_semaphores[current_frame]},
        .pWaitDstStageMask = &[_]vk.PipelineStageFlags{vk.pipeline_stage_color_attachment_output_bit},
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffers[image_index],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphores,
    }};

    try vk.resetFences(device, @as(*[1]vk.Fence, &in_flight_fences[current_frame]));

    try vk.queueSubmit(graphics_queue, &submit_info, in_flight_fences[current_frame]);

    const present_info = vk.PresentInfo{ // XXX
        .sType = .VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = null,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &signal_semaphores,
        .swapchainCount = 1,
        .pSwapchains = &[_]vk.Swapchain{swap_chain},
        .pImageIndices = &image_index,
        .pResults = null,
    };

    if (vk.queuePresent(present_queue, &present_info)) |result| {
        if (result == .Suboptimal or framebuffer_resized) {
            framebuffer_resized = false;
            try recreateSwapChain();
        }
    } else |err| switch (err) {
        error.OutOfDate => {
            framebuffer_resized = false;
            try recreateSwapChain();
        },
        else => return err,
    }

    current_frame = (current_frame + 1) % max_frames_in_flight;
}

fn getRequiredExtensions(allocator: *Allocator) ![]const [*c]const u8 {
    var glfw_extension_count: u32 = 0;
    const glfw_extensions = glfw.getRequiredInstanceExtensions(&glfw_extension_count) orelse
        return error.GlfwGetRequiredInstanceExtensionsFailed;

    // TODO Use unknown length pointer over C pointer
    var extensions = try ArrayList([*c]const u8).initCapacity(allocator, glfw_extension_count + 1);
    errdefer extensions.deinit();

    extensions.appendSliceAssumeCapacity(glfw_extensions[0..glfw_extension_count]);
    if (builtin.mode == .Debug) {
        extensions.appendAssumeCapacity(vk.debug_utils_extension_name);
    }

    return extensions.toOwnedSlice();
}

fn checkValidationLayerSupport(allocator: *Allocator) !bool {
    var layer_count: u32 = 0;
    _ = try vk.enumerateInstanceLayerProperties(&layer_count, null);

    const available_layers = try allocator.alloc(vk.LayerProperties, layer_count);
    defer allocator.free(available_layers);
    _ = try vk.enumerateInstanceLayerProperties(&layer_count, available_layers);

    for (validation_layers) |validation_layer| {
        var layer_found = false;

        for (available_layers) |available_layer| {
            const layer = mem.spanZ(@ptrCast([*:0]const u8, &available_layer.layerName));
            if (mem.eql(u8, layer, validation_layer)) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found)
            return false;
    }

    return true;
}

fn createInstance(allocator: *Allocator) !void {
    const app_info = vk.ApplicationInfo{
        .sType = .VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "Triangle",
        .applicationVersion = vk.makeVersion(1, 0, 0),
        .pEngineName = null,
        .engineVersion = 0,
        .apiVersion = 0,
    };

    const extensions = try getRequiredExtensions(allocator);
    defer allocator.free(extensions);

    var create_info = vk.InstanceCreateInfo{
        .sType = .VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = @intCast(u32, extensions.len),
        .ppEnabledExtensionNames = extensions.ptr,
    };

    if (builtin.mode == .Debug) {
        if (!try checkValidationLayerSupport(allocator))
            return error.ValidationLayersNotAvailable;

        const debug_create_info = vk.DebugUtilsMessengerCreateInfo{
            .sType = .VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = null,
            .flags = 0,
            .messageSeverity = vk.debug_utils_message_severity_warning_bit |
                vk.debug_utils_message_severity_error_bit,
            .messageType = vk.debug_utils_message_type_general_bit |
                vk.debug_utils_message_type_validation_bit |
                vk.debug_utils_message_type_performance_bit,
            .pfnUserCallback = debugCallback,
            .pUserData = null,
        };

        create_info.pNext = &debug_create_info;
        create_info.enabledLayerCount = validation_layers.len;
        create_info.ppEnabledLayerNames = @ptrCast([*]const [*]const u8, &validation_layers);
    }

    try vk.createInstance(&create_info, null, &instance);
}

fn createCommandBuffers(allocator: *Allocator) !void {
    command_buffers = try allocator.alloc(vk.CommandBuffer, swap_chain_framebuffers.len);
    errdefer allocator.free(command_buffers);

    const alloc_info = vk.CommandBufferAllocateInfo{
        .sType = .VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .commandPool = command_pool,
        .level = .VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = @intCast(u32, command_buffers.len),
    };

    try vk.allocateCommandBuffers(device, &alloc_info, command_buffers);

    for (command_buffers) |command_buffer, i| {
        const begin_info = vk.CommandBufferBeginInfo{
            .sType = .VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        };

        const clear_color = vk.ClearValue{
            .color = vk.ClearColorValue{
                .float32 = [_]f32{ 0.0, 0.0, 0.0, 1.0 },
            },
        };

        const render_pass_info = vk.RenderPassBeginInfo{
            .sType = .VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = null,
            .renderPass = render_pass,
            .framebuffer = swap_chain_framebuffers[i],
            .renderArea = vk.Rect2D{
                .offset = vk.Offset2D{ .x = 0, .y = 0 },
                .extent = swap_chain_extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };

        try vk.beginCommandBuffer(command_buffer, &begin_info);

        vk.cmdBeginRenderPass(command_buffer, &render_pass_info, .VK_SUBPASS_CONTENTS_INLINE);
        {
            vk.cmdBindPipeline(command_buffer, .VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
            vk.cmdDraw(command_buffer, 3, 1, 0, 0);
        }
        vk.cmdEndRenderPass(command_buffer);

        try vk.endCommandBuffer(command_buffer);
    }
}

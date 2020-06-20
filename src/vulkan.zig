usingnamespace @import("c.zig");

// Types
pub const Bool32 = VkBool32;
pub const DebugUtilsMessageTypeFlags = VkDebugUtilsMessageTypeFlagsEXT;
pub const PipelineStageFlags = VkPipelineStageFlags;

// Constants
pub const debug_utils_extension_name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
pub const layer_khronos_validation = "VK_LAYER_KHRONOS_validation"; // XXX
pub const subpass_external = VK_SUBPASS_EXTERNAL;
pub const swapchain_extension_name = VK_KHR_SWAPCHAIN_EXTENSION_NAME;

// Flags
pub const access_color_attachment_write_bit = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
pub const color_component_a_bit = VK_COLOR_COMPONENT_A_BIT;
pub const color_component_b_bit = VK_COLOR_COMPONENT_B_BIT;
pub const color_component_g_bit = VK_COLOR_COMPONENT_G_BIT;
pub const color_component_r_bit = VK_COLOR_COMPONENT_R_BIT;
pub const cull_mode_back_bit = VK_CULL_MODE_BACK_BIT;
pub const debug_utils_message_severity_error_bit = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
pub const debug_utils_message_severity_verbose_bit = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
pub const debug_utils_message_severity_warning_bit = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
pub const debug_utils_message_type_general_bit = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
pub const debug_utils_message_type_performance_bit = VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
pub const debug_utils_message_type_validation_bit = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
pub const fence_create_signaled_bit = VK_FENCE_CREATE_SIGNALED_BIT;
pub const image_aspect_color_bit = VK_IMAGE_ASPECT_COLOR_BIT;
pub const image_usage_color_attachment_bit = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
pub const pipeline_stage_color_attachment_output_bit = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
pub const queue_graphics_bit = VK_QUEUE_GRAPHICS_BIT;

// Enumerations
pub const DebugUtilsMessageSeverityFlagBits = VkDebugUtilsMessageSeverityFlagBitsEXT;
pub const Format = VkFormat;
pub const PresentMode = VkPresentModeKHR;

// Unions
pub const ClearColorValue = VkClearColorValue;
pub const ClearValue = VkClearValue;

// Structures
const AllocationCallbacks = VkAllocationCallbacks;

pub const ApplicationInfo = VkApplicationInfo;
pub const AttachmentDescription = VkAttachmentDescription;
pub const AttachmentReference = VkAttachmentReference;
pub const CommandBufferAllocateInfo = VkCommandBufferAllocateInfo;
pub const CommandBufferBeginInfo = VkCommandBufferBeginInfo;
pub const CommandPoolCreateInfo = VkCommandPoolCreateInfo;
pub const ComponentMapping = VkComponentMapping;
pub const DebugUtilsMessengerCallbackData = VkDebugUtilsMessengerCallbackDataEXT;
pub const DebugUtilsMessengerCreateInfo = VkDebugUtilsMessengerCreateInfoEXT;
pub const DeviceCreateInfo = VkDeviceCreateInfo;
pub const DeviceQueueCreateInfo = VkDeviceQueueCreateInfo;
pub const ExtensionProperties = VkExtensionProperties;
pub const Extent2D = VkExtent2D;
pub const FenceCreateInfo = VkFenceCreateInfo;
pub const FramebufferCreateInfo = VkFramebufferCreateInfo;
pub const GraphicsPipelineCreateInfo = VkGraphicsPipelineCreateInfo;
pub const ImageSubresourceRange = VkImageSubresourceRange;
pub const ImageViewCreateInfo = VkImageViewCreateInfo;
pub const InstanceCreateInfo = VkInstanceCreateInfo;
pub const LayerProperties = VkLayerProperties;
pub const Offset2D = VkOffset2D;
pub const PhysicalDeviceFeatures = VkPhysicalDeviceFeatures;
pub const PipelineColorBlendAttachmentState = VkPipelineColorBlendAttachmentState;
pub const PipelineColorBlendStateCreateInfo = VkPipelineColorBlendStateCreateInfo;
pub const PipelineInputAssemblyStateCreateInfo = VkPipelineInputAssemblyStateCreateInfo;
pub const PipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo;
pub const PipelineMultisampleStateCreateInfo = VkPipelineMultisampleStateCreateInfo;
pub const PipelineRasterizationStateCreateInfo = VkPipelineRasterizationStateCreateInfo;
pub const PipelineShaderStageCreateInfo = VkPipelineShaderStageCreateInfo;
pub const PipelineVertexInputStateCreateInfo = VkPipelineVertexInputStateCreateInfo;
pub const PipelineViewportStateCreateInfo = VkPipelineViewportStateCreateInfo;
pub const PresentInfo = VkPresentInfoKHR;
pub const QueueFamilyProperties = VkQueueFamilyProperties;
pub const Rect2D = VkRect2D;
pub const RenderPassBeginInfo = VkRenderPassBeginInfo;
pub const RenderPassCreateInfo = VkRenderPassCreateInfo;
pub const SemaphoreCreateInfo = VkSemaphoreCreateInfo;
pub const ShaderModuleCreateInfo = VkShaderModuleCreateInfo;
pub const SubmitInfo = VkSubmitInfo;
pub const SubpassDependency = VkSubpassDependency;
pub const SubpassDescription = VkSubpassDescription;
pub const SurfaceCapabilities = VkSurfaceCapabilitiesKHR;
pub const SurfaceFormat = VkSurfaceFormatKHR;
pub const SwapchainCreateInfo = VkSwapchainCreateInfoKHR;
pub const Viewport = VkViewport;

// Opaque Handles
const PipelineCache = VkPipelineCache;

pub const CommandBuffer = VkCommandBuffer;
pub const CommandPool = VkCommandPool;
pub const DebugUtilsMessenger = VkDebugUtilsMessengerEXT;
pub const Device = VkDevice;
pub const Fence = VkFence;
pub const Framebuffer = VkFramebuffer;
pub const Image = VkImage;
pub const ImageView = VkImageView;
pub const Instance = VkInstance;
pub const PhysicalDevice = VkPhysicalDevice;
pub const Pipeline = VkPipeline;
pub const PipelineLayout = VkPipelineLayout;
pub const Queue = VkQueue;
pub const RenderPass = VkRenderPass;
pub const Semaphore = VkSemaphore;
pub const ShaderModule = VkShaderModule;
pub const Surface = VkSurfaceKHR;
pub const Swapchain = VkSwapchainKHR;

// Functions
pub const cmdBeginRenderPass = vkCmdBeginRenderPass;
pub const cmdBindPipeline = vkCmdBindPipeline;
pub const cmdDraw = vkCmdDraw;
pub const cmdEndRenderPass = vkCmdEndRenderPass;
pub const destroyCommandPool = vkDestroyCommandPool;
pub const destroyDevice = vkDestroyDevice;
pub const destroyFence = vkDestroyFence;
pub const destroyFramebuffer = vkDestroyFramebuffer;
pub const destroyImageView = vkDestroyImageView;
pub const destroyInstance = vkDestroyInstance;
pub const destroyPipeline = vkDestroyPipeline;
pub const destroyPipelineLayout = vkDestroyPipelineLayout;
pub const destroyRenderPass = vkDestroyRenderPass;
pub const destroySemaphore = vkDestroySemaphore;
pub const destroyShaderModule = vkDestroyShaderModule;
pub const destroySurface = vkDestroySurfaceKHR;
pub const destroySwapchain = vkDestroySwapchainKHR;
pub const freeCommandBuffers = vkFreeCommandBuffers;
pub const getDeviceQueue = vkGetDeviceQueue;
pub const getPhysicalDeviceQueueFamilyProperties = vkGetPhysicalDeviceQueueFamilyProperties;

pub inline fn acquireNextImage(
    device: Device,
    swapchain: Swapchain,
    timeout: u64,
    semaphore: Semaphore,
    fence: Fence,
    image_index: *u32,
) !Success {
    return switch (vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, image_index)) {
        .VK_SUCCESS => .Success,
        .VK_TIMEOUT => .Timeout,
        .VK_NOT_READY => .NotReady,
        .VK_SUBOPTIMAL_KHR => .Suboptimal,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        .VK_ERROR_OUT_OF_DATE_KHR => error.OutOfDate,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        .VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => error.FullScreenExclusiveModeLost,
        else => unreachable,
    };
}

pub inline fn allocateCommandBuffers(
    device: Device,
    allocate_info: *const CommandBufferAllocateInfo,
    command_buffers: []CommandBuffer,
) !void {
    return switch (vkAllocateCommandBuffers(device, allocate_info, command_buffers.ptr)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn beginCommandBuffer(
    command_buffer: CommandBuffer,
    begin_info: *const CommandBufferBeginInfo,
) !void {
    return switch (vkBeginCommandBuffer(command_buffer, begin_info)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createCommandPool(
    device: Device,
    create_info: *const CommandPoolCreateInfo,
    allocator: ?*const AllocationCallbacks,
    command_pool: *CommandPool,
) !void {
    return switch (vkCreateCommandPool(device, create_info, allocator, command_pool)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createDevice(
    physical_device: PhysicalDevice,
    create_info: *const DeviceCreateInfo,
    allocator: ?*const AllocationCallbacks,
    device: *Device,
) !void {
    return switch (vkCreateDevice(physical_device, create_info, allocator, device)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        .VK_ERROR_EXTENSION_NOT_PRESENT => error.ExtensionNotPresent,
        .VK_ERROR_FEATURE_NOT_PRESENT => error.FeatureNotPresent,
        .VK_ERROR_TOO_MANY_OBJECTS => error.TooManyObjects,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        else => unreachable,
    };
}

pub inline fn createFence(
    device: Device,
    create_info: *const FenceCreateInfo,
    allocator: ?*const AllocationCallbacks,
    fence: *Fence,
) !void {
    return switch (vkCreateFence(device, create_info, allocator, fence)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createFramebuffer(
    device: Device,
    create_info: *const FramebufferCreateInfo,
    allocator: ?*const AllocationCallbacks,
    framebuffer: *Framebuffer,
) !void {
    return switch (vkCreateFramebuffer(device, create_info, allocator, framebuffer)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createGraphicsPipelines(
    device: Device,
    pipeline_cache: PipelineCache,
    create_infos: []const GraphicsPipelineCreateInfo,
    allocator: ?*const AllocationCallbacks,
    pipelines: []Pipeline,
) !Success {
    return switch (vkCreateGraphicsPipelines(device, pipeline_cache, @intCast(u32, create_infos.len), create_infos.ptr, allocator, pipelines.ptr)) {
        .VK_SUCCESS => .Success,
        .VK_PIPELINE_COMPILE_REQUIRED_EXT => .PipelineCompileRequired,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_INVALID_SHADER_NV => error.InvalidShader,
        else => unreachable,
    };
}

pub inline fn createImageView(
    device: Device,
    create_info: *const ImageViewCreateInfo,
    allocator: ?*const AllocationCallbacks,
    view: *ImageView,
) !void {
    return switch (vkCreateImageView(device, create_info, allocator, view)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createInstance(
    create_info: *const InstanceCreateInfo,
    allocator: ?*const AllocationCallbacks,
    instance: *Instance,
) !void {
    return switch (vkCreateInstance(create_info, allocator, instance)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        .VK_ERROR_LAYER_NOT_PRESENT => error.LayerNotPresent,
        .VK_ERROR_EXTENSION_NOT_PRESENT => error.ExtensionNotPresent,
        .VK_ERROR_INCOMPATIBLE_DRIVER => error.IncompatibleDriver,
        else => unreachable,
    };
}

pub inline fn createPipelineLayout(
    device: Device,
    create_info: *const PipelineLayoutCreateInfo,
    allocator: ?*const AllocationCallbacks,
    pipeline_layout: *PipelineLayout,
) !void {
    return switch (vkCreatePipelineLayout(device, create_info, allocator, pipeline_layout)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createRenderPass(
    device: Device,
    create_info: *const RenderPassCreateInfo,
    allocator: ?*const AllocationCallbacks,
    render_pass: *RenderPass,
) !void {
    return switch (vkCreateRenderPass(device, create_info, allocator, render_pass)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createSemaphore(
    device: Device,
    create_info: *const SemaphoreCreateInfo,
    allocator: ?*const AllocationCallbacks,
    semaphore: *Semaphore,
) !void {
    return switch (vkCreateSemaphore(device, create_info, allocator, semaphore)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn createShaderModule(
    device: Device,
    create_info: *const ShaderModuleCreateInfo,
    allocator: ?*const AllocationCallbacks,
    shader_module: *ShaderModule,
) !void {
    return switch (vkCreateShaderModule(device, create_info, allocator, shader_module)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_INVALID_SHADER_NV => error.InvalidShader,
        else => unreachable,
    };
}

pub inline fn createSwapchain(
    device: Device,
    create_info: *const SwapchainCreateInfo,
    allocator: ?*const AllocationCallbacks,
    swapchain: *Swapchain,
) !void {
    return switch (vkCreateSwapchainKHR(device, create_info, allocator, swapchain)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        .VK_ERROR_NATIVE_WINDOW_IN_USE_KHR => error.NativeWindowInUse,
        .VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        else => unreachable,
    };
}

pub inline fn deviceWaitIdle(
    device: Device,
) !void {
    return switch (vkDeviceWaitIdle(device)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        else => unreachable,
    };
}

pub inline fn endCommandBuffer(
    command_buffer: CommandBuffer,
) !void {
    return switch (vkEndCommandBuffer(command_buffer)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn enumerateDeviceExtensionProperties(
    physical_device: PhysicalDevice,
    layer_name: ?*const u8,
    property_count: *u32,
    properties: ?[]ExtensionProperties,
) !Success {
    return switch (vkEnumerateDeviceExtensionProperties(physical_device, layer_name, property_count, if (properties) |props| props.ptr else null)) {
        .VK_SUCCESS => .Success,
        .VK_INCOMPLETE => .Incomplete,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_LAYER_NOT_PRESENT => error.LayerNotPresent,
        else => unreachable,
    };
}

pub inline fn enumerateInstanceLayerProperties(
    property_count: *u32,
    properties: ?[]LayerProperties,
) !Success {
    return switch (vkEnumerateInstanceLayerProperties(property_count, if (properties) |props| props.ptr else null)) {
        .VK_SUCCESS => .Success,
        .VK_INCOMPLETE => .Incomplete,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn enumeratePhysicalDevices(
    instance: Instance,
    physical_device_count: *u32,
    physical_devices: ?[]PhysicalDevice,
) !Success {
    return switch (vkEnumeratePhysicalDevices(instance, physical_device_count, if (physical_devices) |devs| devs.ptr else null)) {
        .VK_SUCCESS => .Success,
        .VK_INCOMPLETE => .Incomplete,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        else => unreachable,
    };
}

pub inline fn getPhysicalDeviceSurfaceCapabilities(
    physical_device: PhysicalDevice,
    surface: Surface,
    surface_capabilities: *SurfaceCapabilities,
) !void {
    return switch (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, surface_capabilities)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        else => unreachable,
    };
}

pub inline fn getPhysicalDeviceSurfaceFormats(
    physical_device: PhysicalDevice,
    surface: Surface,
    surface_format_count: *u32,
    surface_formats: ?[]SurfaceFormat,
) !Success {
    return switch (vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, surface_format_count, if (surface_formats) |fmts| fmts.ptr else null)) {
        .VK_SUCCESS => .Success,
        .VK_INCOMPLETE => .Incomplete,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        else => unreachable,
    };
}

pub inline fn getPhysicalDeviceSurfacePresentModes(
    physical_device: PhysicalDevice,
    surface: Surface,
    present_mode_count: *u32,
    present_modes: ?[]PresentMode,
) !Success {
    return switch (vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, present_mode_count, if (present_modes) |modes| modes.ptr else null)) {
        .VK_SUCCESS => .Success,
        .VK_INCOMPLETE => .Incomplete,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        else => unreachable,
    };
}

pub inline fn getPhysicalDeviceSurfaceSupport(
    physical_device: PhysicalDevice,
    queue_family_index: u32,
    surface: Surface,
    supported: *align(@sizeOf(Bool32)) bool,
) !void {
    return switch (vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index, surface, @ptrCast(*Bool32, supported))) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        else => unreachable,
    };
}

pub inline fn getSwapchainImages(
    device: Device,
    swapchain: Swapchain,
    swapchain_image_count: *u32,
    swapchain_images: ?[]Image,
) !Success {
    return switch (vkGetSwapchainImagesKHR(device, swapchain, swapchain_image_count, if (swapchain_images) |imgs| imgs.ptr else null)) {
        .VK_SUCCESS => .Success,
        .VK_INCOMPLETE => .Incomplete,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn queuePresent(
    queue: Queue,
    present_info: *const PresentInfo,
) !Success {
    return switch (vkQueuePresentKHR(queue, present_info)) {
        .VK_SUCCESS => .Success,
        .VK_SUBOPTIMAL_KHR => .Suboptimal,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        .VK_ERROR_OUT_OF_DATE_KHR => error.OutOfDate,
        .VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        .VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => error.FullScreenExclusiveModeLost,
        else => unreachable,
    };
}

pub inline fn queueSubmit(
    queue: Queue,
    submits: []const SubmitInfo,
    fence: Fence,
) !void {
    return switch (vkQueueSubmit(queue, @intCast(u32, submits.len), submits.ptr, fence)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        else => unreachable,
    };
}

pub inline fn resetFences(
    device: Device,
    fences: []const Fence,
) !void {
    return switch (vkResetFences(device, @intCast(u32, fences.len), fences.ptr)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        else => unreachable,
    };
}

pub inline fn waitForFences(
    device: Device,
    fences: []const Fence,
    wait_all: bool,
    timeout: u64,
) !Success {
    return switch (vkWaitForFences(device, @intCast(u32, fences.len), fences.ptr, @boolToInt(wait_all), timeout)) {
        .VK_SUCCESS => .Success,
        .VK_TIMEOUT => .Timeout,
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        .VK_ERROR_OUT_OF_DEVICE_MEMORY => error.OutOfDeviceMemory,
        .VK_ERROR_DEVICE_LOST => error.DeviceLost,
        else => unreachable,
    };
}

pub inline fn createDebugUtilsMessenger(
    instance: Instance,
    create_info: *const DebugUtilsMessengerCreateInfo,
    allocator: ?*const AllocationCallbacks,
    messenger: *DebugUtilsMessenger,
) !void {
    const vkCreateDebugUtilsMessenger = @ptrCast(
        PFN_vkCreateDebugUtilsMessengerEXT,
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"),
    ) orelse return error.ExtensionNotPresent;

    return switch (vkCreateDebugUtilsMessenger(instance, create_info, allocator, messenger)) {
        .VK_SUCCESS => {},
        .VK_ERROR_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        else => unreachable,
    };
}

pub inline fn destroyDebugUtilsMessenger(
    instance: Instance,
    messenger: DebugUtilsMessenger,
    allocator: ?*const AllocationCallbacks,
) void {
    const vkDestroyDebugUtilsMessenger = @ptrCast(
        PFN_vkDestroyDebugUtilsMessengerEXT,
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"),
    ) orelse return;

    vkDestroyDebugUtilsMessenger(instance, messenger, allocator);
}

/// Constructs an API version number with major, minor and patch version numbers.
pub inline fn makeVersion(major: u10, minor: u10, patch: u12) u32 {
    return @as(u32, major) << 22 | @as(u22, minor) << 12 | patch;
}

/// Status of commands reported via VkResult return values and represented by
/// successful completion codes.
const Success = enum {
    Success,
    NotReady,
    Timeout,
    EventSet,
    EventReset,
    Incomplete,
    Suboptimal,
    ThreadIdle,
    ThreadDone,
    OperationDeferred,
    OperationNotDeferred,
    PipelineCompileRequired,
};

/// Status of commands reported via VkResult return values and represented by
/// run time error codes.
const Error = error{
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

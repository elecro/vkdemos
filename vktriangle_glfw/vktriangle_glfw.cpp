/**
 * Single file Vulkan triangle example with GLFW.
 * The example renders in memory then copies it to a "readable" image and
 * saves the result into a binary PPM image file.
 *
 * The example uses a single vertex input.
 * Look for the "V.X." comments to see the vertex input handling ("X" is a number).
 *
 * Compile without shaderc:
 * $ g++ vktriangle_glfw.cpp -o triangle_glfw -lvulkan -lglfw -std=c++11
 *
 * Re-Compile shaders (optional):
 * $ glslangValidator -V passthrough.vert -o passthrough.vert.spv
 * $ glslangValidator -V passthrough.frag -o passthrough.frag.spv
 *
 * Compile with shaderc:
 * $ g++ vktriangle_glfw.cpp -o triangle_glfw -lvulkan -lglfw -lshaderc_shared -std=c++11 -DHAVE_SHADERC=1
 *
 * Run: the "passthrough.{vert,frag}*" files must be in the same dir.
 * $ ./triangle_glfw
 *
 * Env variables:
 * DEMO_USE_VALIDATION: Enables (1) or disables (0) the usage of validation layers. Default: 0
 * DEMO_OUTPUT: Output PPM file name. Default: out.ppm
 *
 * Dependencies:
 *  * C++11
 *  * Vulkan 1.0
 *  * Vulkan loader
 *  * GLFW
 *  * One of the following:
 *    * glslangValidator (HAVE_SHADERC=0)
 *    * shaderc with glslang (HAVE_SHADERC=1)
 *
 * Includes:
 *  * Validation layer enable.
 *  * PPM image output.
 *
 * Excludes:
 *  * No swapchain.
 *
 * MIT License
 * Copyright (c) 2020 elecro
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * OFTWARE.
 */
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <libgen.h>
#include <unistd.h>
#include <limits.h>

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#ifndef HAVE_SHADERC
#define HAVE_SHADERC 0
#endif

#if HAVE_SHADERC
#include <shaderc/shaderc.hpp>
#endif

const std::vector<const char*> g_validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> g_swapchainDeviceExtension = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static uint32_t FindQueueFamily(const VkPhysicalDevice device, const VkSurfaceKHR surface, bool *hasIdx);
static uint32_t FindMemoryType(const VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

#if HAVE_SHADERC
static std::vector<char> LoadGLSL(const std::string name);
static std::vector<uint32_t> CompileGLSL(shaderc_shader_kind shaderType, const std::vector<char> &vertSrc);
#else
static std::vector<uint32_t> LoadSPIRV(const std::string name);
#endif

static void CopyImageToLinearImage(const VkPhysicalDevice physicalDevice,
                                   const VkDevice device,
                                   const VkQueue queue,
                                   const VkCommandPool cmdPool,
                                   const VkImage renderImage,
                                   float renderImageWidth,
                                   float renderImageHeight,
                                   VkImage *outImage,
                                   VkDeviceMemory *outMemory);

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const char *envValidation = getenv("DEMO_USE_VALIDATION");
    const char *envOutputName = getenv("DEMO_OUTPUT");

    bool enableValidationLayers = ((envValidation != NULL) && (strncmp("1", envValidation, 2) == 0));
    const char *outputFileName = "out.ppm";

    if (envOutputName != NULL) {
        outputFileName = envOutputName;
    }

    printf("Validation: %s\n", (enableValidationLayers ? "ON" : "OFF"));
    printf("Using shaderc: %s\n", (HAVE_SHADERC ? "YES" : "NO"));
    printf("Output: %s\n", outputFileName);

    // G.0. Initialize GLFW.
    {
        glfwInit();

        printf("GLFW Vulkan supported: %s\n", (glfwVulkanSupported() ? "YES" : "NO"));
    }

    // G.1. Create a Window.
    uint32_t windowWidth = 512;
    uint32_t windowHeight = 512;
    GLFWwindow* window;
    {
        // With GLFW_CLIENT_API set to GLFW_NO_API there will be no OpenGL (ES) context.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(windowWidth, windowHeight, "vktriangle GLFW", NULL, NULL);
    }

    // 1. Create Vulkan Instance.
    // A Vulkan instance is the base for all other Vulkan API calls.
    // This is similar an the OpenGL context.
    VkInstance instance;
    {
        std::vector<const char*> extensions{};

        // G.2. Add VK_KHR_surface extensions for the instance creation.
        // With this a presentation surface can be accessed.
        uint32_t count;
        const char** surfaceExtensions = glfwGetRequiredInstanceExtensions(&count);
        for (uint32_t idx = 0; idx < count; idx++) {
            extensions.push_back(surfaceExtensions[idx]);
        }

        // 1.1. Specify the application infromation.
        // One important info is the "apiVersion"
        VkApplicationInfo appInfo;
        {
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "MinimalVkTriangle";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "RAW";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;
        }

        // 1.2. Specify the Instance creation information.
        // The Instance level Validation and debug layers must be specified here.
        VkInstanceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.pApplicationInfo = &appInfo;
            createInfo.enabledLayerCount = 0;

            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                createInfo.ppEnabledLayerNames = g_validationLayers.data();
                // If a debug callbacks should be enabled:
                //  * The extension must be specified and
                //  * The "pNext" should point to a valid "VkDebugUtilsMessengerCreateInfoEXT" struct.
                // extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                //createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugInfo;
            }

            createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            createInfo.ppEnabledExtensionNames = extensions.data();
        }

        // 1.3. Create the Vulkan instance.
        if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    // G.3. Create a Vulkan Surface using GLFW.
    // By using GLFW the current windowing system's surface is created (xcb, win32, etc..)
    VkSurfaceKHR surface;
    {
        if (glfwCreateWindowSurface(instance, window, NULL, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // 2. Select PhysicalDevice and Queue Family Index.
    VkPhysicalDevice physicalDevice;
    uint32_t graphicsQueueFamilyIdx;
    {
        // 2.1 Query the number of physical devices.
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        // 2.2. Get all avaliable physical devices.
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // 2.3. Select a physical device (based on some info).
        // Currently the first physical device is selected if it supports Graphics Queue.
        for (const VkPhysicalDevice& device : devices) {
            bool hasIdx;
            graphicsQueueFamilyIdx = FindQueueFamily(device, surface, &hasIdx);
            if (hasIdx) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    // 3. Create a logical Vulkan Device.
    // Most Vulkan API calls require a logical device.
    // To use device level layer, they should be provided here.
    VkDevice device;
    {
        // 3.1. Build the device queue create info data (use only a singe queue).
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo;
        {
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.pNext = NULL;
            queueCreateInfo.flags = 0;
            queueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIdx;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
        }

        // 3.2. The queue family/families must be provided to allow the device to use them.
        std::vector<uint32_t> uniqueQueueFamilies = { graphicsQueueFamilyIdx };

        // G.X. TODO: add device swapchane extension check.

        // 3.3. Specify the device creation information.
        VkDeviceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.queueCreateInfoCount = 1;
            createInfo.pQueueCreateInfos = &queueCreateInfo;
            createInfo.pEnabledFeatures = NULL;
            // G.4. Specify the swapchain extension when creating a VkDevice.
            createInfo.enabledExtensionCount = (uint32_t)g_swapchainDeviceExtension.size();
            createInfo.ppEnabledExtensionNames = g_swapchainDeviceExtension.data();
            createInfo.enabledLayerCount = 0;

            if (enableValidationLayers) {
                // To have device level validation information, the layers are added here.
                createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                createInfo.ppEnabledLayerNames = g_validationLayers.data();
            }
        }

        // 3.4. Create the logical device.
        if (vkCreateDevice(physicalDevice, &createInfo, NULL, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
    }

    // 4. Get the selected Queue family's first queue.
    // A Queue is used to issue recorded command buffers to the GPU for execution.
    VkQueue queue;
    {
        vkGetDeviceQueue(device, graphicsQueueFamilyIdx, 0, &queue);
    }

    // G.5. Create the Swapchain.
    // Creating a correct Swapchain requires querying a few things.
    // Like: surface format, max/min size, presentation mode.
    VkSurfaceFormatKHR surfaceFormat;
    // By standard the FIFO presentation mode should always be available.
    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D swapExtent = { windowWidth, windowHeight };

    VkSwapchainKHR swapchain;
    {
        // G.5.1. Query the surface capabilities and select the swapchain's extent (width, height).
        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        {
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilities);

            if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
                swapExtent = surfaceCapabilities.currentExtent;
            }
        }

        // G.5.2. Select a surface format.
        {
            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, NULL);

            std::vector<VkSurfaceFormatKHR> surfaceFormats;
            surfaceFormats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, surfaceFormats.data());

            for (VkSurfaceFormatKHR &entry : surfaceFormats) {
                if ((entry.format == VK_FORMAT_B8G8R8A8_SRGB) && (entry.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) {
                    surfaceFormat = entry;
                    break;
                }
            }
        }

        // G.5.3. TODO: Select a presentation mode, if the FIFO is not good.
        /*
        {
            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, NULL);

            std::vector<VkPresentModeKHR> presentModes;
            presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());
        }
        */

        // G.5.4. Specify the number of images in the swap chain.
        // For better performance using "min + 1";
        uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
        VkSwapchainCreateInfoKHR createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.surface = surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = swapExtent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            // If the graphics and presentation queue is different this should not be exclusive.
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = NULL;
            createInfo.preTransform = surfaceCapabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            createInfo.presentMode = swapchainPresentMode;
            createInfo.clipped = VK_TRUE;
            createInfo.oldSwapchain = VK_NULL_HANDLE;
        }

        if (vkCreateSwapchainKHR(device, &createInfo, NULL, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
    }

    // G.6. Get the Swapchain images.
    std::vector<VkImage> swapImages;
    {
        uint32_t imageCount;
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, NULL);
        swapImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapImages.data());
    }

    // Old 5. and 6. steps are removed.
    // The Swapchain creation takes care of the render target image creation.
    uint32_t renderImageWidth = swapExtent.width;
    uint32_t renderImageHeight = swapExtent.height;

    // G.7. Create Image Views for the Swapchain Images.
    // This replaces the old 7. step.
    std::vector<VkImageView> swapImageViews;
    swapImageViews.resize(swapImages.size());
    {
        VkImageViewCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            //createInfo.image = renderImage;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = surfaceFormat.format;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
        }

        for (size_t idx = 0; idx < swapImages.size(); idx++) {
            createInfo.image = swapImages[idx];

            if (vkCreateImageView(device, &createInfo, NULL, &swapImageViews[idx]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    // V.0. Prepare the Vertex Coordinates.
    std::vector<float> vertexCoordinates = {
         0.0, -0.5,
         0.5,  0.5,
        -0.5,  0.5
    };

    // V.1. Create the Vulkan buffer which will hold the Vertex Input data.
    // This buffer will hold the Vertex coordinates in a vec2 like format.
    VkBuffer vertexBuffer;
    {
        VkBufferCreateInfo bufferInfo;
        {
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.pNext = NULL;
            bufferInfo.flags = 0;
            bufferInfo.size = sizeof(float) * vertexCoordinates.size();
            // The buffer will be used as a Vertex Input attribute.
            bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            bufferInfo.queueFamilyIndexCount = 0;
            bufferInfo.pQueueFamilyIndices = NULL;
        }

        if (vkCreateBuffer(device, &bufferInfo, NULL, &vertexBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vertex buffer!");
        }
    }

    // V.2. Allocate and bind the memory for the Vertex Buffer.
    // For each Buffer a memory should be allocated on the GPU otherwise it can't be used.
    VkDeviceMemory vertexBufferMemory;
    {
        // 6.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

        // 6.2 Find a memory type based on the requirements.
        // Here a device (gpu) local memory type is requested (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
        uint32_t memoryTypeIndex = FindMemoryType(physicalDevice,
                                                  memRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        // 6.3. Based on the memory requirements specify the allocation information.
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // 6.3 Allocate the memory.
        if (vkAllocateMemory(device, &allocInfo, NULL, &vertexBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // 6.4 "Connect" the Vertex buffer with the allocated memory.
        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
    }

    // V.3. Upload the Vertex Buffer data.
    {
        // V.3.1. Map buffer memory to "data" pointer.
        void *data;
        if (vkMapMemory(device, vertexBufferMemory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("failed to map vertex buffer!");
        }

        // V.3.2. Copy data into the "data".
        ::memcpy(data, vertexCoordinates.data(), sizeof(float) * vertexCoordinates.size());

        // V.3.3. Flush the data.
        // This is required as a non-coherent buffer was created.
        VkMappedMemoryRange memoryRange;
        {
            memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            memoryRange.pNext = NULL;
            memoryRange.memory = vertexBufferMemory;
            memoryRange.offset = 0;
            memoryRange.size = VK_WHOLE_SIZE;
        }
        vkFlushMappedMemoryRanges(device, 1, &memoryRange);

        // V.3.4. Unmap the mapped vertex buffer.
        // After this the "data" pointer is a non-valid pointer.
        vkUnmapMemory(device, vertexBufferMemory);
    }

    // 8. Create a Render Pass.
    // A Render Pass is required to use vkCmdDraw* commands.
    VkRenderPass renderPass;
    {
        VkAttachmentDescription colorAttachment;
        {
            colorAttachment.flags = 0;
            colorAttachment.format = surfaceFormat.format;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            // G.XX. To present an image to a surface the layout should be in VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkAttachmentReference colorAttachmentRef;
        {
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkSubpassDescription subpass;
        {
            subpass.flags = 0;
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.inputAttachmentCount = 0;
            subpass.pInputAttachments = NULL;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;
            subpass.pResolveAttachments = NULL;
            subpass.pDepthStencilAttachment = NULL;
            subpass.preserveAttachmentCount = 0;
            subpass.pPreserveAttachments = NULL;
        }

        VkRenderPassCreateInfo renderPassInfo;
        {
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.pNext = NULL;
            renderPassInfo.flags = 0;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = 0;
            renderPassInfo.pDependencies = NULL;
        }

        if (vkCreateRenderPass(device, &renderPassInfo, NULL, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    char binaryPath[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", binaryPath, PATH_MAX);
    char *binaryDir = dirname(binaryPath);
    std::string sourceDir = std::string(binaryDir);

    // 9. Create Vertex shader.
    VkShaderModule vertShaderModule;
    {
        // 9.1. Load the GLSL shader from file and compile it with shaderc.
        #if HAVE_SHADERC
        std::vector<char> vertSrc = LoadGLSL(sourceDir + "/passthrough.vert");
        std::vector<uint32_t> vertCode = CompileGLSL(shaderc_vertex_shader, vertSrc);
        #else
        std::vector<uint32_t> vertCode = LoadSPIRV(sourceDir + "/passthrough.vert.spv");
        #endif

        if (vertCode.size() == 0) {
            throw std::runtime_error("failed to load vertex shader!");
        }

        // 9.2. Specify the vertex shader module information.
        // Notes:
        // * "codeSize" is in bytes.
        // * "pCode" points to an array of SPIR-V opcodes.
        VkShaderModuleCreateInfo vertInfo;
        {
            vertInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            vertInfo.pNext = NULL;
            vertInfo.flags = 0;
            vertInfo.codeSize = vertCode.size() * sizeof(uint32_t);
            vertInfo.pCode = reinterpret_cast<uint32_t*>(vertCode.data());
        }

        // 9.3. Create the Vertex Shader Module.
        if (vkCreateShaderModule(device, &vertInfo, NULL, &vertShaderModule) != VK_SUCCESS) {
           throw std::runtime_error("failed to create shader module!");
        }
    }

    // 10. Create Fragment shader.
    VkShaderModule fragShaderModule;
    {
        // 10.1. Load the GLSL shader from file and compile it with shaderc.
        #if HAVE_SHADERC
        std::vector<char> fragSrc = LoadGLSL(sourceDir + "/passthrough.frag");
        std::vector<uint32_t> fragCode = CompileGLSL(shaderc_fragment_shader, fragSrc);
        #else
        std::vector<uint32_t> fragCode = LoadSPIRV(sourceDir + "/passthrough.frag.spv");
        #endif

        if (fragCode.size() == 0) {
            throw std::runtime_error("failed to load fragment shader!");
        }

        // 10.2. Specify the fragment shader module information.
        VkShaderModuleCreateInfo fragInfo;
        {
            fragInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fragInfo.pNext = NULL;
            fragInfo.flags = 0;
            fragInfo.codeSize = fragCode.size() * sizeof(uint32_t);
            fragInfo.pCode = reinterpret_cast<uint32_t*>(fragCode.data());
        }

        // 10.3. Create the Fragment Shader Module.
        if (vkCreateShaderModule(device, &fragInfo, NULL, &fragShaderModule) != VK_SUCCESS) {
           throw std::runtime_error("failed to create shader module!");
        }
    }

    // 11. Create Pipeline Layout.
    // Currently there are no descriptors added (no uniforms).
    VkPipelineLayout pipelineLayout;
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo;
        {
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.pNext = NULL;
            pipelineLayoutInfo.flags = 0;
            pipelineLayoutInfo.setLayoutCount = 0;
            pipelineLayoutInfo.pSetLayouts = NULL;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
        }

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    // 12. Create the Rendering Pipeline
    VkPipeline pipeline;
    {
        VkPipelineShaderStageCreateInfo vertShaderStageInfo;
        {
            vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertShaderStageInfo.pNext = NULL;
            vertShaderStageInfo.flags = 0;
            vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";
            vertShaderStageInfo.pSpecializationInfo = NULL;
        }

        VkPipelineShaderStageCreateInfo fragShaderStageInfo;
        {
            fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageInfo.pNext = NULL;
            fragShaderStageInfo.flags = 0;
            fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";
            fragShaderStageInfo.pSpecializationInfo = NULL;
        }

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // V.4. Describe the Vertex input binding information.
        VkVertexInputBindingDescription vec2VertexBinding;
        {
            // The binding information is mapped to the VkVertexInputAttributeDescription.binding.
            vec2VertexBinding.binding = 0;
            // The stride information is based on the vertex input type: vec2. (see shader)
            vec2VertexBinding.stride = sizeof(float) * 2;
            vec2VertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        }

        // V.5. Describe the Vertex input attribute information.
        VkVertexInputAttributeDescription positionVertexAttribute;
        {
            positionVertexAttribute.binding = 0;
            // The attribute location is from the Vertex shader.
            positionVertexAttribute.location = 0;
            // Format and offset is used during the data read from the buffer.
            positionVertexAttribute.format = VK_FORMAT_R32G32_SFLOAT;
            positionVertexAttribute.offset = 0;
        }

        // V.6. Connect the Attribute and Binding infors to the VertexInputState.
        VkPipelineVertexInputStateCreateInfo vertexInputInfo;
        {
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.pNext = NULL;
            vertexInputInfo.flags = 0;
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.pVertexBindingDescriptions = &vec2VertexBinding;
            vertexInputInfo.vertexAttributeDescriptionCount = 1;
            vertexInputInfo.pVertexAttributeDescriptions = &positionVertexAttribute;
        }

        VkPipelineInputAssemblyStateCreateInfo inputAssembly;
        {
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.pNext = NULL;
            inputAssembly.flags = 0;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;
        }

        VkViewport viewport;
        {
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float) swapExtent.width;
            viewport.height = (float) swapExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
        }

        VkRect2D scissor;
        {
            scissor.offset = { 0, 0 };
            scissor.extent = swapExtent;
        }

        VkPipelineViewportStateCreateInfo viewportState{};
        {
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;
        }

        VkPipelineRasterizationStateCreateInfo rasterizer;
        {
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.pNext = NULL;
            rasterizer.flags = 0;
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0;
            rasterizer.depthBiasClamp = 0.0;
            rasterizer.depthBiasSlopeFactor = 0.0;
            rasterizer.lineWidth = 1.0f;
        }

        VkPipelineMultisampleStateCreateInfo multisampling;
        {
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.pNext = NULL;
            multisampling.flags = 0;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.minSampleShading = 0.0;
            multisampling.pSampleMask = NULL;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;
        }

        VkPipelineColorBlendAttachmentState colorBlendAttachment;
        {
            colorBlendAttachment.blendEnable = VK_FALSE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
                                                  | VK_COLOR_COMPONENT_G_BIT
                                                  | VK_COLOR_COMPONENT_B_BIT
                                                  | VK_COLOR_COMPONENT_A_BIT;
        }

        VkPipelineColorBlendStateCreateInfo colorBlending;
        {
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.pNext = NULL;
            colorBlending.flags = 0;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = VK_LOGIC_OP_COPY;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;
            colorBlending.blendConstants[0] = 0.0f;
            colorBlending.blendConstants[1] = 0.0f;
            colorBlending.blendConstants[2] = 0.0f;
            colorBlending.blendConstants[3] = 0.0f;
        }

        VkGraphicsPipelineCreateInfo pipelineInfo;
        {
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.pNext = NULL;
            pipelineInfo.flags = 0;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pTessellationState = NULL;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = NULL;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = NULL;
            pipelineInfo.layout = pipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo.basePipelineIndex = 0;
        }

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    // G.8. Create Frambuffer for each Swapchain Image view.
    // Replaces old 13. step.
    std::vector<VkFramebuffer> framebuffers;
    {
        framebuffers.resize(swapImageViews.size());

        VkFramebufferCreateInfo framebufferInfo;
        {
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.pNext = NULL;
            framebufferInfo.flags = 0;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            //framebufferInfo.pAttachments = &renderImageView;
            framebufferInfo.width = swapExtent.width;
            framebufferInfo.height = swapExtent.height;
            framebufferInfo.layers = 1;
        }

        for (size_t idx = 0; idx < framebuffers.size(); idx++) {
            framebufferInfo.pAttachments = &swapImageViews[idx];

            if (vkCreateFramebuffer(device, &framebufferInfo, NULL, &framebuffers[idx]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    // 14. Create Command Pool.
    // Required to create Command buffers.
    VkCommandPool cmdPool;
    {
        VkCommandPoolCreateInfo poolInfo;
        {
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.pNext = NULL;
            poolInfo.flags = 0;
            poolInfo.queueFamilyIndex = graphicsQueueFamilyIdx;
        }

        if (vkCreateCommandPool(device, &poolInfo, NULL, &cmdPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // G.9. Create a Command Buffer for each Swapchain Image View (Framebuffer).
    std::vector<VkCommandBuffer> cmdBuffers;
    {
        cmdBuffers.resize(swapImageViews.size());

        VkCommandBufferAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.commandPool = cmdPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = cmdBuffers.size();
        }

        if (vkAllocateCommandBuffers(device, &allocInfo, cmdBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // Start recording draw commands.
    // G.10. In the current example all Command Buffers will have the same data.

    // 16. Start Command Buffer
    // G.11. Start all Command Buffers.
    for (size_t idx = 0; idx < cmdBuffers.size(); idx++)
    {
        VkCommandBufferBeginInfo beginInfo;
        {
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.pNext = NULL;
            // G.XX. As a command buffer is submitted multiple times the VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT flag
            // can't be used.
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = NULL;
        }

        if (vkBeginCommandBuffer(cmdBuffers[idx], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
    }

    // 17. Insert draw commands into Command Buffer.
    // G.12. Insert same draw commands into all Command Buffers.
    for (size_t idx = 0; idx < cmdBuffers.size(); idx++)
    {
        // 17.1. Add Begin RenderPass command
        // This makes it possible to use the vmCmdDraw* calls.
        VkClearValue clearColor = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
        VkRenderPassBeginInfo renderPassInfo;
        {
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.pNext = NULL;
            renderPassInfo.renderPass = renderPass;
            // G.12.1. Each Command Buffer will use a different Framebuffer
            renderPassInfo.framebuffer = framebuffers[idx];
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = { (uint32_t)renderImageWidth, (uint32_t)renderImageHeight };
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;
        }

        vkCmdBeginRenderPass(cmdBuffers[idx], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // 17.2. Bind the Graphics pipeline inside the Current Render Pass.
        vkCmdBindPipeline(cmdBuffers[idx], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        // V.7. Bind the Vertex buffers as specified by the pipeline.
        VkDeviceSize bufferOffsets[] = { 0 };
        vkCmdBindVertexBuffers(cmdBuffers[idx], 0, 1, &vertexBuffer, bufferOffsets);

        // 17.3. Add a Draw command.
        // Draw 3 vertices using the pipeline bound previously.
        uint32_t vertexCount = 3;
        uint32_t instanceCount = 1;
        // G.XX. Each Command buffer uses a different starting Instance Index (idx).
        // This will be used in the shader to provide a bit of animation.
        vkCmdDraw(cmdBuffers[idx], vertexCount, instanceCount, 0, idx);

        // 17.4. End the Render Pass.
        vkCmdEndRenderPass(cmdBuffers[idx]);
    }

    // 18. End the Command Buffer recording.
    // G.13. End all Command Buffers.
    for (size_t idx = 0; idx < cmdBuffers.size(); idx++)
    {
        if (vkEndCommandBuffer(cmdBuffers[idx]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // Recording of the draw commands into the Command Buffer is done.
    // Now the Command Buffer should be sent to the GPU.

    const uint32_t imagesInFlight = 2;

    // G.14. As there are multiple images now more sync objects are required.
    // This replaces the old 19. step.
    // imageAvailableSemaphores will store the Semaphores to check if the next image is available to draw on to.
    std::vector<VkSemaphore> imageAvailableSemaphores;
    // renderFinishedSemaphores will store the Semaphores to check if the image finished rendering.
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> swapImagesFences;
    std::vector<VkFence> activeFences;
    {
        imageAvailableSemaphores.resize(imagesInFlight);
        renderFinishedSemaphores.resize(imagesInFlight);
        activeFences.resize(imagesInFlight);
        swapImagesFences.resize(swapImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo;
        {
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphoreInfo.pNext = NULL;
            semaphoreInfo.flags = 0;
        }

        VkFenceCreateInfo fenceInfo;
        {
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.pNext = 0;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;;
        }

        for (uint32_t idx = 0; idx < imagesInFlight; idx++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, NULL, &imageAvailableSemaphores[idx]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, NULL, &renderFinishedSemaphores[idx]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, NULL, &activeFences[idx]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    // G.25. Draw and Present loop.
    // Draw and Present a series of images.
    uint32_t activeSyncIdx = 0;
    while (!glfwWindowShouldClose(window)) {
        // G.25.0. Run GLFW event polling.
        glfwPollEvents();

        // G.25.1. Wait for the previous fence to "finish".
        vkWaitForFences(device, 1, &activeFences[activeSyncIdx], VK_TRUE, UINT64_MAX);

        // G.25.2. Get the next Swapchain Image Index.
        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[activeSyncIdx], VK_NULL_HANDLE, &imageIndex);

        // G.25.3. Wait for the target image to be available.
        if (swapImagesFences[imageIndex] != VK_NULL_HANDLE) {
            vkWaitForFences(device, 1, &swapImagesFences[imageIndex], VK_TRUE, UINT64_MAX);
        }
        // Connect the current fence to the given swapchaing image.
        swapImagesFences[imageIndex] = activeFences[activeSyncIdx];

        // Configure a few sync points.
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[activeSyncIdx] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[activeSyncIdx] };

        // G.25.4. Build the Submit info using the sync points.
        VkSubmitInfo submitInfo;
        {
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = NULL;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuffers[imageIndex];
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;
        }

        vkResetFences(device, 1, &activeFences[activeSyncIdx]);

        // A fence is provided to have a CPU side sync point.
        if (vkQueueSubmit(queue, 1, &submitInfo, activeFences[activeSyncIdx]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit command buffer!");
        }

        VkPresentInfoKHR presentInfo;
        {
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.pNext = 0;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &swapchain;
            presentInfo.pImageIndices = &imageIndex;
            presentInfo.pResults = NULL;
        }

        vkQueuePresentKHR(queue, &presentInfo);

        activeSyncIdx = (activeSyncIdx + 1) % imagesInFlight;

        // G.XX. Artificially slow down the rendering to avoid Epileptic seizure.
        usleep((int)(0.15 * 1000000));
    }

    // At this point the image is rendered into the Framebuffer's attachment which is an ImageView.

    {
        // Start readback process of the rendered image.
        // 22. Copy the rendered image into a buffer which can be mapped and read.
        // Note: this is the most basic process to the the image into a readeable memory.
        VkImage readableImage;
        VkDeviceMemory readableImageMemory;
        CopyImageToLinearImage(physicalDevice, device, queue, cmdPool, swapImages[0], renderImageWidth, renderImageHeight, &readableImage, &readableImageMemory);

        // 23. Get layout of the readable image (including row pitch).
        VkImageSubresource subResource;
        {
            subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subResource.mipLevel = 0;
            subResource.arrayLayer = 0;
        }
        VkSubresourceLayout subResourceLayout;
        vkGetImageSubresourceLayout(device, readableImage, &subResource, &subResourceLayout);

        // 24. Map image memory so we can start copying from it.
        const uint8_t* data;
        {
            vkMapMemory(device, readableImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
            data += subResourceLayout.offset;
        }

        // 25. Write out the image to a ppm file.
        {
            std::ofstream file(outputFileName, std::ios::out | std::ios::binary);
            // ppm header
            file << "P6\n" << renderImageWidth << "\n" << renderImageHeight << "\n" << 255 << "\n";

            // ppm binary pixel data
            // As the image format is R8G8B8A8 one "pixel" size is 4 bytes (uint32_t)
            for (uint32_t y = 0; y < renderImageHeight; y++) {
                uint32_t *row = (uint32_t*)data;
                for (uint32_t x = 0; x < renderImageWidth; x++) {
                    // Only copy the RGB values (3)
                    file.write((const char*)row, 3);
                    row++;
                }

                data += subResourceLayout.rowPitch;
            }
            file.close();
        }

        // 26. UnMap the readable image memory.
        vkUnmapMemory(device, readableImageMemory);

        // XX. Free linear image's memory.
        vkFreeMemory(device, readableImageMemory, NULL);

        // XX. Destory linar image memory.
        vkDestroyImage(device, readableImage, NULL);
    }

    // G.XX. Destroy Sync object.
    for (uint32_t idx = 0; idx < imagesInFlight; idx++) {
        vkDestroySemaphore(device, imageAvailableSemaphores[idx], NULL);
        vkDestroySemaphore(device, renderFinishedSemaphores[idx], NULL);
        vkDestroyFence(device, activeFences[idx], NULL);
    }

    // G.XX. Free Command Buffers.
    vkFreeCommandBuffers(device, cmdPool, cmdBuffers.size(), cmdBuffers.data());

    // XX. Destroy Command Pool
    vkDestroyCommandPool(device, cmdPool, NULL);

    // G.XX. Destory Framebuffers
    for (size_t idx = 0; idx < framebuffers.size(); idx++) {
        vkDestroyFramebuffer(device, framebuffers[idx], NULL);
    }

    // XX. Destory Pipeline.
    vkDestroyPipeline(device, pipeline, NULL);

    // XX. Destory Fragment Shader Module.
    vkDestroyShaderModule(device, fragShaderModule, NULL);

    // XX. Destory Vertex Shader Module.
    vkDestroyShaderModule(device, vertShaderModule, NULL);

    // XX. Destory Pipeline Layout.
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);

    // XX. Destory Render Pass.
    vkDestroyRenderPass(device, renderPass, NULL);

    // XX. Free the Vertex Buffer's memory.
    vkFreeMemory(device, vertexBufferMemory, NULL);

    // XX. Destroy the Vertex Buffer.
    vkDestroyBuffer(device, vertexBuffer, NULL);

    // G.XX. Destroy swapchain image views.
    for (size_t idx = 0; idx < swapImageViews.size(); idx++) {
        vkDestroyImageView(device, swapImageViews[idx], NULL);
    }

    // G.XX. Destroy swapchain.
    vkDestroySwapchainKHR(device, swapchain, NULL);

    // XX. Destroy Device
    vkDestroyDevice(device, NULL);

    // G.XX. Destory Surface.
    vkDestroySurfaceKHR(instance, surface, NULL);

    // XY. Destroy instance
    vkDestroyInstance(instance, NULL);

    // G.XX. Destory GLFW Window.
    glfwDestroyWindow(window);

    // G.XX. Cleanup GLFW.
    glfwTerminate();

    return 0;
}

static uint32_t FindQueueFamily(const VkPhysicalDevice device, const VkSurfaceKHR surface, bool *hasIdx) {
    if (hasIdx != nullptr) {
        *hasIdx = false;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    uint32_t queueFamilyIdx = 0;
    for (const VkQueueFamilyProperties& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            if (hasIdx != NULL) {
                *hasIdx = true;
            }

            // G.XX. Check if the selected graphics queue family supports presentation.
            // At the moment the example expects that the graphics and presentation queue is the same.
            // This is not always the case.
            // TODO: add support for different graphics and presentation family indices.
            VkBool32 presentSupport;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, queueFamilyIdx, surface, &presentSupport);
            if (presentSupport) {
                return queueFamilyIdx;
            }
        }

        // TODO?: check if device supports the target surface
        queueFamilyIdx++;
    }

    return UINT32_MAX;
}

uint32_t FindMemoryType(const VkPhysicalDevice physicalDevice,
                        uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

#if HAVE_SHADERC

std::vector<char> LoadGLSL(const std::string name) {
    std::ifstream input(name, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("failed to open file:" + name);
    }

    return std::vector<char>(std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>());
}

std::vector<uint32_t> CompileGLSL(shaderc_shader_kind shaderType, const std::vector<char> &vertSrc) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(vertSrc.data(), vertSrc.size(), shaderType, "src", options);

    return { module.begin(), module.end() };
}

#else

std::vector<uint32_t> LoadSPIRV(const std::string name) {
    std::ifstream file(name, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + name);
    }

    size_t fileSize = (size_t) file.tellg();

    if ((fileSize % sizeof(uint32_t)) != 0) {
        throw std::runtime_error("spirv file is not divisable by 4: " + name);
    }

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);

    file.close();

    return buffer;
}

#endif

void CopyImageToLinearImage(const VkPhysicalDevice physicalDevice,
                            const VkDevice device,
                            const VkQueue queue,
                            const VkCommandPool cmdPool,
                            const VkImage inputImage,
                            float inputImageWidth,
                            float inputImageHeight,
                            VkImage *outImage,
                            VkDeviceMemory *outMemory) {
    // TODO: do format checks and blit support check.

    // A.1. Create a readable linear Image as the copy destination. (destination image)
    VkFormat readableImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkImage readableImage;
    {
        VkImageCreateInfo imageInfo;
        {
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.pNext = NULL;
            imageInfo.flags = 0;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = readableImageFormat;
            imageInfo.extent = { (uint32_t)inputImageWidth, (uint32_t)inputImageHeight, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.queueFamilyIndexCount = 0;
            imageInfo.pQueueFamilyIndices = NULL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        if (vkCreateImage(device, &imageInfo, NULL, &readableImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create 2D image!");
        }
    }

    *outImage = readableImage;

    // A.2 Allocate and bind the memory for linear image
    VkDeviceMemory readableImageMemory;
    {
        // A.2.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, readableImage, &memRequirements);

        // A.2.2 Find a memory type based on the requirements.
        // Here a memory which is mappable is requested.
        uint32_t memoryTypeIndex = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // A.2.3 Allocate the memory.
        if (vkAllocateMemory(device, &allocInfo, NULL, &readableImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // A.2.4 "Connect" the image with the allocated memory.
        vkBindImageMemory(device, readableImage, readableImageMemory, 0);
    }

    *outMemory = readableImageMemory;

    // A.3. Create Command Buffer to record image copy operations.
    VkCommandBuffer cmdBuffer;
    {
        VkCommandBufferAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.pNext = NULL;
            allocInfo.commandPool = cmdPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
        }

        if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // Start recording image copy commands.

    // A.4. Start Command Buffer.
    {
        VkCommandBufferBeginInfo beginInfo;
        {
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.pNext = NULL;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            beginInfo.pInheritanceInfo = NULL;
        }

        if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
    }

    // A.4. Transition destination image to transfer destination layout.
    {
        VkImageMemoryBarrier imageMemoryBarrier;
        {
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.pNext = NULL;
            imageMemoryBarrier.srcAccessMask = 0;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.srcQueueFamilyIndex = 0;
            imageMemoryBarrier.dstQueueFamilyIndex = 0;
            imageMemoryBarrier.image = readableImage;
            imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imageMemoryBarrier);
    }

    // A.5. Transition source image to transfer source layout
    {
        VkImageMemoryBarrier imageMemoryBarrier;
        {
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.pNext = NULL;
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            imageMemoryBarrier.srcQueueFamilyIndex = 0;
            imageMemoryBarrier.dstQueueFamilyIndex = 0;
            imageMemoryBarrier.image = inputImage;
            imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imageMemoryBarrier);
    }

    // A.6. Add image copy command.
    {
        // Note: requires us to manually flip components
        VkImageCopy imageCopyRegion;
        {
            imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopyRegion.srcSubresource.mipLevel = 0;
            imageCopyRegion.srcSubresource.baseArrayLayer = 0;
            imageCopyRegion.srcSubresource.layerCount = 1;
            imageCopyRegion.srcOffset = { 0, 0, 0 };
            imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageCopyRegion.dstSubresource.mipLevel = 0;
            imageCopyRegion.dstSubresource.baseArrayLayer = 0;
            imageCopyRegion.dstSubresource.layerCount = 1;
            imageCopyRegion.dstOffset = { 0, 0, 0 };
            imageCopyRegion.extent.width = inputImageWidth;
            imageCopyRegion.extent.height = inputImageHeight;
            imageCopyRegion.extent.depth = 1;
        }

        // Issue the copy command
        vkCmdCopyImage(cmdBuffer,
                       inputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       readableImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &imageCopyRegion);
    }

    // A.7. Transition destination image to general layout, which is the required layout for mapping the image memory later on.
    {
        VkImageMemoryBarrier imageMemoryBarrier;
        {
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.pNext = NULL;
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageMemoryBarrier.srcQueueFamilyIndex = 0;
            imageMemoryBarrier.dstQueueFamilyIndex = 0;
            imageMemoryBarrier.image = readableImage;
            imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imageMemoryBarrier);
    }

    // A.8. End the Command Buffer.
    {
        if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    // A.9. Create a Fence.
    VkFence fence;
    {
        VkFenceCreateInfo fenceInfo;
        {
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.pNext = 0;
            fenceInfo.flags = 0;
        }

        if (vkCreateFence(device, &fenceInfo, NULL, &fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization object!");
        }

        //vkResetFences(device, 1, &fence);
    }

    // A.10. Submit the recorded Command Buffer to the Queue.
    {
        VkSubmitInfo submitInfo;
        {
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = NULL;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = NULL;
            submitInfo.pWaitDstStageMask = NULL;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = NULL;
        }

        // A fence is provided to have a CPU side sync point.
        if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit command buffer!");
        }
    }

    // A.11. Wait the submitted Command Buffer to finish.
    {
        // -1 means to wait for ever to finish.
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, -1) != VK_SUCCESS) {
            throw std::runtime_error("failed to wait for fence!");
        }
    }

    // XX. Destroy Fence.
    vkDestroyFence(device, fence, NULL);

    // XX. Free Command Buffer.
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

/**
 * Single file Vulkan triangle example rendered into an exported/imporeted image with GLFW.
 * The example creates a thread that renders a triangle into an exported image.
 * The image is then imported in the main thread with a different instance.
 *
 * The example uses vertex data baked into the shaders.
 * Look for the "T.X." comments to see the thread related parts ("X" is a number).
 *
 * Compile without shaderc:
 * $ g++ vktriangle_external_memory.cpp -o triangle_external_memory -lvulkan -lglfw -std=c++11
 *
 * Re-Compile shaders (optional):
 * $ glslangValidator -V passthrough.vert -o passthrough.vert.spv
 * $ glslangValidator -V passthrough.frag -o passthrough.frag.spv
 *
 * Compile with shaderc:
 * $ g++ vktriangle_externak_memory.cpp -o triangle_external_memory -lvulkan -lglfw -lshaderc_shared -std=c++11 -DHAVE_SHADERC=1
 *
 * Run: the "passthrough.{vert,frag}*" files must be in the same dir.
 * $ ./triangle_external_memory
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
 *
 * Excludes:
 *  * No swapchain.
 *
 * MIT License
 * Copyright (c) 2022 elecro
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
#include <cassert>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <thread>
#include <mutex>
#include <condition_variable>

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

// Minimum set of required extensions
const std::vector<const char*> g_instanceExtensions = {
    "VK_KHR_external_memory_capabilities",
    "VK_KHR_get_physical_device_properties2",
};
const std::vector<const char*> g_deviceExtensions = {
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_fd", // _fd for Linux file descriptor
    "VK_KHR_dedicated_allocation",
    "VK_KHR_get_memory_requirements2",
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

struct VulkanThreadOptions {
    bool enableValidationLayers;
    volatile int exposedImageFd;

    std::mutex syncMutex;
    std::condition_variable signal;
};

static void *VulkanImageProducerThread(void *arg) {
    assert(arg != nullptr);
    VulkanThreadOptions *options = (VulkanThreadOptions*)arg;

    // T.1. Create a new vulkan instance in a thread.
    VkInstance threadInstance;
    {
        // T.1.1. Specify the application infromation.
        // One important info is the "apiVersion"
        VkApplicationInfo appInfo;
        {
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pNext = NULL;
            appInfo.pApplicationName = "ThreadInstance";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "RAW";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;
        }

        // T.1.2. Specify the Instance creation information.
        // The Instance level Validation and debug layers must be specified here.
        VkInstanceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.pApplicationInfo = &appInfo;
            createInfo.enabledLayerCount = 0;

            if (options->enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                createInfo.ppEnabledLayerNames = g_validationLayers.data();
                // If a debug callbacks should be enabled:
                //  * The extension must be specified and
                //  * The "pNext" should point to a valid "VkDebugUtilsMessengerCreateInfoEXT" struct.
                // extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                //createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugInfo;
            }

            createInfo.enabledExtensionCount = static_cast<uint32_t>(g_instanceExtensions.size());
            createInfo.ppEnabledExtensionNames = g_instanceExtensions.data();
        }

        // T.1.3. Create the Vulkan instance.
        if (vkCreateInstance(&createInfo, NULL, &threadInstance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create thread instance!");
        }
    }

    // T.2. Select PhysicalDevice and Queue Family Index.
    VkPhysicalDevice threadPhysicalDevice;
    uint32_t threadGraphicsQueueFamilyIdx;
    {
        // T.2.1 Query the number of physical devices.
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(threadInstance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        // T.2.2. Get all avaliable physical devices.
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(threadInstance, &deviceCount, devices.data());

        // T.2.3. Select a physical device (based on some info).
        // Currently the first physical device is selected if it supports Graphics Queue.
        for (const VkPhysicalDevice& device : devices) {
            bool hasIdx;
            threadGraphicsQueueFamilyIdx = FindQueueFamily(device, VK_NULL_HANDLE, &hasIdx);
            if (hasIdx) {
                threadPhysicalDevice = device;
                break;
            }
        }

        if (threadPhysicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    // T.3. Create a logical Vulkan Device.
    // Most Vulkan API calls require a logical device.
    // To use device level layer, they should be provided here.
    VkDevice threadDevice;
    {
        // T.3.1. Build the device queue create info data (use only a singe queue).
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo;
        {
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.pNext = NULL;
            queueCreateInfo.flags = 0;
            queueCreateInfo.queueFamilyIndex = threadGraphicsQueueFamilyIdx;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
        }

        // T.3.2. The queue family/families must be provided to allow the device to use them.
        std::vector<uint32_t> uniqueQueueFamilies = { threadGraphicsQueueFamilyIdx };

        // T.3.3. Specify the device creation information.
        VkDeviceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.queueCreateInfoCount = 1;
            createInfo.pQueueCreateInfos = &queueCreateInfo;
            createInfo.pEnabledFeatures = NULL;
            // G.4. Specify the swapchain extension when creating a VkDevice.
            createInfo.enabledExtensionCount = (uint32_t)g_deviceExtensions.size();
            createInfo.ppEnabledExtensionNames = g_deviceExtensions.data();
            createInfo.enabledLayerCount = 0;

            if (options->enableValidationLayers) {
                // To have device level validation information, the layers are added here.
                createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                createInfo.ppEnabledLayerNames = g_validationLayers.data();
            }
        }

        // 3.4. Create the logical device.
        if (vkCreateDevice(threadPhysicalDevice, &createInfo, NULL, &threadDevice) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
    }

    // T.4. Get the selected Queue family's first queue.
    // A Queue is used to issue recorded command buffers to the GPU for execution.
    VkQueue threadQueue;
    {
        vkGetDeviceQueue(threadDevice, threadGraphicsQueueFamilyIdx, 0, &threadQueue);
    }

    // T.5. Create a 256x256 2D Image to draw onto.
    // This will be the render target image.
    // Note: An Image by itself does not allocate memory on the GPU.
    uint32_t renderImageWidth = 256;
    uint32_t renderImageHeight = 256;
    VkFormat renderImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkImage renderImage;
    {
        // T.5.1. Create the required external memory structs
        VkExternalMemoryImageCreateInfoKHR externalInfo;
        {
            externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;
            externalInfo.pNext = NULL;
            // VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR for linux
            externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
        }

        // T.5.2. Specify the image creation information.
        VkImageCreateInfo imageInfo;
        {
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.pNext = &externalInfo;
            imageInfo.flags = 0;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = renderImageFormat;
            imageInfo.extent = { renderImageWidth, renderImageHeight,  1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            // Tiling optimal means that the image is in a GPU optimal mode.
            // Usually this means that it should not be accessed from the CPU side directly as
            // the image color channels can be in any order.
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            // Specifying the usage is important:
            // * VK_IMAGE_USAGE_TRANSFER_SRC_BIT: the image can be used as a source for a transfer/copy operation.
            // * VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: the image can be used as a color attachment (aka can render on it).
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.queueFamilyIndexCount = 0;
            imageInfo.pQueueFamilyIndices = NULL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // T.5.3. Create the image.
        if (vkCreateImage(threadDevice, &imageInfo, NULL, &renderImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create 2D image!");
        }
    }

    // T.6. Allocate and bind the memory for the render target image.
    // For each Image (or Buffer) a memory should be allocated on the GPU otherwise it can't be used.
    // To enable memory sharing the VkExportMemoryAllocateInfo struct is required.
    VkDeviceMemory renderImageMemory;
    {
        // T.6.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(threadDevice, renderImage, &memRequirements);

        // T.6.2 Find a memory type based on the requirements.
        // Here a device (gpu) local memory type is requested (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
        uint32_t memoryTypeIndex = FindMemoryType(threadPhysicalDevice,
                                                  memRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // T.6.3. Specify the memory export information
        VkMemoryDedicatedAllocateInfoKHR dedicatedInfo;
        {
            dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
            dedicatedInfo.pNext = NULL;
            dedicatedInfo.image = renderImage;
            dedicatedInfo.buffer = VK_NULL_HANDLE;
        }

        VkExportMemoryAllocateInfoKHR exportInfo;
        {
            exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
            exportInfo.pNext = &dedicatedInfo;
            exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
        }

        // T.6.4. Based on the memory requirements specify the allocation information.
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = &exportInfo;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // T.6.5 Allocate the memory.
        if (vkAllocateMemory(threadDevice, &allocInfo, NULL, &renderImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // T.6.6 "Connect" the image with the allocated memory.
        vkBindImageMemory(threadDevice, renderImage, renderImageMemory, 0);
    }

    // T.7. Get the file descriptor which can be shared with the other Vulkan Instance
    int imageFd = -1;
    {
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(threadDevice, "vkGetMemoryFdKHR");

        VkMemoryGetFdInfoKHR getFdInfo;
        {
            getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            getFdInfo.pNext = NULL;
            getFdInfo.memory = renderImageMemory;
            getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
        }

        if (vkGetMemoryFdKHR(threadDevice, &getFdInfo, &imageFd) != VK_SUCCESS) {
            throw std::runtime_error("unable to get image FD!");
        }
        printf("[thread] FD: %d\n", imageFd);
    }

    // TODO: sync
    {
        std::unique_lock<std::mutex> lock(options->syncMutex);
        options->exposedImageFd = imageFd;
        options->signal.notify_one(); // signal the othre side that the FD is ready
    }

    // T.8. Create an Image View for the Render Target Image.
    // Will be used by the Framebuffer as Color Attachment.
    VkImageView renderImageView;
    {
        // T.8.1. Specify the view information.
        VkImageViewCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.image = renderImage;
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = renderImageFormat;
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

        // T.8.2. Create the Image View.
        if (vkCreateImageView(threadDevice, &createInfo, NULL, &renderImageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }

    // T.9. Create a Render Pass.
    // A Render Pass is required to use vkCmdDraw* commands.
    VkRenderPass renderPass;
    {
        VkAttachmentDescription colorAttachment;
        {
            colorAttachment.flags = 0;
            colorAttachment.format = renderImageFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

        if (vkCreateRenderPass(threadDevice, &renderPassInfo, NULL, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    // T.10. Create Vertex shader.
    VkShaderModule vertShaderModule;
    {
        // T.10.1. Load the GLSL shader from file and compile it with shaderc.
        #if HAVE_SHADERC
        std::vector<char> vertSrc = LoadGLSL("passthrough.vert");
        std::vector<uint32_t> vertCode = CompileGLSL(shaderc_vertex_shader, vertSrc);
        #else
        std::vector<uint32_t> vertCode = LoadSPIRV("passthrough.vert.spv");
        #endif

        if (vertCode.size() == 0) {
            throw std::runtime_error("failed to load vertex shader!");
        }

        // T.10.2. Specify the vertex shader module information.
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

        // T.10.3. Create the Vertex Shader Module.
        if (vkCreateShaderModule(threadDevice, &vertInfo, NULL, &vertShaderModule) != VK_SUCCESS) {
           throw std::runtime_error("failed to create shader module!");
        }
    }

    // T.11. Create Fragment shader.
    VkShaderModule fragShaderModule;
    {
        // 10.1. Load the GLSL shader from file and compile it with shaderc.
        #if HAVE_SHADERC
        std::vector<char> fragSrc = LoadGLSL("passthrough.frag");
        std::vector<uint32_t> fragCode = CompileGLSL(shaderc_fragment_shader, fragSrc);
        #else
        std::vector<uint32_t> fragCode = LoadSPIRV("passthrough.frag.spv");
        #endif

        if (fragCode.size() == 0) {
            throw std::runtime_error("failed to load fragment shader!");
        }

        // T.11.2. Specify the fragment shader module information.
        VkShaderModuleCreateInfo fragInfo;
        {
            fragInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            fragInfo.pNext = NULL;
            fragInfo.flags = 0;
            fragInfo.codeSize = fragCode.size() * sizeof(uint32_t);
            fragInfo.pCode = reinterpret_cast<uint32_t*>(fragCode.data());
        }

        // T.11.3. Create the Fragment Shader Module.
        if (vkCreateShaderModule(threadDevice, &fragInfo, NULL, &fragShaderModule) != VK_SUCCESS) {
           throw std::runtime_error("failed to create shader module!");
        }
    }

    // T.12. Create Pipeline Layout.
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

        if (vkCreatePipelineLayout(threadDevice, &pipelineLayoutInfo, NULL, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
    }

    // T.13. Create the Rendering Pipeline
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

        VkPipelineVertexInputStateCreateInfo vertexInputInfo;
        {
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.pNext = NULL;
            vertexInputInfo.flags = 0;
            vertexInputInfo.vertexBindingDescriptionCount = 0;
            vertexInputInfo.pVertexBindingDescriptions = NULL;
            vertexInputInfo.vertexAttributeDescriptionCount = 0;
            vertexInputInfo.pVertexAttributeDescriptions = NULL;
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
            viewport.width = (float) renderImageWidth;
            viewport.height = (float) renderImageHeight;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
        }

        VkRect2D scissor;
        {
            scissor.offset = { 0, 0 };
            scissor.extent = { (uint32_t)viewport.width, (uint32_t)viewport.height };
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

        if (vkCreateGraphicsPipelines(threadDevice, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    // T.14. Create Framebuffer.
    // Frame buffer is the render target.
    VkFramebuffer framebuffer;
    {
        VkFramebufferCreateInfo framebufferInfo;
        {
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.pNext = NULL;
            framebufferInfo.flags = 0;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &renderImageView;
            framebufferInfo.width = renderImageWidth;
            framebufferInfo.height = renderImageHeight;
            framebufferInfo.layers = 1;
        }

        if (vkCreateFramebuffer(threadDevice, &framebufferInfo, NULL, &framebuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }

    // T.15. Create Command Pool.
    // Required to create Command buffers.
    VkCommandPool cmdPool;
    {
        VkCommandPoolCreateInfo poolInfo;
        {
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.pNext = NULL;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = threadGraphicsQueueFamilyIdx;
        }

        if (vkCreateCommandPool(threadDevice, &poolInfo, NULL, &cmdPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // T.16. Create Command Buffer to record draw commands.
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

        if (vkAllocateCommandBuffers(threadDevice, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // T.20. Create a Fence.
    // This Fence will be used to synchronize between CPU and GPU.
    // The Fence is created in an unsignaled state, thus no need to reset it.
    VkFence fence;
    {
        VkFenceCreateInfo fenceInfo;
        {
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.pNext = 0;
            fenceInfo.flags = 0;
        }

        if (vkCreateFence(threadDevice, &fenceInfo, NULL, &fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }

    // Start recording draw commands.

    std::unique_lock<std::mutex> syncLock(options->syncMutex);
    int counter = 0;
    while (options->signal.wait_for(syncLock, std::chrono::seconds(1)) == std::cv_status::timeout)
    {
        vkResetCommandBuffer(cmdBuffer, 0);

        // T.17. Start Command Buffer
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

        // T.18. Insert draw commands into Command Buffer.
        {
            // T.18.1. Add Begin RenderPass command
            // This makes it possible to use the vmCmdDraw* calls.
            VkClearValue clearColor = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
            VkRenderPassBeginInfo renderPassInfo;
            {
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.pNext = NULL;
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = framebuffer;
                renderPassInfo.renderArea.offset = { 0, 0 };
                renderPassInfo.renderArea.extent = { (uint32_t)renderImageWidth, (uint32_t)renderImageHeight };
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = &clearColor;
            }

            vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // T.18.2. Bind the Graphics pipeline inside the Current Render Pass.
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

            // T.18.3. Add a Draw command.
            // Draw 3 vertices using the pipeline bound previously.
            uint32_t vertexCount = 3;
            uint32_t instanceCount = 1;
            vkCmdDraw(cmdBuffer, vertexCount, instanceCount, 0, counter);
            counter = (counter + 1) % 3;

            // T.18.4. End the Render Pass.
            vkCmdEndRenderPass(cmdBuffer);
        }

        // T.19. End the Command Buffer recording.
        {
            if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }

        // T.21. Submit the recorded Command Buffer to the Queue.
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
            if (vkQueueSubmit(threadQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit command buffer!");
            }
        }

        // T.22. Wait the submitted Command Buffer to finish.
        {
            // -1 means to wait for ever to finish.
            if (vkWaitForFences(threadDevice, 1, &fence, VK_TRUE, -1) != VK_SUCCESS) {
                throw std::runtime_error("failed to wait for fence!");
            }
            vkResetFences(threadDevice, 1, &fence);
        }
    }


    {
        // Start readback process of the rendered image.
        // 22. Copy the rendered image into a buffer which can be mapped and read.
        // Note: this is the most basic process to the the image into a readeable memory.
        VkImage readableImage;
        VkDeviceMemory readableImageMemory;
        CopyImageToLinearImage(threadPhysicalDevice,
                               threadDevice,
                               threadQueue,
                               cmdPool,
                               renderImage,
                               renderImageWidth, renderImageHeight,
                               &readableImage,
                               &readableImageMemory);

        // 23. Get layout of the readable image (including row pitch).
        VkImageSubresource subResource;
        {
            subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subResource.mipLevel = 0;
            subResource.arrayLayer = 0;
        }
        VkSubresourceLayout subResourceLayout;
        vkGetImageSubresourceLayout(threadDevice, readableImage, &subResource, &subResourceLayout);

        // 24. Map image memory so we can start copying from it.
        const uint8_t* data;
        {
            vkMapMemory(threadDevice, readableImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
            data += subResourceLayout.offset;
        }

        // 25. Write out the image to a ppm file.
        {
            std::ofstream file("thread_out.ppm", std::ios::out | std::ios::binary);
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
        vkUnmapMemory(threadDevice, readableImageMemory);

        // XX. Free linear image's memory.
        vkFreeMemory(threadDevice, readableImageMemory, NULL);

        // XX. Destory linar image memory.
        vkDestroyImage(threadDevice, readableImage, NULL);
    }
    printf("written out the image\n");


    // T.X. Release resources
    vkFreeCommandBuffers(threadDevice, cmdPool, 1, &cmdBuffer);
    vkDestroyCommandPool(threadDevice, cmdPool, NULL);

    vkDestroyFence(threadDevice, fence, NULL);

    vkDestroyShaderModule(threadDevice, vertShaderModule, NULL);
    vkDestroyShaderModule(threadDevice, fragShaderModule, NULL);

    vkDestroyPipelineLayout(threadDevice, pipelineLayout, NULL);
    vkDestroyPipeline(threadDevice, pipeline, NULL);
    vkDestroyRenderPass(threadDevice, renderPass, NULL);
    vkDestroyFramebuffer(threadDevice, framebuffer, NULL);

    vkDestroyImageView(threadDevice, renderImageView, NULL);
    vkFreeMemory(threadDevice, renderImageMemory, NULL);
    vkDestroyImage(threadDevice, renderImage, NULL);
    vkDestroyDevice(threadDevice, NULL);
    vkDestroyInstance(threadInstance, NULL);

    //pthread_exit(NULL);
    return NULL;
}

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

    // T.X.
    VulkanThreadOptions threadOptions;
    {
        threadOptions.enableValidationLayers = enableValidationLayers;
        threadOptions.exposedImageFd = -1;
    }

    std::thread renderThread = std::thread(VulkanImageProducerThread, &threadOptions);

    // G.0. Initialize GLFW.
    {
        glfwInit();

        printf("GLFW Vulkan supported: %s\n", (glfwVulkanSupported() ? "YES" : "NO"));
    }

    // G.1. Create a Window.
    uint32_t windowWidth = 1024;
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
        std::vector<const char*> extensions = g_instanceExtensions;

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
            appInfo.pNext = NULL;
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
        std::vector<const char*> deviceExtensions = g_swapchainDeviceExtension;
        for (const char* extension : g_deviceExtensions) {
            deviceExtensions.push_back(extension);
        }

        VkDeviceCreateInfo createInfo;
        {
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.queueCreateInfoCount = 1;
            createInfo.pQueueCreateInfos = &queueCreateInfo;
            createInfo.pEnabledFeatures = NULL;
            // G.4. Specify the swapchain extension when creating a VkDevice.
            createInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();
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
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

    // T.XX. Wait for the other side to provide the image FD
    {
        std::unique_lock<std::mutex> lock(threadOptions.syncMutex);
        printf("Waiting for FD\n");
        if (threadOptions.exposedImageFd == -1) {
            threadOptions.signal.wait(lock);
        }
        printf("Waiting for done: %d\n", threadOptions.exposedImageFd);
    }
    int importedImageFd = threadOptions.exposedImageFd;

    uint32_t importedImageWidth = 256;
    uint32_t importedImageHeight = 256;
    VkFormat importedImageFormat = VK_FORMAT_R8G8B8A8_UNORM;

    VkImage importedImage;
    {
        // T.5.1. Create the required external memory structs
        VkExternalMemoryImageCreateInfoKHR externalInfo;
        {
            externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;
            externalInfo.pNext = NULL;
            // VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR for linux
            externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
        }

        // T.5.2. Specify the image creation information.
        VkImageCreateInfo imageInfo;
        {
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.pNext = &externalInfo;
            imageInfo.flags = 0;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = importedImageFormat;
            imageInfo.extent = { importedImageWidth, importedImageHeight,  1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            // Tiling optimal means that the image is in a GPU optimal mode.
            // Usually this means that it should not be accessed from the CPU side directly as
            // the image color channels can be in any order.
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            // Specifying the usage is important:
            // * VK_IMAGE_USAGE_TRANSFER_SRC_BIT: the image can be used as a source for a transfer/copy operation.
            // * VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: the image can be used as a color attachment (aka can render on it).
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.queueFamilyIndexCount = 0;
            imageInfo.pQueueFamilyIndices = NULL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // T.5.3. Create the image.
        if (vkCreateImage(device, &imageInfo, NULL, &importedImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create 2D image!");
        }
    }

    // T.6. Allocate and bind the memory for the render target image.
    // For each Image (or Buffer) a memory should be allocated on the GPU otherwise it can't be used.
    // To enable memory sharing the VkExportMemoryAllocateInfo struct is required.
    VkDeviceMemory importedImageMemory;
    {
        // T.6.1 Query the memory requirments for the image.
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, importedImage, &memRequirements);

        // T.6.2 Find a memory type based on the requirements.
        // Here a device (gpu) local memory type is requested (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT).
        uint32_t memoryTypeIndex = FindMemoryType(physicalDevice,
                                                  memRequirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // T.6.3. Specify the memory import information
        VkMemoryDedicatedAllocateInfoKHR dedicatedInfo;
        {
            dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
            dedicatedInfo.pNext = NULL;
            dedicatedInfo.image = importedImage;
            dedicatedInfo.buffer = VK_NULL_HANDLE;
        }

        VkImportMemoryFdInfoKHR importInfo;
        {
            importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
            importInfo.pNext = &dedicatedInfo;
            importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
            importInfo.fd = importedImageFd;
        }

        // T.6.4. Based on the memory requirements specify the allocation information.
        VkMemoryAllocateInfo allocInfo;
        {
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.pNext = &importInfo;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = memoryTypeIndex;
        }

        // T.6.5 Allocate the memory.
        if (vkAllocateMemory(device, &allocInfo, NULL, &importedImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        // T.6.6 "Connect" the image with the allocated memory.
        VkResult r = vkBindImageMemory(device, importedImage, importedImageMemory, 0);
        printf("import Bind result: %d :%d\n", r == VK_SUCCESS, r);
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
        VkImageMemoryBarrier baseStartBarrier;
        {
            baseStartBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            baseStartBarrier.pNext = NULL;
            //startBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            //startBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            baseStartBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            //startBarrier.newLayout =
            baseStartBarrier.srcQueueFamilyIndex = graphicsQueueFamilyIdx;
            baseStartBarrier.dstQueueFamilyIndex = graphicsQueueFamilyIdx;
            //startBarrier.image = ..
            baseStartBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }

        VkImageMemoryBarrier importedImageBarrier = baseStartBarrier;
        {
            importedImageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            importedImageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            importedImageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            importedImageBarrier.image = importedImage;
        }

        VkImageMemoryBarrier presentImageStartBarrier = baseStartBarrier;
        {
            presentImageStartBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            presentImageStartBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            presentImageStartBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            presentImageStartBarrier.image = swapImages[idx];
        }

        VkImageMemoryBarrier startBarriers[2] = { importedImageBarrier, presentImageStartBarrier };

        vkCmdPipelineBarrier(cmdBuffers[idx], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, NULL, // memory barriers
                             0, NULL, // buffer barriers
                             2, startBarriers);

        VkImageBlit blitRegion;
        {
            blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitRegion.srcOffsets[0] = { 0, 0, 0 };
            blitRegion.srcOffsets[1] = { (int32_t)importedImageWidth, (int32_t)importedImageHeight, 1 };
            blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitRegion.dstOffsets[0] = { 0, 0, 0 };
            blitRegion.dstOffsets[1] = { (int32_t)swapExtent.width, (int32_t)swapExtent.height, 1 };
        }

        vkCmdBlitImage(cmdBuffers[idx], importedImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);
        // TODO: swapImage image transition to present src khr

        VkImageMemoryBarrier presentImageEndBarrier = presentImageStartBarrier;
        {
            presentImageEndBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            presentImageEndBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            presentImageEndBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            presentImageEndBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            presentImageEndBarrier.image = swapImages[idx];
        }
        VkImageMemoryBarrier endBarriers[1] = { presentImageEndBarrier };
        vkCmdPipelineBarrier(cmdBuffers[idx], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, NULL, // memory barriers
                             0, NULL, // buffer barriers
                             1, endBarriers);

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
        //usleep((int)(0.15 * 1000000));
    }

    // At this point the image is rendered into the Framebuffer's attachment which is an ImageView.

    printf("--- getting last image\n");
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

    vkDestroyImage(device, importedImage, NULL);
    vkFreeMemory(device, importedImageMemory, NULL);

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


    printf("waiting for render thread end\n");
    {
        std::unique_lock<std::mutex> lock(threadOptions.syncMutex);
        threadOptions.signal.notify_one(); // signal the othre side that the FD is ready
    }
    renderThread.join();


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

            if (surface != VK_NULL_HANDLE) {
                // G.XX. Check if the selected graphics queue family supports presentation.
                // At the moment the example expects that the graphics and presentation queue is the same.
                // This is not always the case.
                // TODO: add support for different graphics and presentation family indices.
                VkBool32 presentSupport;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, queueFamilyIdx, surface, &presentSupport);
                if (presentSupport) {
                    return queueFamilyIdx;
                }
            } else {
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
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; //VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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
